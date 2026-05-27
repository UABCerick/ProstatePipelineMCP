"""
server.py — Servidor MCP principal: ProstatePipelineMCP

Fase 1/6: Tools de carga y validación DICOM + Resource de registro de casos.

Arquitectura:
  - FastMCP como framework del servidor
  - Toda entrada validada por Pydantic antes de ejecutarse
  - SlicerBridge para comunicación con 3D Slicer
  - CaseRegistry como estado persistente

Ejecutar:
    python -m mcp_server.server
    o bien:
    fastmcp run mcp_server/server.py
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP
from loguru import logger
from pydantic import ValidationError

from .case_registry import CaseRegistry
from .dicom_utils import (
    extract_metadata,
    scan_dicom_directory,
    validate_series_for_registration,
)
from .schemas import (
    GetMetadataInput,
    ImageMetadata,
    LoadDicomInput,
    LoadDicomResult,
    ValidateSeriesInput,
    ValidationResult,
)
from .slicer_bridge import SlicerBridge, SlicerConnectionError
from .schemas_phase2 import DetectProstateInput, DetectionResult, ImageModalityDetect
from .dataset_manager import DatasetManager
from .registration_engine import run_registration, compute_registration_metrics, DICE_THRESHOLD, HD95_THRESHOLD_MM
from .raw_patient_manager import RawPatientManager

from .coarse_localizer import (
    CoarseCNN, load_coarse_model, preprocess_raw_volume,
    run_coarse_inference, crop_fine_volume, save_prob_heatmap_nifti,
    PATCH_SIZE, DST_SPACING, CROP_SIZE,
)
from .preprocessor import preprocess_and_save
from .localizer_model import (
    load_localizer, preprocess_volume, run_inference,
    extract_centroid, centroid_to_global_mm, voxel_to_ras,
    HALF_FOV_MM,
)

load_dotenv()

# ── Inicialización ────────────────────────────────────────────────────────────

mcp = FastMCP("ProstatePipelineMCP")

_registry = CaseRegistry(
    registry_path=os.getenv("CASE_REGISTRY_PATH", "./data/case_registry.json")
)
_slicer = SlicerBridge(
    host=os.getenv("SLICER_HOST", "localhost"),
    port=int(os.getenv("SLICER_PORT", "2016")),
)

# ── Modelos de detección (cargados una vez al arrancar) ───────────────────────
# Se cargan lazy en el primer uso para no bloquear el arranque del servidor.
_models: dict = {}   # {"MRI": (model, device), "TRUS": (model, device)}

# ── Dataset Manager (cargado lazy) ───────────────────────────────────────────
_dataset_manager: "DatasetManager | None" = None

def _get_dataset() -> "DatasetManager":
    """Carga el dataset index la primera vez que se necesita. O(n) primera vez."""
    global _dataset_manager
    if _dataset_manager is None:
        index_path = os.getenv(
            "DATASET_INDEX_PATH",
            r"C:\Codes\Us_MRI_Fusion\TCIA\CLEAN_COUD_P_REGISTER\Preprocessed"
            r"\Localizer_160_0p565\dataset_index_160_localizer_641.json"
        )
        _dataset_manager = DatasetManager(index_path)
    return _dataset_manager


# ── Modelos CoarseCNN (cargados lazy) ────────────────────────────────────────
_coarse_models: dict = {}

COARSE_MODEL_PATHS = {
    "MRI":  r"C:\Codes\Us_MRI_Fusion\PRE-TCIA\Training\step1_coarse\MRI\models\coarse_MRI_fold0_best.pth",
    "TRUS": r"C:\Codes\Us_MRI_Fusion\PRE-TCIA\Training\step1_coarse\TRUS\models\coarse_TRUS_fold0_best.pth",
}

# ── RawPatientManager (stage1_selected_pairs.json) ───────────────────────────
_raw_manager = RawPatientManager(
    json_path=os.getenv("STAGE1_JSON_PATH", r"C:\Codes\Us_MRI_Fusion\PRE-TCIA\Preprocessing\pipeline_v3\stage1_selected_pairs.json"),
    preproc_dir=os.getenv("PREPROC_DIR", ""),
)


def _get_coarse_model(modality: str):
    """Carga el CoarseCNN en el primer uso. O(P) primera vez, O(1) después."""
    if modality not in _coarse_models:
        path = os.getenv(
            f"COARSE_{modality}_PATH",
            COARSE_MODEL_PATHS.get(modality, "")
        )
        if not path:
            raise ValueError(f"Ruta del modelo coarse {modality} no configurada.")
        model, device = load_coarse_model(path)
        _coarse_models[modality] = (model, device)
        logger.info(f"CoarseCNN {modality} cargado en {device}")
    return _coarse_models[modality]


def _get_model(modality: str):
    """Carga el modelo en el primer uso. O(P) primera vez, O(1) después."""
    if modality not in _models:
        if modality == "MRI":
            path = os.getenv("MODEL_MRI_PATH", "")
        else:
            path = os.getenv("MODEL_TRUS_PATH", "")

        if not path:
            raise ValueError(
                f"Ruta del modelo {modality} no configurada. "
                f"Agrega MODEL_{modality}_PATH en tu .env"
            )
        model, device = load_localizer(path)
        _models[modality] = (model, device)
        logger.info(f"Modelo {modality} cargado en {device}")

    return _models[modality]


# ── TOOLS ─────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="load_dicom",
    description=(
        "Carga un directorio DICOM de próstata (MRI o US) y registra el caso. "
        "Escanea todas las series presentes y retorna su información estructurada. "
        "Primer paso obligatorio del pipeline."
    ),
)
async def load_dicom(
    dicom_path: str,
    modality: str = "UNKNOWN",
    case_id: str | None = None,
) -> dict[str, Any]:
    """
    Carga y registra un directorio DICOM.

    Complejidad: O(n) donde n = número de archivos DICOM.
    La lectura es solo de headers (stop_before_pixels=True).
    """
    # ── Validación de entrada (anti-injection) ────────────────────────────────
    try:
        validated = LoadDicomInput(
            dicom_path=dicom_path,
            modality=modality,
            case_id=case_id,
        )
    except ValidationError as e:
        logger.warning(f"Input inválido en load_dicom: {e}")
        return {"success": False, "error": f"Input inválido: {e.errors()[0]['msg']}"}

    # ── Lógica principal ──────────────────────────────────────────────────────
    try:
        series_list = scan_dicom_directory(validated.dicom_path)

        if not series_list:
            return LoadDicomResult(
                success=False,
                case_id="",
                dicom_path=validated.dicom_path,
                series_found=[],
                error="No se encontraron series DICOM válidas en el directorio.",
            ).model_dump()

        # Registrar caso
        cid = _registry.register_case(
            dicom_path=validated.dicom_path,
            case_id=validated.case_id,
        )

        # Guardar series en el registry
        for s in series_list:
            _registry.update_series(cid, s.series_uid, s.model_dump())

        # Cargar en Slicer si está disponible (no bloquea si Slicer no está abierto)
        slicer_loaded = False
        slicer_warning = None
        if await _slicer.is_alive():
            try:
                await _slicer.load_dicom_volume(validated.dicom_path)
                slicer_loaded = True
                logger.info(f"Volumen cargado en 3D Slicer para caso {cid}")
            except Exception as e:
                slicer_warning = f"Slicer disponible pero falló la carga: {e}"
                logger.warning(slicer_warning)
        else:
            slicer_warning = "3D Slicer no disponible. Continuando sin visualización."
            logger.info(slicer_warning)

        warnings = [slicer_warning] if slicer_warning else []

        result = LoadDicomResult(
            success=True,
            case_id=cid,
            dicom_path=validated.dicom_path,
            series_found=series_list,
            warnings=warnings,
        )
        logger.info(
            f"load_dicom OK | caso={cid} | series={len(series_list)} | slicer={slicer_loaded}"
        )
        return result.model_dump()

    except FileNotFoundError as e:
        return LoadDicomResult(
            success=False, case_id="", dicom_path=dicom_path,
            series_found=[], error=str(e),
        ).model_dump()
    except Exception as e:
        logger.error(f"Error inesperado en load_dicom: {e}", exc_info=True)
        return {"success": False, "error": f"Error interno: {type(e).__name__}: {e}"}


@mcp.tool(
    name="validate_series",
    description=(
        "Valida si una serie DICOM específica cumple los criterios de calidad "
        "para registro MRI-US prostático (PI-RADS v2.1). "
        "Retorna quality: ACCEPTABLE | MARGINAL | REJECTED con justificación clínica. "
        "Debe llamarse después de load_dicom."
    ),
)
async def validate_series(
    case_id: str,
    series_uid: str,
    expected_modality: str,
    expected_sequence: str | None = None,
    min_slices: int = 10,
) -> dict[str, Any]:
    """
    Valida una serie DICOM contra criterios clínicos.

    Complejidad: O(1) — solo comparaciones sobre metadatos ya cargados.
    """
    # ── Validación de entrada ─────────────────────────────────────────────────
    try:
        validated = ValidateSeriesInput(
            case_id=case_id,
            series_uid=series_uid,
            expected_modality=expected_modality,
            expected_sequence=expected_sequence,
            min_slices=min_slices,
        )
    except ValidationError as e:
        return {"success": False, "error": f"Input inválido: {e.errors()[0]['msg']}"}

    # ── Verificar que el caso existe ──────────────────────────────────────────
    case = _registry.get_case(validated.case_id)
    if not case:
        return {
            "success": False,
            "error": f"Caso '{validated.case_id}' no encontrado. Llama load_dicom primero."
        }

    series_data = case.get("series", {}).get(validated.series_uid)
    if not series_data:
        return {
            "success": False,
            "error": f"Serie '{validated.series_uid}' no encontrada en el caso."
        }

    # ── Lógica de validación ──────────────────────────────────────────────────
    from .schemas import DicomSeriesInfo
    series_info = DicomSeriesInfo(**series_data)

    result = validate_series_for_registration(
        series_info=series_info,
        expected_modality=validated.expected_modality,
        expected_sequence=validated.expected_sequence,
        min_slices=validated.min_slices,
    )
    result.case_id = validated.case_id

    logger.info(
        f"validate_series | caso={case_id} | calidad={result.quality.value} | "
        f"issues={len(result.issues)}"
    )
    return result.model_dump()


@mcp.tool(
    name="get_image_metadata",
    description=(
        "Extrae metadatos DICOM clínicamente relevantes de una serie. "
        "Solo permite tags de la whitelist clínica (no expone datos sensibles). "
        "Útil para que el LLM razone sobre las características de la imagen."
    ),
)
async def get_image_metadata(
    case_id: str,
    series_uid: str,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """
    Extrae metadatos DICOM específicos.

    Complejidad: O(n_files) para encontrar la serie + O(t) para extraer tags.
    """
    default_fields = [
        "PatientID", "Modality", "SeriesDescription",
        "SliceThickness", "PixelSpacing", "Rows", "Columns",
        "StudyDate", "MagneticFieldStrength", "ProtocolName",
    ]

    try:
        validated = GetMetadataInput(
            case_id=case_id,
            series_uid=series_uid,
            fields=fields or default_fields,
        )
    except ValidationError as e:
        return {"success": False, "error": f"Input inválido: {e.errors()[0]['msg']}"}

    case = _registry.get_case(validated.case_id)
    if not case:
        return {"success": False, "error": f"Caso '{case_id}' no encontrado."}

    try:
        tags = extract_metadata(
            dicom_path=case["dicom_path"],
            series_uid=validated.series_uid,
            fields=validated.fields,
        )
        result = ImageMetadata(
            case_id=validated.case_id,
            series_uid=validated.series_uid,
            tags=tags,
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(f"get_image_metadata OK | caso={case_id} | tags={len(tags)}")
        return result.model_dump()

    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error en get_image_metadata: {e}", exc_info=True)
        return {"success": False, "error": f"Error interno: {e}"}


@mcp.tool(
    name="check_slicer_connection",
    description=(
        "Verifica si 3D Slicer está disponible y mcp-slicer activo. "
        "Retorna estado de conexión y lista de nodos en la escena actual."
    ),
)
async def check_slicer_connection() -> dict[str, Any]:
    """Diagnóstico de conexión con 3D Slicer. O(1) red."""
    alive = await _slicer.is_alive()
    result: dict[str, Any] = {
        "slicer_available": alive,
        "slicer_url": _slicer._base_url,
    }
    if alive:
        try:
            nodes = await _slicer.get_scene_nodes()
            result["scene_nodes"] = nodes
            result["node_count"] = len(nodes)
            result["message"] = "3D Slicer conectado y operativo."
        except Exception as e:
            result["message"] = f"Slicer responde pero hay error: {e}"
    else:
        result["message"] = (
            "3D Slicer no disponible. Abre Slicer 5.8+ con el módulo mcp-slicer activo."
        )
    return result


# ── TOOL: list_dataset_cases ─────────────────────────────────────────────────

@mcp.tool(
    name="list_dataset_cases",
    description=(
        "Lista los casos disponibles en el dataset preprocesado (Localizer_160_0p565). "
        "Lee el índice JSON y retorna PIDs, usabilidad y calidad de cada caso. "
        "Primer paso para seleccionar un caso a analizar."
    ),
)
async def list_dataset_cases(
    modality: str = "MRI",
    only_usable: bool = True,
) -> dict:
    """Lista casos del dataset. O(n) donde n = casos en el índice."""
    try:
        ds = _get_dataset()
        cases = ds.list_cases(modality=modality, only_usable=only_usable)
        return {
            "success": True,
            "modality": modality,
            "total": len(cases),
            "cases": [c.summary() for c in cases],
            "dataset_info": ds.to_resource_dict(),
        }
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error en list_dataset_cases: {e}")
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


@mcp.tool(
    name="get_case_info",
    description=(
        "Retorna todos los metadatos de un caso específico del dataset: "
        "rutas, ground truth de centroides, parámetros de normalización, "
        "calidad (min_margin_vox) y usabilidad por modalidad."
    ),
)
async def get_case_info(pid: str) -> dict:
    """Metadatos completos de un caso. O(1)."""
    import re
    if not re.match(r'^[a-zA-Z0-9_\-]{1,20}$', pid):
        return {"success": False, "error": "PID inválido."}
    try:
        ds   = _get_dataset()
        case = ds.get_case(pid)
        if case is None:
            pids = ds.list_pids()
            return {
                "success": False,
                "error": f"Caso '{pid}' no encontrado.",
                "available_pids": pids[:10],
            }
        from dataclasses import asdict
        return {"success": True, "case": asdict(case)}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


# ── TOOL: preprocess_volume ───────────────────────────────────────────────────

@mcp.tool(
    name="preprocess_volume",
    description=(
        "Preprocesa un volumen DICOM (MRI) o NIfTI (TRUS) crudo y guarda "
        "el resultado normalizado listo para el LocalizerCNN. "
        "Reproduce el Script 1 del doctorado: resampleo 0.565mm → LPS → "
        "crop 160³ desde centro FOV → z-score. "
        "Úsalo cuando tengas datos crudos que aún no están en el dataset preprocesado."
    ),
)
async def preprocess_volume_tool(
    input_path: str,
    output_dir: str,
    pid: str,
    modality: str = "MRI",
    overwrite: bool = False,
) -> dict:
    """
    Preprocesa y guarda un volumen crudo. O(V_orig).
    V_orig típico: MRI ~3.9M voxeles, TRUS ~29M voxeles.
    """
    import re
    # Validar entradas
    if ".." in input_path or ".." in output_dir:
        return {"success": False, "error": "Path traversal detectado."}
    if not re.match(r'^[a-zA-Z0-9_\-]{1,20}$', pid):
        return {"success": False, "error": "PID inválido."}
    if modality not in ("MRI", "TRUS"):
        return {"success": False, "error": "Modalidad debe ser MRI o TRUS."}

    is_dicom = modality == "MRI"
    logger.info(f"Preprocesando {pid}/{modality} desde {input_path}")

    result = preprocess_and_save(
        input_path=input_path,
        output_dir=output_dir,
        pid=pid,
        modality=modality,
        is_dicom=is_dicom,
        overwrite=overwrite,
    )
    return result


# ── TOOL: load_nifti_in_slicer ───────────────────────────────────────────────

@mcp.tool(
    name="load_nifti_in_slicer",
    description=(
        "Genera el comando para cargar un NIfTI en 3D Slicer. "
        "Debido a limitaciones del Web Server de Slicer 5.10, retorna el script "
        "Python listo para ejecutar en la consola de Slicer (View > Python Interactor)."
    ),
)
async def load_nifti_in_slicer(
    file_path: str,
    node_name: str = "",
) -> dict:
    """
    Genera comando de carga para Slicer. O(1).
    
    Slicer 5.10 limitacion: el Web Server ejecuta en hilo secundario
    y loadVolume requiere el hilo principal de Qt.
    Solucion: retornar el script Python para ejecutar en la consola de Slicer.
    """
    import re as _re
    if ".." in file_path:
        return {"success": False, "error": "Path traversal detectado."}

    safe_path = file_path.replace("\\", "/").replace("/", "/")
    label = node_name or safe_path.split("/")[-1].replace(".nii.gz", "").replace(".nii", "")

    # Script listo para pegar en la consola Python de Slicer
    # Usar saltos de línea reales para que Gradio lo muestre correctamente
    slicer_script = "\n".join([
        "import slicer",
        f'node = slicer.util.loadVolume(r"{safe_path}")',
        f'node.SetName("{label}")',
        "slicer.util.resetSliceViews()",
        "print('Cargado:', node.GetName())",
    ])


    # Intentar igual via exec (por si en alguna sesion funciona)
    slicer_available = await _slicer.is_alive()
    exec_attempted = False
    if slicer_available:
        code = f"""
import slicer
try:
    node = slicer.util.loadVolume(r"{safe_path}")
    if node:
        node.SetName("{label}")
        slicer.util.resetSliceViews()
except:
    pass
"""
        await _slicer.execute_python(code)
        exec_attempted = True

    return {
        "success": True,
        "file_path": file_path,
        "node_name": label,
        "exec_attempted": exec_attempted,
        "slicer_script": slicer_script,
        "instructions": (
            "IMPORTANTE: Copia el script de 'slicer_script' y pegalo en "
            "3D Slicer > View > Python Interactor > Enter. "
            "Esto carga el volumen directamente en el hilo principal de Slicer."
        ),
        "shortcut": "En Slicer: Ctrl+3 abre el Python Interactor",
    }


# ── TOOL: detect_prostate_center ─────────────────────────────────────────────# ── TOOL: detect_prostate_center ─────────────────────────────────────────────

@mcp.tool(
    name="detect_prostate_center",
    description=(
        "Detecta el centro de la próstata en una imagen MRI o TRUS usando el "
        "LocalizerUNet del doctorado. Retorna el centroide en coordenadas de voxel, "
        "mm globales, y RAS para Slicer. Siempre requiere confirmación HITL antes "
        "de proceder al registro."
    ),
)
async def detect_prostate_center(
    image_path: str,
    modality: str,
    case_id: str | None = None,
    pid: str | None = None,
    threshold: float = 0.01,
    visualize_in_slicer: bool = True,
) -> dict:
    # Si se pasa un PID del dataset, usar el volumen preprocesado directamente
    if pid:
        try:
            ds = _get_dataset()
            tensor, case_entry = ds.get_ready_tensor(pid, modality)
            # Obtener crop_origin del dataset (ya calculado en el Script 1)
            import numpy as np
            from .localizer_model import extract_centroid, centroid_to_global_mm, voxel_to_ras, HALF_FOV_MM
            model, device = _get_model(modality)
            heatmap = run_inference(model, tensor, device)
            c_vox, confidence = extract_centroid(heatmap, threshold=threshold)
            crop_origin = np.array(case_entry.crop_origin(modality))
            center_global_mm, crop_fine_mm = centroid_to_global_mm(c_vox, crop_origin)
            center_ras = voxel_to_ras(center_global_mm, None)
            # Calcular error vs ground truth
            error = ds.compute_error_mm(pid, modality, c_vox.tolist())
            result = DetectionResult(
                success=True,
                case_id=case_id or pid,
                modality=modality,
                centroid_vox=c_vox.tolist(),
                centroid_global_mm=center_global_mm.tolist(),
                centroid_ras=list(center_ras),
                crop_fine_mm=crop_fine_mm.tolist(),
                confidence=round(confidence, 4),
                fov_radius_mm=HALF_FOV_MM,
                hitl_required=True,
                device_used=str(device),
            )
            conf_pct = round(confidence * 100, 1)
            ras_fmt  = [round(v, 1) for v in center_ras]
            result.hitl_message = (
                f"PID {pid} | Centro en RAS {ras_fmt} mm | "
                f"Confianza: {conf_pct}% | "
                f"Error vs GT: {error.get('error_mm', 'N/A')} mm | "
                f"Dentro de umbral 10mm: {error.get('within_threshold', 'N/A')}. "
                f"Verifica el FOV en Slicer y aprueba para continuar."
            )
            result.preprocessing_notes = [
                f"Volumen cargado desde dataset preprocesado: {case_entry.volume_path(modality)}",
                f"crop_origin_mm del dataset: {crop_origin.tolist()}",
                f"Ground truth error: {error}",
            ]
            fov_label = f"FOV_{modality}_{case_id or pid}"
            # Generar script FOV siempre, independiente de si Slicer está disponible
            try:
                fov_result = await _slicer.show_fov_sphere(
                    center_ras=center_ras,
                    radius_mm=HALF_FOV_MM,
                    label=fov_label,
                    heatmap_path=heatmap_path_pid if "heatmap_path_pid" in dir() else "",
                )
                result.slicer_fov_script = fov_result.get("slicer_script", "")
                result.slicer_visualization = bool(result.slicer_fov_script)
            except Exception as e:
                logger.warning(f"FOV script error: {e}")
                # Generar script básico como fallback
                cx, cy, cz = center_ras
                result.slicer_fov_script = chr(10).join([
                    "import slicer",
                    f'fid = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "{fov_label}")',
                    f"fid.AddControlPoint({cx:.4f}, {cy:.4f}, {cz:.4f})",
                    "fid.GetDisplayNode().SetSelectedColor(0.2, 1.0, 0.3)",
                    "fid.GetDisplayNode().SetGlyphScale(3.0)",
                    "slicer.util.resetSliceViews()",
                    f'print("Centro: ({cx:.1f},{cy:.1f},{cz:.1f})")',
                ])
            return result.model_dump()
        except Exception as e:
            return DetectionResult(
                success=False,
                error=f"Error con PID {pid}: {type(e).__name__}: {e}"
            ).model_dump()
    # Sin PID: usar image_path normal (flujo original)
    """
    Pipeline completo: preprocesamiento → LocalizerUNet → centroide → FOV.

    Complejidad:
      preprocesamiento : O(V_orig) — V_orig = voxeles del volumen original
      inferencia       : O(V_crop) = O(160³) — forward pass de la red
      centroide        : O(160³)  — una pasada sobre el heatmap
      total            : O(V_orig) dominado por el resampleo

    Punto HITL #1: retorna siempre hitl_required=True.
    El agente LangGraph pausa aquí hasta que el investigador confirme.
    """
    # ── Validación Pydantic ───────────────────────────────────────────────────
    try:
        validated = DetectProstateInput(
            image_path=image_path,
            modality=modality,
            case_id=case_id,
            threshold=threshold,
            visualize_in_slicer=visualize_in_slicer,
        )
    except Exception as e:
        return DetectionResult(
            success=False,
            error=f"Input inválido: {e}"
        ).model_dump()

    result = DetectionResult(
        success=False,
        case_id=validated.case_id or "",
        modality=validated.modality.value,
    )

    try:
        # ── Cargar modelo (lazy) ──────────────────────────────────────────────
        model, device = _get_model(validated.modality.value)
        result.model_path = os.getenv(f"MODEL_{validated.modality.value}_PATH", "")
        result.device_used = str(device)

        # ── Preprocesamiento ──────────────────────────────────────────────────
        is_dicom = validated.modality == ImageModalityDetect.MRI
        logger.info(f"Preprocesando {validated.modality.value}: {image_path}")

        tensor, crop_origin_mm, img_iso = preprocess_volume(
            image_path=validated.image_path,
            is_dicom=is_dicom,
        )
        result.preprocessing_notes.append(
            f"Resampleo a {0.565}mm isotrópico completado."
        )
        result.preprocessing_notes.append(
            f"Crop 160³ desde centro FOV. crop_origin_mm={crop_origin_mm.tolist()}"
        )

        # ── Inferencia ────────────────────────────────────────────────────────
        logger.info(f"Inferencia LocalizerUNet en {device}...")
        heatmap = run_inference(model, tensor, device)


        # ── Guardar heatmap como NIfTI para visualización en Slicer ──────────
        heatmap_path = ""
        try:
            import os
            import SimpleITK as _sitk
            heatmap_dir = os.path.join(os.getenv("DATA_DIR", "./data"), "heatmaps")
            os.makedirs(heatmap_dir, exist_ok=True)
            pid_label = validated.case_id or "unknown"
            heatmap_path = os.path.join(
                heatmap_dir,
                f"{pid_label}_{validated.modality.value}_heatmap.nii.gz"
            )
            hmap_arr = heatmap[0, 0].astype("float32")
            hmap_img = _sitk.GetImageFromArray(hmap_arr)
            hmap_img.SetSpacing([0.565] * 3)
            _sitk.WriteImage(hmap_img, heatmap_path, useCompression=True)
            logger.info(f"Heatmap guardado: {heatmap_path}")
        except Exception as _he:
            logger.warning(f"No se pudo guardar heatmap: {_he}")
            heatmap_path = ""

        # ── Extracción del centroide ──────────────────────────────────────────
        c_vox, confidence = extract_centroid(heatmap, threshold=validated.threshold)
        center_global_mm, crop_fine_mm = centroid_to_global_mm(c_vox, crop_origin_mm)
        center_ras = voxel_to_ras(center_global_mm, img_iso)

        result.centroid_vox        = c_vox.tolist()
        result.centroid_global_mm  = center_global_mm.tolist()
        result.centroid_ras        = list(center_ras)
        result.crop_fine_mm        = crop_fine_mm.tolist()
        result.confidence          = round(confidence, 4)
        result.fov_radius_mm       = HALF_FOV_MM

        # ── Registrar en CaseRegistry ─────────────────────────────────────────
        if validated.case_id:
            if _registry.case_exists(validated.case_id):
                _registry.update_pipeline_state(validated.case_id, "detected")
            result.case_id = validated.case_id

        # ── Generar script FOV y visualizar en Slicer ───────────────────────────
        fov_label = f"FOV_{validated.modality.value}_{validated.case_id or 'case'}"
        try:
            fov_result = await _slicer.show_fov_sphere(
                center_ras=center_ras,
                radius_mm=HALF_FOV_MM,
                label=fov_label,
                heatmap_path=heatmap_path,
            )
            result.slicer_fov_script = fov_result.get("slicer_script", "")
            result.slicer_visualization = bool(result.slicer_fov_script)
        except Exception as e:
            logger.warning(f"FOV script error: {e}")
            cx, cy, cz = center_ras
            result.slicer_fov_script = chr(10).join([
                "import slicer",
                f'fid = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "{fov_label}")',
                f"fid.AddControlPoint({cx:.4f}, {cy:.4f}, {cz:.4f})",
                "fid.GetDisplayNode().SetSelectedColor(0.2, 1.0, 0.3)",
                "fid.GetDisplayNode().SetGlyphScale(3.0)",
                "slicer.util.resetSliceViews()",
            ])

        # ── Mensaje HITL ──────────────────────────────────────────────────────
        conf_pct = round(confidence * 100, 1)
        ras_fmt  = [round(v, 1) for v in center_ras]
        result.hitl_message = (
            f"Centro detectado en RAS {ras_fmt} mm | "
            f"Confianza: {conf_pct}% | "
            f"FOV radio: {HALF_FOV_MM}mm. "
            f"¿El FOV cubre correctamente la próstata en Slicer? "
            f"Aprueba para continuar al registro o rechaza para ajustar."
        )
        result.hitl_required = True
        result.success = True

        logger.info(
            f"detect_prostate_center OK | {validated.modality.value} | "
            f"confidence={confidence:.4f} | ras={ras_fmt}"
        )

    except FileNotFoundError as e:
        result.error = str(e)
        logger.error(f"Archivo no encontrado: {e}")
    except ValueError as e:
        result.error = str(e)
        logger.error(f"Error de configuración: {e}")
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        logger.error(f"Error en detect_prostate_center: {e}", exc_info=True)

    return result.model_dump()


# ── TOOL: run_registration ───────────────────────────────────────────────────

@mcp.tool(
    name="run_registration",
    description=(
        "Ejecuta registro deformable MRI-US usando SimpleElastix (ITK 6.0). "
        "Registro rígido inicial + B-spline deformable con parámetros del LLM. "
        "Calcula Dice, HD95 sobre máscaras reales. "
        "Requiere volúmenes preprocesados 160³ a 0.565mm del dataset."
    ),
)
async def run_registration_tool(
    pid: str,
    params_json: str,
    output_dir: str = "",
    initial_translation_json: str = "",
    fixed_path: str = "",
    moving_path: str = "",
    fixed_mask_path: str = "",
    moving_mask_path: str = "",
) -> dict:
    """
    Registro MRI-US real con SimpleElastix.
    Complejidad: O(V × I × G³) dominado por el registro deformable.
    V=160³, I=iteraciones, G=nodos del grid B-spline.
    """
    import re, json
    if not re.match(r'^[a-zA-Z0-9_\-]{1,20}$', pid):
        return {"success": False, "error": "PID inválido."}

    # Obtener rutas del dataset
    try:
        ds   = _get_dataset()
        case = ds.get_case(pid)
        if case is None:
            return {"success": False, "error": f"Caso {pid} no encontrado."}
    except Exception as e:
        return {"success": False, "error": str(e)}

    # Usar rutas provistas o fallback al dataset
    from pathlib import Path
    _fixed       = fixed_path       or case.mri_volume
    _moving      = moving_path      or case.trus_volume
    _fixed_mask  = fixed_mask_path  or case.mri_mask
    _moving_mask = moving_mask_path or case.trus_mask

    required = {
        "MRI volume":  _fixed,
        "TRUS volume": _moving,
        "MRI mask":    _fixed_mask,
        "TRUS mask":   _moving_mask,
    }
    for name, path in required.items():
        if not path or not Path(path).exists():
            return {"success": False, "error": f"{name} no encontrado: {path}"}

    # Parsear parámetros
    try:
        from langgraph_agent.state import RegistrationParams
        params_data = json.loads(params_json)
        params = RegistrationParams(**params_data)
    except Exception as e:
        return {"success": False, "error": f"Parámetros inválidos: {e}"}

    # Directorio de salida
    if not output_dir:
        import os
        output_dir = os.path.join(
            os.getenv("DATA_DIR", "./data"),
            "registrations", pid
        )

    logger.info(f"Iniciando registro real PID={pid} | grid={params.grid_spacing_mm}mm")

    # Ejecutar en thread para no bloquear el servidor MCP
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        # Parse initial translation from detection dual
        initial_translation = None
        if initial_translation_json:
            try:
                initial_translation = json.loads(initial_translation_json)
            except Exception:
                pass

        result = await loop.run_in_executor(
            pool,
            lambda: run_registration(
                fixed_path=_fixed,
                moving_path=_moving,
                moving_mask_path=_moving_mask,
                fixed_mask_path=_fixed_mask,
                params=params,
                output_dir=output_dir,
                initial_translation=initial_translation,
            )
        )

    if result["success"]:
        # Actualizar estado del caso (solo si existe en el registry)
        if _registry.case_exists(pid):
            _registry.update_pipeline_state(pid, "registered")
        logger.info(
            f"Registro OK PID={pid} | "
            f"Dice={result['dice']:.3f} | HD95={result['hd95_mm']:.2f}mm"
        )

    return result


# ── TOOL: get_raw_patient_info ───────────────────────────────────────────────

@mcp.tool(
    name="get_raw_patient_info",
    description=(
        "Retorna rutas crudas de un paciente desde stage1_selected_pairs.json. "
        "Incluye DICOM MRI, NIfTI TRUS exportado, y máscaras para registro."
    ),
)
async def get_raw_patient_info_tool(pid: str) -> dict:
    """
    Lookup O(1) — busca en este orden:
      1. custom_patients.json (pacientes registrados por el usuario)
      2. stage1_selected_pairs.json (dataset TCIA)
    """
    import re, json as _j
    from pathlib import Path as _P

    if not re.match(r'^[a-zA-Z0-9_\-]{1,20}$', pid):
        return {"success": False, "error": "PID inválido."}

    # ── Buscar en custom_patients.json primero ────────────────────────────
    data_dir      = os.getenv("DATA_DIR", "./data")
    if not os.path.isabs(data_dir):
        data_dir  = os.path.abspath(data_dir)
    custom_path   = os.path.join(data_dir, "custom_patients.json")
    if _P(custom_path).exists():
        try:
            custom = _j.loads(_P(custom_path).read_text())
            if pid in custom:
                entry = custom[pid]
                logger.info(f"[get_raw_patient_info] PID={pid} encontrado en custom_patients.json")
                return {
                    "success": True,
                    "pid":     pid,
                    "case": {
                        "_source":           "custom",
                        "mri_dicom_path":    entry.get("mri_dicom_path", ""),
                        "trus_nifti_path":   entry.get("trus_nifti_path", ""),
                        "mri_mask_path":     "",
                        "trus_mask_path":    "",
                        "mri_vol_iso_path":  "",
                        "trus_vol_iso_path": "",
                        "has_raw":           True,
                        "has_masks":         False,
                        "has_vol_iso":       False,
                        "stage1_ok":         True,
                        "stage1_flags":      [],
                        # Campos compatibles con el pipeline
                        "mri_volume":        "",
                        "trus_volume":       "",
                        "mri_mask":          "",
                        "trus_mask":         "",
                    }
                }
        except Exception as e:
            logger.warning(f"[get_raw_patient_info] Error leyendo custom_patients.json: {e}")

    # ── Buscar en stage1 JSON ─────────────────────────────────────────────
    case = _raw_manager.get_case(pid)
    if case is None:
        return {"success": False, "error": f"PID {pid} no encontrado en custom_patients.json ni en stage1 JSON."}

    return {
        "success":           True,
        "pid":               pid,
        "case": {
            "_source":           "stage1_raw",
            "mri_dicom_path":    case.mri_dicom_path,
            "trus_nifti_path":   case.trus_nifti_path,
            "mri_mask_path":     case.mri_mask_path,
            "trus_mask_path":    case.trus_mask_path,
            "mri_vol_iso_path":  case.mri_vol_iso_path,
            "trus_vol_iso_path": case.trus_vol_iso_path,
            "has_raw":           case.has_raw(),
            "has_masks":         case.has_masks(),
            "has_vol_iso":       case.has_vol_iso(),
            "mri_technique":     case.mri_technique,
            "mri_n_files":       case.mri_n_files,
            "mri_score":         case.mri_score,
            "trus_size_mb":      case.trus_size_mb,
            "stage1_ok":         case.stage1_ok,
            "stage1_flags":      case.stage1_flags,
            # Campos compatibles con el dataset Localizer (para el registro)
            "mri_volume":        case.mri_vol_iso_path,
            "trus_volume":       case.trus_vol_iso_path,
            "mri_mask":          case.mri_mask_path,
            "trus_mask":         case.trus_mask_path,
        }
    }


# ── TOOL: preprocess_raw_volume ──────────────────────────────────────────────

@mcp.tool(
    name="preprocess_raw_volume",
    description=(
        "Preprocesa un volumen crudo (DICOM MRI o NIfTI TRUS) para inferencia. "
        "Aplica: DICOMOrient(LPS) → origen(0,0,0) → resampleo 0.565mm isotrópico. "
        "Guarda vol_iso como NIfTI. No requiere máscaras ni STL."
    ),
)
async def preprocess_raw_volume_tool(
    input_path: str,
    modality: str = "MRI",
    output_dir: str = "",
    pid: str = "",
    is_dicom: bool = True,
) -> dict:
    """Preprocesa volumen crudo. O(V_orig) dominado por resampleo."""
    import re
    if ".." in input_path:
        return {"success": False, "error": "Path traversal detectado."}
    if modality not in ("MRI", "TRUS"):
        return {"success": False, "error": "Modalidad debe ser MRI o TRUS."}

    try:
        import SimpleITK as sitk
        # MRI siempre DICOM; TRUS siempre NIfTI exportado desde Slicer
        is_dicom = (modality == "MRI") if is_dicom else False
        vol_iso  = preprocess_raw_volume(input_path, is_dicom=is_dicom)

        # Guardar vol_iso con ruta absoluta
        import os
        if not output_dir:
            data_dir = os.getenv("DATA_DIR", "./data")
            if not os.path.isabs(data_dir):
                data_dir = os.path.abspath(data_dir)
            output_dir = os.path.join(data_dir, "preprocessed", pid or "patient")
        else:
            output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{pid or 'vol'}_{modality}_vol_iso.nii.gz")
        sitk.WriteImage(vol_iso, out_path, useCompression=True)

        size = vol_iso.GetSize()
        return {
            "success":      True,
            "vol_iso_path": out_path,
            "size":         list(size),
            "spacing":      list(vol_iso.GetSpacing()),
            "fov_mm":       [round(s * DST_SPACING, 1) for s in size],
        }
    except Exception as e:
        logger.error(f"Error en preprocess_raw: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── TOOL: detect_coarse_center ────────────────────────────────────────────────

@mcp.tool(
    name="detect_coarse_center",
    description=(
        "Localiza la próstata en un volumen isotrópico usando CoarseCNN. "
        "Divide el volumen en patches 32³, predice probabilidad de próstata "
        "por patch, y calcula el centroide ponderado. "
        "Funciona con cualquier volumen crudo — no requiere el dataset preprocesado. "
        "Guarda heatmap de probabilidades como NIfTI para visualización en Slicer."
    ),
)
async def detect_coarse_center_tool(
    vol_iso_path: str,
    modality: str = "MRI",
    output_dir: str = "",
    pid: str = "",
) -> dict:
    """
    Inferencia CoarseCNN sobre grid de patches.
    O(N_patches × 32³) donde N = número de patches del volumen.
    """
    import re, os
    if ".." in vol_iso_path:
        return {"success": False, "error": "Path traversal detectado."}
    if modality not in ("MRI", "TRUS"):
        return {"success": False, "error": "Modalidad debe ser MRI o TRUS."}
    from pathlib import Path as _Path
    if not _Path(vol_iso_path).exists():
        return {"success": False, "error": f"Archivo no encontrado: {vol_iso_path}"}

    try:
        import SimpleITK as sitk
        import numpy as np
        model, device = _get_coarse_model(modality)
        vol_iso = sitk.ReadImage(vol_iso_path)
        logger.info(f"Corriendo CoarseCNN {modality} sobre {_Path(vol_iso_path).name}")

        # Inferencia en thread para no bloquear el servidor
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            infer_result = await loop.run_in_executor(
                pool,
                lambda: run_coarse_inference(vol_iso, model, device)
            )

        centroid_mm = infer_result["centroid_mm"]
        max_prob    = infer_result["max_prob"]

        # Guardar heatmap con ruta absoluta
        if not output_dir:
            data_dir = os.getenv("DATA_DIR", "./data")
            if not os.path.isabs(data_dir):
                data_dir = os.path.abspath(data_dir)
            output_dir = os.path.join(data_dir, "coarse", pid or "patient")
        else:
            output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        heatmap_path = os.path.join(output_dir, f"{pid or 'vol'}_{modality}_coarse_heatmap.nii.gz")

        save_prob_heatmap_nifti(
            prob_map=np.array(infer_result["prob_map"]),
            grid_shape=infer_result["grid_shape"],
            orig_vol_iso=vol_iso,
            output_path=heatmap_path,
        )

        # Generar script Slicer para visualizar el heatmap
        safe_vol  = vol_iso_path.replace("\\\\", "/").replace("\\", "/")
        safe_heat = heatmap_path.replace("\\\\", "/").replace("\\", "/")
        cx, cy, cz = [round(v, 1) for v in centroid_mm]

        slicer_script = chr(10).join([
            "import slicer",
            "",
            "# ── Cargar vol_iso (origen ya en 0,0,0 por preprocesamiento) ──",
            f'vol = slicer.util.loadVolume(r"{safe_vol}")',
            f'vol.SetName("{pid}_{modality}_vol_iso")',
            "# Forzar origen 0,0,0 por si Slicer lo interpreta diferente",
            "vol.SetOrigin(0.0, 0.0, 0.0)",
            "",
            "# ── Cargar heatmap y colormap Rainbow (funciona en Slicer 5.10) ──",
            f'heat = slicer.util.loadVolume(r"{safe_heat}")',
            f'heat.SetName("Coarse_Heatmap_{pid}")',
            "heat.SetOrigin(0.0, 0.0, 0.0)",
            "dn = heat.GetDisplayNode()",
            'dn.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")',
            "dn.SetAutoWindowLevel(False)",
            "dn.SetWindowLevelMinMax(0.05, 1.0)",
            "dn.SetOpacity(0.7)",
            "slicer.util.setSliceViewerLayers(background=vol, foreground=heat, foregroundOpacity=0.5)",
            "",
            "# ── Fiducial centroide coarse (LPS->RAS: negar X e Y) ──",
            f'fid = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Centroid_Coarse_{pid}")',
            f"fid.AddControlPoint({-cx}, {-cy}, {cz})",
            "fdn = fid.GetDisplayNode()",
            "fdn.SetSelectedColor(0.2, 1.0, 0.3)",
            "fdn.SetGlyphScale(3.5)",
            "fdn.SetTextScale(3.5)",
            "fdn.SetSliceProjection(True)",
            "fdn.SetSliceProjectionColor(0.2, 1.0, 0.3)",
            "fdn.SetSliceProjection(True)",
            "slicer.util.resetSliceViews()",
            "slicer.util.resetSliceViews()",
            f'print("Centroide coarse: ({cx},{cy},{cz}) mm | prob={max_prob:.3f}")',
        ])

        logger.info(
            f"detect_coarse OK | {modality} | "
            f"centroide=({cx},{cy},{cz})mm | "
            f"max_prob={infer_result['max_prob']:.3f}"
        )

        return {
            "success":        True,
            "centroid_mm":    centroid_mm,
            "centroid_ras":   [-centroid_mm[0], -centroid_mm[1], centroid_mm[2]],
            "max_prob":       infer_result["max_prob"],
            "grid_shape":     infer_result["grid_shape"],
            "total_patches":  infer_result["total_patches"],
            "heatmap_path":   heatmap_path,
            "slicer_script":  slicer_script,
        }

    except Exception as e:
        logger.error(f"Error en detect_coarse: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── TOOL: crop_fine_from_coarse ───────────────────────────────────────────────

@mcp.tool(
    name="crop_fine_from_coarse",
    description=(
        "Genera el crop 160³ a 0.565mm centrado en el centroide coarse. "
        "Este es el volumen de entrada para el LocalizerUNet (detección fina) "
        "o directamente para el registro. Guarda el crop como NIfTI."
    ),
)
async def crop_fine_from_coarse_tool(
    vol_iso_path: str,
    centroid_mm: list,
    modality: str = "MRI",
    output_dir: str = "",
    pid: str = "",
) -> dict:
    """Crop 160³ centrado en el centroide coarse. O(160³)."""
    import os
    if ".." in vol_iso_path:
        return {"success": False, "error": "Path traversal detectado."}
    if len(centroid_mm) != 3:
        return {"success": False, "error": "centroid_mm debe tener 3 elementos [x,y,z]."}

    try:
        import SimpleITK as sitk
        vol_iso = sitk.ReadImage(vol_iso_path)
        vol_crop, crop_origin_mm = crop_fine_volume(vol_iso, centroid_mm)

        if not output_dir:
            data_dir = os.getenv("DATA_DIR", "./data")
            if not os.path.isabs(data_dir):
                data_dir = os.path.abspath(data_dir)
            output_dir = os.path.join(data_dir, "coarse", pid or "patient")
        else:
            output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{pid or 'vol'}_{modality}_loc_160.nii.gz")
        import SimpleITK as _sitk2
        _sitk2.WriteImage(vol_crop, out_path, useCompression=True)

        # Cropear también el heatmap si existe
        heatmap_crop_path = ""
        coarse_dir = os.path.join(
            os.getenv("DATA_DIR", "./data") if not os.path.isabs(os.getenv("DATA_DIR", "./data"))
            else os.getenv("DATA_DIR", "./data"),
            "coarse", pid or "patient"
        )
        if not os.path.isabs(coarse_dir):
            coarse_dir = os.path.abspath(coarse_dir)
        heatmap_path = os.path.join(coarse_dir, f"{pid or 'vol'}_{modality}_coarse_heatmap.nii.gz")
        if os.path.exists(heatmap_path):
            try:
                heat_iso  = sitk.ReadImage(heatmap_path)
                heat_crop, _ = crop_fine_volume(heat_iso, centroid_mm)
                heatmap_crop_path = os.path.join(
                    output_dir, f"{pid or 'vol'}_{modality}_coarse_heatmap_crop160.nii.gz"
                )
                _sitk2.WriteImage(heat_crop, heatmap_crop_path, useCompression=True)
                logger.info(f"heatmap crop OK | {heatmap_crop_path}")
            except Exception as eh:
                logger.warning(f"No se pudo cropear heatmap: {eh}")

        logger.info(f"crop_fine OK | {out_path}")
        return {
            "success":            True,
            "crop_path":          out_path,
            "heatmap_crop_path":  heatmap_crop_path,
            "crop_origin_mm":     crop_origin_mm,
            "centroid_mm":        centroid_mm,
            "size":               list(vol_crop.GetSize()),
            "spacing":            list(vol_crop.GetSpacing()),
        }
    except Exception as e:
        logger.error(f"Error en crop_fine: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── TOOL: rigid_registration_by_centroid ─────────────────────────────────────

@mcp.tool(
    name="rigid_registration_by_centroid",
    description=(
        "Realiza el registro rígido MRI-TRUS en 3D Slicer usando los centroides "
        "detectados por CoarseCNN. Lee automáticamente los centroides desde "
        "last_pipeline_meta.json y las rutas de los crops desde data/coarse/{pid}/. "
        "Ejecuta el registro en Slicer vía MCP-Slicer (stdio). "
        "Requiere: pid del caso, Slicer abierto con Web Server activo (puerto 2016)."
    ),
)
async def rigid_registration_by_centroid_tool(pid: str) -> dict:
    """
    Registro rígido por centroides CoarseCNN.
    O(1) — solo lectura de JSON y ejecución de script en Slicer.
    """
    import re, json as _j
    from pathlib import Path as _P

    if not re.match(r'^[a-zA-Z0-9_\-]{1,20}$', pid):
        return {"success": False, "error": "PID inválido."}

    # 1. Leer centroides desde last_pipeline_meta.json
    data_dir = os.getenv("DATA_DIR", "./data")
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)

    meta_path = os.path.join(data_dir, "last_pipeline_meta.json")
    if not _P(meta_path).exists():
        return {"success": False, "error": f"No se encontró last_pipeline_meta.json. Corre el pipeline primero."}

    try:
        meta = _j.loads(_P(meta_path).read_text())
    except Exception as e:
        return {"success": False, "error": f"Error leyendo metadata: {e}"}

    if meta.get("pid") != pid:
        return {"success": False, "error": f"La metadata es del PID {meta.get('pid')}, no {pid}. Corre el pipeline para este PID primero."}

    dual  = meta.get("detection_dual", {})
    c_mri  = dual.get("mri_centroid_mm", [])
    c_trus = dual.get("trus_centroid_mm", [])

    if not c_mri or not c_trus:
        return {"success": False, "error": "Centroides no encontrados en metadata."}

    # 2. Construir rutas de crops
    coarse_dir     = os.path.join(data_dir, "coarse", pid)
    mri_crop_path  = os.path.join(coarse_dir, f"{pid}_MRI_loc_160.nii.gz").replace("\\\\", "/").replace("\\", "/")
    trus_crop_path = os.path.join(coarse_dir, f"{pid}_TRUS_loc_160.nii.gz").replace("\\\\", "/").replace("\\", "/")

    if not _P(mri_crop_path).exists():
        return {"success": False, "error": f"Crop MRI no encontrado: {mri_crop_path}"}
    if not _P(trus_crop_path).exists():
        return {"success": False, "error": f"Crop TRUS no encontrado: {trus_crop_path}"}

    # 3. Generar script Python con valores hardcodeados
    mc = [round(v, 4) for v in c_mri]
    tc = [round(v, 4) for v in c_trus]

    script = f"""import vtk
slicer.mrmlScene.Clear(0)

# Centroides LPS → RAS
ras_mri  = [{-mc[0]}, {-mc[1]}, {mc[2]}]
ras_trus = [{-tc[0]}, {-tc[1]}, {tc[2]}]

# Fiducial MRI — verde
fid_mri = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "C_MRI")
fid_mri.AddControlPoint(*ras_mri)
fid_mri.GetDisplayNode().SetSelectedColor(0.0, 1.0, 0.0)
fid_mri.GetDisplayNode().SetGlyphScale(6.0)
fid_mri.GetDisplayNode().SetSliceProjection(True)

# Fiducial TRUS — naranja
fid_trus = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "C_TRUS")
fid_trus.AddControlPoint(*ras_trus)
fid_trus.GetDisplayNode().SetSelectedColor(1.0, 0.5, 0.0)
fid_trus.GetDisplayNode().SetGlyphScale(6.0)
fid_trus.GetDisplayNode().SetSliceProjection(True)

# Cargar crops 160³ @ 0.565mm
vol_mri  = slicer.util.loadVolume(r"{mri_crop_path}")
vol_mri.SetName("MRI_crop_160"); vol_mri.SetOrigin(0.0, 0.0, 0.0)
vol_trus = slicer.util.loadVolume(r"{trus_crop_path}")
vol_trus.SetName("TRUS_crop_160"); vol_trus.SetOrigin(0.0, 0.0, 0.0)

# Transform posición MRI
t_mri = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "T_MRI_pos")
m_mri = vtk.vtkMatrix4x4()
m_mri.SetElement(0,3,ras_mri[0]); m_mri.SetElement(1,3,ras_mri[1]); m_mri.SetElement(2,3,ras_mri[2])
t_mri.SetMatrixTransformToParent(m_mri)
vol_mri.SetAndObserveTransformNodeID(t_mri.GetID())

# Transform posición TRUS
t_trus_pos = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "T_TRUS_pos")
m_trus = vtk.vtkMatrix4x4()
m_trus.SetElement(0,3,ras_trus[0]); m_trus.SetElement(1,3,ras_trus[1]); m_trus.SetElement(2,3,ras_trus[2])
t_trus_pos.SetMatrixTransformToParent(m_trus)
vol_trus.SetAndObserveTransformNodeID(t_trus_pos.GetID())

# Transform de corrección (mueve TRUS → MRI)
dx={round(-mc[0]-(-tc[0]),4)}; dy={round(-mc[1]-(-tc[1]),4)}; dz={round(mc[2]-tc[2],4)}
t_corr = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "T_correccion")
m_corr = vtk.vtkMatrix4x4()
m_corr.SetElement(0,3,dx); m_corr.SetElement(1,3,dy); m_corr.SetElement(2,3,dz)
t_corr.SetMatrixTransformToParent(m_corr)
t_trus_pos.SetAndObserveTransformNodeID(t_corr.GetID())
fid_trus.SetAndObserveTransformNodeID(t_corr.GetID())

# Visualizar
vol_mri.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")
vol_mri.GetDisplayNode().SetAutoWindowLevel(True)
vol_trus.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileHotToColdRainbow.txt")
vol_trus.GetDisplayNode().SetAutoWindowLevel(True)
slicer.util.setSliceViewerLayers(background=vol_mri, foreground=vol_trus, foregroundOpacity=0.4)
slicer.util.resetSliceViews()
__execResult = f"Registro rigido OK PID={pid} | dx={{dx:.1f}} dy={{dy:.1f}} dz={{dz:.1f}} mm"
""".replace('pid', pid)

    # 4. Ejecutar en Slicer vía MCP-Slicer
    uvx_path = os.getenv("UVX_PATH", r"C:\Users\erick\.local\bin\uvx.exe")
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(command=uvx_path, args=["mcp-slicer"])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("execute_python_code", {"code": script})
                content_out = ""
                if hasattr(result, "content") and result.content:
                    item = result.content[0]
                    content_out = item.text if hasattr(item, "text") else str(item)
                logger.info(f"rigid_registration OK PID={pid} | {content_out[:100]}")
                return {
                    "success":       True,
                    "pid":           pid,
                    "c_mri":         mc,
                    "c_trus":        tc,
                    "mri_crop":      mri_crop_path,
                    "trus_crop":     trus_crop_path,
                    "slicer_result": content_out,
                }
    except RuntimeError as e:
        return {"success": False, "error": f"Slicer no disponible: {e}"}
    except Exception as e:
        logger.error(f"rigid_registration error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── TOOL: load_heatmaps_with_grid ────────────────────────────────────────────

@mcp.tool(
    name="load_heatmaps_with_grid",
    description=(
        "Carga los heatmaps MRI y TRUS del CoarseCNN en 3D Slicer con cubos 32³ visibles "
        "(sin interpolación) y los alinea con el registro rígido actual. "
        "Requiere: pid del caso, registro rígido ya ejecutado en Slicer (T_MRI_pos y T_correccion). "
        "Usa esta tool cuando el usuario pida ver cómo el modelo detectó la próstata, "
        "ver los cubos del modelo, o visualizar los heatmaps alineados."
    ),
)
async def load_heatmaps_with_grid_tool(pid: str) -> dict:
    """
    Carga heatmaps MRI+TRUS con SetInterpolate(0) y aplica transforms del registro.
    O(1) — lectura de JSON + ejecución de script en Slicer.
    """
    import re, json as _j
    from pathlib import Path as _P

    if not re.match(r'^[a-zA-Z0-9_\-]{1,20}$', pid):
        return {"success": False, "error": "PID inválido."}

    # Leer metadata
    data_dir  = os.getenv("DATA_DIR", "./data")
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)

    meta_path = os.path.join(data_dir, "last_pipeline_meta.json")
    if not _P(meta_path).exists():
        return {"success": False, "error": "No se encontró last_pipeline_meta.json. Corre el pipeline primero."}

    meta = _j.loads(_P(meta_path).read_text())
    if meta.get("pid") != pid:
        return {"success": False, "error": f"Metadata es del PID {meta.get('pid')}, no {pid}."}

    dual           = meta.get("detection_dual", {})
    coarse_dir     = os.path.join(data_dir, "coarse", pid)

    # Preferir heatmap recortado a 160³ si existe
    mri_heat_crop  = os.path.join(coarse_dir, f"{pid}_MRI_coarse_heatmap_crop160.nii.gz")
    trus_heat_crop = os.path.join(coarse_dir, f"{pid}_TRUS_coarse_heatmap_crop160.nii.gz")
    mri_heat_full  = dual.get("mri_heatmap", "").replace("\\\\", "/").replace("\\", "/")
    trus_heat_full = dual.get("trus_heatmap", "").replace("\\\\", "/").replace("\\", "/")

    mri_heat_path  = mri_heat_crop  if _P(mri_heat_crop).exists()  else mri_heat_full
    trus_heat_path = trus_heat_crop if _P(trus_heat_crop).exists() else trus_heat_full
    using_crop     = _P(mri_heat_crop).exists()

    if not mri_heat_path or not _P(mri_heat_path).exists():
        return {
            "success": False,
            "error": (
                f"Heatmap MRI no encontrado. "
                f"Corre el pipeline agente con PID {pid} hasta el paso hitl_crops "
                f"para generar los heatmaps recortados a 160³."
            )
        }
    if not trus_heat_path or not _P(trus_heat_path).exists():
        return {
            "success": False,
            "error": (
                f"Heatmap TRUS no encontrado. "
                f"Corre el pipeline agente con PID {pid} hasta el paso hitl_crops "
                f"para generar los heatmaps recortados a 160³."
            )
        }

    script = f"""
# Cargar heatmap MRI
heat_mri = slicer.util.loadVolume(r"{mri_heat_path}")
heat_mri.SetName("Heatmap_MRI_{pid}")
heat_mri.SetOrigin(0.0, 0.0, 0.0)
dn_mri = heat_mri.GetDisplayNode()
dn_mri.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")
dn_mri.SetAutoWindowLevel(False)
dn_mri.SetWindowLevelMinMax(0.05, 1.0)
dn_mri.SetOpacity(0.6)
dn_mri.SetInterpolate(0)

# Cargar heatmap TRUS
heat_trus = slicer.util.loadVolume(r"{trus_heat_path}")
heat_trus.SetName("Heatmap_TRUS_{pid}")
heat_trus.SetOrigin(0.0, 0.0, 0.0)
dn_trus = heat_trus.GetDisplayNode()
dn_trus.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileHotToColdRainbow.txt")
dn_trus.SetAutoWindowLevel(True)
dn_trus.SetOpacity(0.7)
dn_trus.SetInterpolate(0)

# Aplicar transforms del registro si existen
t_mri      = slicer.mrmlScene.GetFirstNodeByName("T_MRI_pos")
t_trus_pos = slicer.mrmlScene.GetFirstNodeByName("T_TRUS_pos")
t_corr     = slicer.mrmlScene.GetFirstNodeByName("T_correccion")
if t_mri:
    heat_mri.SetAndObserveTransformNodeID(t_mri.GetID())
if t_trus_pos:
    heat_trus.SetAndObserveTransformNodeID(t_trus_pos.GetID())
elif t_corr:
    heat_trus.SetAndObserveTransformNodeID(t_corr.GetID())

slicer.util.resetSliceViews()
__execResult = ("Heatmaps crop 160³" if True else "Heatmaps vol_iso") + " con cubos visibles" + (" alineados" if t_mri else "")
"""

    uvx_path = os.getenv("UVX_PATH", r"C:\Users\erick\.local\bin\uvx.exe")
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(command=uvx_path, args=["mcp-slicer"])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("execute_python_code", {"code": script})
                content_out = ""
                if hasattr(result, "content") and result.content:
                    item = result.content[0]
                    content_out = item.text if hasattr(item, "text") else str(item)
                return {"success": True, "pid": pid, "slicer_result": content_out}
    except RuntimeError as e:
        return {"success": False, "error": f"Slicer no disponible: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── RESOURCES ─────────────────────────────────────────────────────────────────

@mcp.resource(
    uri="prostate://case_registry",
    name="case_registry",
    description=(
        "Registro de todos los casos DICOM cargados en el sistema. "
        "Incluye estado del pipeline, número de series, y timestamps."
    ),
    mime_type="application/json",
)
async def resource_case_registry() -> str:
    """
    MCP Resource: estado del registro de casos.
    Retorna JSON con resumen de todos los casos. O(n).
    """
    import json
    return json.dumps(_registry.to_resource_dict(), indent=2)


@mcp.resource(
    uri="prostate://slicer_scene",
    name="slicer_scene",
    description="Estado actual de la escena de 3D Slicer: nodos cargados.",
    mime_type="application/json",
)
async def resource_slicer_scene() -> str:
    """MCP Resource: nodos activos en 3D Slicer. O(1) red."""
    import json
    if not await _slicer.is_alive():
        return json.dumps({"available": False, "nodes": []})
    try:
        nodes = await _slicer.get_scene_nodes()
        return json.dumps({"available": True, "nodes": nodes})
    except Exception as e:
        return json.dumps({"available": True, "error": str(e), "nodes": []})


# ── Entrypoint ────────────────────────────────────────────────────────────────
#
# Dos modos de transporte:
#   python -m mcp_server.server          → stdio  (LangGraph, Fase 3)
#   python -m mcp_server.server --http   → HTTP/SSE en :8765 (UI Gradio)
#
# LangGraph lanza el servidor como subproceso → usa stdio.
# La UI Gradio es una app web independiente → usa HTTP/SSE.
# Ambos modos exponen los mismos tools y resources.

if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))
    logger.info("Iniciando ProstatePipelineMCP — Fase 1/6")

    if "--http" in sys.argv:
        host = os.getenv("MCP_SERVER_HOST", "localhost")
        port = int(os.getenv("MCP_SERVER_PORT", "8765"))
        logger.info(f"Modo HTTP/SSE en http://{host}:{port}/sse")
        logger.info("Conecta la UI Gradio a este endpoint.")
        mcp.run(transport="sse", host=host, port=port)
    else:
        logger.info("Modo stdio — para LangGraph (Fase 3)")
        mcp.run(transport="stdio")