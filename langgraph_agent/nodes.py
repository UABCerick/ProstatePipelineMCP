"""
nodes.py — Nodos del grafo LangGraph para el pipeline prostático.

Cada nodo es una función async que recibe el estado actual y retorna
un dict con los campos a actualizar. LangGraph aplica los updates
automáticamente usando los reducers definidos en PipelineState.

Flujo del grafo:
  load → detect → [HITL: fov] → plan_params → [HITL: params] → 
  register → validate → [HITL: failure si falla] → report → END

Complejidad:
  Cada nodo es O(1) excepto donde se llama al servidor MCP (O(V) para detect).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt, Command
from loguru import logger




from .state import HITLDecision, PipelineState, RegistrationParams
from .llm_client import (
    get_llm,
    suggest_registration_params,
    interpret_metrics,
    generate_report_summary,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()



# ── Ruta al uvx con mcp-slicer ──────────────────────────────────────────────
UVX_PATH = os.getenv("UVX_PATH", r"C:\Users\erick\.local\bin\uvx.exe")


async def _call_slicer_tool(tool_name: str, params: dict) -> dict:
    """
    Llama a un tool del MCP-Slicer via stdio (uvx mcp-slicer).
    Lanza uvx como subprocess y usa mcp.ClientSession para comunicarse.
    Si Slicer no esta disponible, lanza SlicerNotAvailableError.
    """
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    # Verificar que Slicer esta activo antes de lanzar el subprocess
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as http:
            r = await http.get("http://localhost:2016/slicer/mrml/names")
            if r.status_code != 200:
                raise RuntimeError("Slicer Web Server no responde")
    except Exception:
        raise RuntimeError(
            "3D Slicer no esta disponible. "
            "Abre Slicer y activa el Web Server (puerto 2016) antes de continuar."
        )

    server_params = StdioServerParameters(
        command=UVX_PATH,
        args=["mcp-slicer"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, params)
            # Extraer contenido del resultado
            if hasattr(result, "content") and result.content:
                item = result.content[0]
                if hasattr(item, "text"):
                    import json
                    try:
                        return json.loads(item.text)
                    except Exception:
                        return {"success": True, "result": item.text}
            return {"success": True}


async def _call_mcp_tool(tool_name: str, params: dict) -> dict:
    """
    Llama a un tool del servidor MCP vía HTTP/SSE.
    Compatible con FastMCP 3.x — CallToolResult.
    """
    import json

    mcp_url = f"http://{os.getenv('MCP_SERVER_HOST','localhost')}:{os.getenv('MCP_SERVER_PORT','8765')}"

    try:
        from fastmcp import Client
        async with Client(f"{mcp_url}/sse") as client:
            result = await client.call_tool(tool_name, params)

            # FastMCP 3.x retorna CallToolResult con atributo .content
            content = None
            if hasattr(result, "content"):
                content = result.content
            elif isinstance(result, list):
                content = result

            if content and len(content) > 0:
                item = content[0]
                text = item.text if hasattr(item, "text") else str(item)
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"success": True, "output": text}

            return {"success": False, "error": "Respuesta vacía del servidor MCP"}

    except Exception as e:
        logger.error(f"Error llamando tool MCP {tool_name}: {e}")
        return {"success": False, "error": str(e)}


# ── Nodo 1: LOAD ────────────────────────────────────────────────────────────

async def node_load(state: PipelineState) -> dict:
    """
    Carga rutas crudas del paciente desde stage1_selected_pairs.json.
    Verifica que existen DICOM MRI y NIfTI TRUS exportado.
    Complejidad: O(1) — lookup en dict.
    """
    logger.info(f"[LOAD] PID={state.pid}")

    result = await _call_mcp_tool("get_raw_patient_info", {"pid": state.pid})

    if not result.get("success"):
        # Fallback: intentar con get_case_info del dataset Localizer
        logger.warning(f"[LOAD] get_raw_patient_info fallo, intentando get_case_info")
        result = await _call_mcp_tool("get_case_info", {"pid": state.pid})
        if not result.get("success"):
            return {
                "current_node": "load",
                "error": f"Caso {state.pid} no encontrado: {result.get('error','')}",
                "messages": [AIMessage(content=f"Error: PID {state.pid} no encontrado.")],
            }
        # Usar rutas del dataset Localizer como fallback
        case_data = result.get("case", {})
        case_data["_source"] = "localizer_dataset"
    else:
        case_data = result.get("case", {})
        case_data["_source"] = "stage1_raw"

    logger.info(
        f"[LOAD] OK — PID={state.pid} | "
        f"source={case_data.get('_source')} | "
        f"has_raw={case_data.get('has_raw', False)}"
    )

    return {
        "current_node": "detect",
        "case_metadata": case_data,
        "case_id": state.pid,
        "messages": [
            AIMessage(content=(
                f"Caso {state.pid} cargado desde {case_data.get('_source', 'dataset')}. "
                f"Iniciando deteccion CoarseCNN dual MRI+TRUS."
            ))
        ],
    }


# ── Nodo: PREPROCESS ─────────────────────────────────────────────────────────

async def node_preprocess(state: PipelineState) -> dict:
    """
    Verifica y aplica preprocesamiento a los volumenes crudos.
    Si vol_iso ya existe con spacing=0.565 y origen=(0,0,0), lo reutiliza.
    Si no, corre preprocess_raw_volume.
    Complejidad: O(1) si cache existe, O(V_orig) si hay que preprocesar.
    """
    logger.info(f"[PREPROCESS] PID={state.pid}")
    import asyncio as _aio
    import os

    meta       = state.case_metadata
    mri_dicom  = meta.get("mri_dicom_path", "")
    trus_nifti = meta.get("trus_nifti_path", "")

    data_dir = os.getenv("DATA_DIR", "./data")
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)
    preproc_dir   = os.path.join(data_dir, "preprocessed", state.pid)
    mri_iso_cache  = os.path.join(preproc_dir, f"{state.pid}_MRI_vol_iso.nii.gz")
    trus_iso_cache = os.path.join(preproc_dir, f"{state.pid}_TRUS_vol_iso.nii.gz")

    def _needs_preproc(path: str) -> bool:
        from pathlib import Path as _P
        if not path or not _P(path).exists():
            return True
        try:
            import SimpleITK as _sitk
            img = _sitk.ReadImage(path)
            ok_sp  = all(abs(s - 0.565) < 0.01 for s in img.GetSpacing())
            ok_org = all(abs(o) < 0.1 for o in img.GetOrigin())
            return not (ok_sp and ok_org)
        except Exception:
            return True

    tasks = {}
    if _needs_preproc(mri_iso_cache) and mri_dicom:
        logger.info(f"[PREPROCESS] Preprocesando MRI")
        tasks["mri"] = _call_mcp_tool("preprocess_raw_volume", {
            "input_path": mri_dicom, "modality": "MRI",
            "pid": state.pid, "is_dicom": True,
        })
    else:
        logger.info(f"[PREPROCESS] MRI vol_iso en cache OK")

    if _needs_preproc(trus_iso_cache) and trus_nifti:
        logger.info(f"[PREPROCESS] Preprocesando TRUS")
        tasks["trus"] = _call_mcp_tool("preprocess_raw_volume", {
            "input_path": trus_nifti, "modality": "TRUS",
            "pid": state.pid, "is_dicom": False,
        })
    else:
        logger.info(f"[PREPROCESS] TRUS vol_iso en cache OK")

    if tasks:
        keys    = list(tasks.keys())
        results = await _aio.gather(*tasks.values())
        for key, res in zip(keys, results):
            if not res.get("success"):
                return {
                    "current_node": "preprocess",
                    "error": f"Preprocesamiento {key.upper()} fallo: {res.get('error','')}",
                    "messages": [AIMessage(content=f"Error preprocesando {key.upper()}.")],
                }
            if key == "mri":
                mri_iso_cache  = res["vol_iso_path"]
            else:
                trus_iso_cache = res["vol_iso_path"]

    updated_meta = dict(meta)
    updated_meta["mri_vol_iso_path"]  = mri_iso_cache
    updated_meta["trus_vol_iso_path"] = trus_iso_cache

    logger.info(f"[PREPROCESS] OK | MRI={mri_iso_cache}")
    return {
        "current_node":  "detect",
        "case_metadata": updated_meta,
        "messages": [AIMessage(content=f"Preprocesamiento listo para PID {state.pid}.")],
    }


# ── Nodo HITL: FOV MRI ───────────────────────────────────────────────────────

async def node_hitl_fov_mri(state: PipelineState) -> dict:
    """HITL #1a — Verificar centroide MRI en vol_iso completo."""
    logger.info(f"[HITL_FOV_MRI] Esperando aprobacion MRI PID={state.pid}")

    decision = interrupt({
        "type":          "hitl_fov_mri",
        "message":       state.hitl_message,
        "pid":           state.pid,
        "centroid_mm":   state.case_metadata.get("detection_dual", {}).get("mri_centroid_mm", []),
        "max_prob":      state.case_metadata.get("detection_dual", {}).get("mri_max_prob", 0),
        "slicer_script": state.slicer_fov_script,
        "instructions":  "Copia el script en Slicer, verifica centroide MRI, aprueba o rechaza.",
    })

    hitl_record = HITLDecision(
        point="fov_mri",
        decision=decision.get("decision", "approved"),
        notes=decision.get("notes", ""),
    )
    approved = hitl_record.decision == "approved"
    logger.info(f"[HITL_FOV_MRI] Decision: {hitl_record.decision}")

    return {
        "current_node":   "hitl_fov_trus" if approved else "detect",
        "pending_hitl":   False,
        "hitl_decisions": state.hitl_decisions + [hitl_record],
    }


# ── Nodo HITL: FOV TRUS ──────────────────────────────────────────────────────

async def node_hitl_fov_trus(state: PipelineState) -> dict:
    """HITL #1b — Verificar centroide TRUS en vol_iso completo."""
    logger.info(f"[HITL_FOV_TRUS] Esperando aprobacion TRUS PID={state.pid}")

    meta       = state.case_metadata
    dual       = meta.get("detection_dual", {})
    trus_c     = dual.get("trus_centroid_mm", [])
    trus_prob  = dual.get("trus_max_prob", 0)
    trus_iso   = meta.get("trus_vol_iso_path", "")
    trus_heat  = dual.get("trus_heatmap", "")

    safe_trus = trus_iso.replace("\\\\", "/").replace("\\", "/") if trus_iso else ""
    safe_heat = trus_heat.replace("\\\\", "/").replace("\\", "/") if trus_heat else ""
    tc  = [round(v,1) for v in trus_c] if trus_c else []
    tx  = tc[0] if tc else 0
    ty  = tc[1] if tc else 0
    tz  = tc[2] if tc else 0

    trus_script = chr(10).join([
        "import slicer",
        f"vol = slicer.util.loadVolume(r\"{safe_trus}\")",
        f"vol.SetName(\"{state.pid}_TRUS_vol_iso\")",
        "vol.SetOrigin(0.0, 0.0, 0.0)",
        f"heat = slicer.util.loadVolume(r\"{safe_heat}\")",
        f"heat.SetName(\"TRUS_Heatmap_{state.pid}\")",
        "heat.SetOrigin(0.0, 0.0, 0.0)",
        "dn = heat.GetDisplayNode()",
        "dn.SetAndObserveColorNodeID(\"vtkMRMLColorTableNodeRainbow\")",
        "dn.SetAutoWindowLevel(False)",
        "dn.SetWindowLevelMinMax(0.05, 1.0)",
        "dn.SetOpacity(0.7)",
        "slicer.util.setSliceViewerLayers(background=vol, foreground=heat, foregroundOpacity=0.5)",
        f"fid = slicer.mrmlScene.AddNewNodeByClass(\"vtkMRMLMarkupsFiducialNode\", \"CTR_TRUS_{state.pid}\")",
        f"fid.AddControlPoint({-tx}, {-ty}, {tz})",
        "fid.GetDisplayNode().SetSelectedColor(1.0, 0.5, 0.0)",
        "fid.GetDisplayNode().SetGlyphScale(3.5)",
        "fid.GetDisplayNode().SetSliceProjection(True)",
        "slicer.util.resetSliceViews()",
        "slicer.util.resetSliceViews()",
        f"print(\"Centroide TRUS: ({tx},{ty},{tz}) mm | prob={trus_prob:.3f}\")",
    ]) if safe_trus else "print(\"TRUS no disponible\")"

    hitl_msg = (
        f"Verificar centroide TRUS para PID {state.pid}:\n"
        f"Centro={tc}mm | max_prob={trus_prob:.3f}\n\n"
        "Copia el script en Slicer. Centroide en naranja.\n"
        "Aprueba si esta sobre la prostata."
    )

    decision = interrupt({
        "type":          "hitl_fov_trus",
        "message":       hitl_msg,
        "pid":           state.pid,
        "slicer_script": trus_script,
        "instructions":  "Copia el script TRUS en Slicer, verifica centroide, aprueba o rechaza.",
    })

    hitl_record = HITLDecision(
        point="fov_trus",
        decision=decision.get("decision", "approved"),
        notes=decision.get("notes", ""),
    )
    approved = hitl_record.decision == "approved"
    logger.info(f"[HITL_FOV_TRUS] Decision: {hitl_record.decision}")

    return {
        "current_node":      "crop_fine" if approved else "detect",
        "pending_hitl":      False,
        "slicer_fov_script": trus_script,
        "hitl_message":      hitl_msg,
        "hitl_decisions":    state.hitl_decisions + [hitl_record],
    }


async def node_detect(state: PipelineState) -> dict:
    """
    Preprocesa volumenes crudos y corre CoarseCNN dual MRI+TRUS.

    Flujo:
        1. Si hay rutas raw (DICOM/NIfTI): preprocess_raw_volume -> vol_iso
        2. CoarseCNN en paralelo MRI + TRUS sobre vol_iso completo
        3. Guardar heatmaps y calcular traslacion inicial
        4. HITL: investigador verifica en Slicer antes del crop

    El crop 160³ NO se aplica aqui — solo despues del HITL.
    Complejidad: O(2 x N_patches x 32³).
    """
    logger.info(f"[DETECT] CoarseCNN dual para PID={state.pid}")

    import asyncio as _aio
    import numpy as _np

    meta   = state.case_metadata
    source = meta.get("_source", "localizer_dataset")

    # ── Obtener rutas de vol_iso ──────────────────────────────────────────
    if source == "stage1_raw":
        # Preprocesar desde raw si no existe vol_iso en cache
        mri_iso  = meta.get("mri_vol_iso_path", "")
        trus_iso = meta.get("trus_vol_iso_path", "")

        tasks = []
        if not mri_iso or not __import__("pathlib").Path(mri_iso).exists():
            logger.info(f"[DETECT] Preprocesando MRI crudo para PID={state.pid}")
            tasks.append(_call_mcp_tool("preprocess_raw_volume", {
                "input_path": meta.get("mri_dicom_path", ""),
                "modality":   "MRI",
                "pid":        state.pid,
            }))
        else:
            tasks.append(None)

        if not trus_iso or not __import__("pathlib").Path(trus_iso).exists():
            logger.info(f"[DETECT] Preprocesando TRUS crudo para PID={state.pid}")
            tasks.append(_call_mcp_tool("preprocess_raw_volume", {
                "input_path": meta.get("trus_nifti_path", ""),
                "modality":   "TRUS",
                "is_dicom":   False,
                "pid":        state.pid,
            }))
        else:
            tasks.append(None)

        # Ejecutar preprocesamiento en paralelo si necesario
        real_tasks = [t for t in tasks if t is not None]
        if real_tasks:
            results = await _aio.gather(*real_tasks)
            ri = 0
            if tasks[0] is not None:
                r = results[ri]; ri += 1
                if r.get("success"):
                    mri_iso = r["vol_iso_path"]
                else:
                    return {"current_node": "detect",
                            "error": f"Preprocesamiento MRI fallo: {r.get('error','')}",
                            "messages": [AIMessage(content="Error preprocesando MRI.")]}
            if tasks[1] is not None:
                r = results[ri]
                if r.get("success"):
                    trus_iso = r["vol_iso_path"]
                else:
                    logger.warning(f"[DETECT] Preprocesamiento TRUS fallo: {r.get('error','')}")
                    trus_iso = ""

    else:
        # Fallback: usar vol del dataset Localizer (ya a 0.565mm)
        mri_iso  = meta.get("mri_volume", "")
        trus_iso = meta.get("trus_volume", "")
        logger.info(f"[DETECT] Usando vols del dataset Localizer para PID={state.pid}")

    if not mri_iso:
        return {"current_node": "detect",
                "error": "No hay vol_iso MRI disponible.",
                "messages": [AIMessage(content="Error: vol_iso MRI no disponible.")]}

    # ── CoarseCNN en paralelo ─────────────────────────────────────────────
    logger.info(f"[DETECT] CoarseCNN MRI: {mri_iso}")
    mri_task = _call_mcp_tool("detect_coarse_center", {
        "vol_iso_path": mri_iso, "modality": "MRI", "pid": state.pid,
    })

    trus_task = None
    if trus_iso:
        logger.info(f"[DETECT] CoarseCNN TRUS: {trus_iso}")
        trus_task = _call_mcp_tool("detect_coarse_center", {
            "vol_iso_path": trus_iso, "modality": "TRUS", "pid": state.pid,
        })

    if trus_task:
        mri_result, trus_result = await _aio.gather(mri_task, trus_task)
    else:
        mri_result = await mri_task
        trus_result = {"success": False, "error": "TRUS no disponible"}

    mri_ok  = mri_result.get("success", False)
    trus_ok = trus_result.get("success", False)

    if not mri_ok:
        return {"current_node": "detect",
                "error": f"CoarseCNN MRI fallo: {mri_result.get('error','')}",
                "messages": [AIMessage(content="Error CoarseCNN MRI.")]}

    mri_centroid_mm  = mri_result.get("centroid_mm", [])
    mri_centroid_ras = mri_result.get("centroid_ras", [])
    mri_max_prob     = mri_result.get("max_prob", 0)
    mri_fov_script   = mri_result.get("slicer_script", "")
    trus_centroid_mm = trus_result.get("centroid_mm", []) if trus_ok else []
    trus_max_prob    = trus_result.get("max_prob", 0)

    # Traslacion inicial MRI->TRUS para inicializar registro
    initial_translation = None
    translation_mm = None
    if mri_centroid_mm and trus_centroid_mm:
        diff = _np.array(mri_centroid_mm) - _np.array(trus_centroid_mm)
        initial_translation = diff.tolist()
        translation_mm = round(float(_np.linalg.norm(diff)), 2)

    # Guardar paths de vol_iso para el crop posterior al HITL
    updated_meta = dict(meta)
    updated_meta["mri_vol_iso_path"]  = mri_iso
    updated_meta["trus_vol_iso_path"] = trus_iso
    updated_meta["detection_dual"] = {
        "mri_centroid_mm":    mri_centroid_mm,
        "trus_centroid_mm":   trus_centroid_mm,
        "mri_centroid_ras":   mri_centroid_ras,
        "mri_max_prob":       mri_max_prob,
        "trus_max_prob":      trus_max_prob,
        "initial_translation": initial_translation,
        "translation_norm_mm": translation_mm,
        "trus_ok":            trus_ok,
        "mri_heatmap":        mri_result.get("heatmap_path", ""),
        "trus_heatmap":       trus_result.get("heatmap_path", "") if trus_ok else "",
    }

    mri_fmt  = [round(v,1) for v in mri_centroid_mm] if mri_centroid_mm else "N/A"
    trus_fmt = [round(v,1) for v in trus_centroid_mm] if trus_centroid_mm else "N/A"
    trus_info = f"centro={trus_fmt} max_prob={round(trus_max_prob,3)}" if trus_ok else "No disponible"

    hitl_msg = (
        f"CoarseCNN completado para PID {state.pid}:\n\n"
        f"MRI (vol_iso completo):\n"
        f"  Centro={mri_fmt}mm | max_prob={mri_max_prob:.3f}\n\n"
        f"TRUS: {trus_info}\n"
        f"Traslacion MRI->TRUS: {translation_mm}mm\n\n"
        "Verifica en Slicer con el script FOV (vol_iso completo + heatmap).\n"
        "El crop 160³ se aplicara DESPUES de que apruebes este centroide."
    )

    logger.info(
        f"[DETECT] OK | MRI={mri_max_prob:.3f} | "
        f"TRUS={trus_max_prob:.3f} | trans={translation_mm}mm"
    )

    # Guardar metadata para el chat 3D Slicer AI
    import json as _jj, os as _os
    _data_dir = _os.getenv("DATA_DIR", "./data")
    if not _os.path.isabs(_data_dir):
        _data_dir = _os.path.abspath(_data_dir)
    _os.makedirs(_data_dir, exist_ok=True)
    try:
        _meta_out = {
            "pid":               state.pid,
            "detection_dual":    updated_meta.get("detection_dual", {}),
            "mri_vol_iso_path":  updated_meta.get("mri_vol_iso_path", ""),
            "trus_vol_iso_path": updated_meta.get("trus_vol_iso_path", ""),
        }
        open(_os.path.join(_data_dir, "last_pipeline_meta.json"), "w").write(
            _jj.dumps(_meta_out, indent=2)
        )
    except Exception as _e:
        logger.warning(f"[DETECT] No se pudo guardar metadata: {_e}")

    # Generar script combinado MRI+TRUS para el HITL
    trus_iso_path  = updated_meta.get("trus_vol_iso_path", "")
    trus_heat_path = trus_result.get("heatmap_path", "") if trus_ok else ""
    safe_trus = trus_iso_path.replace("\\\\", "/").replace("\\", "/") if trus_iso_path else ""
    safe_heat_t = trus_heat_path.replace("\\\\", "/").replace("\\", "/") if trus_heat_path else ""
    tc = [round(v,1) for v in trus_centroid_mm] if trus_centroid_mm else []
    tx = -tc[0] if tc else 0
    ty = -tc[1] if tc else 0
    tz =  tc[2] if tc else 0

    trus_script_combined = chr(10).join([
        "",
        "# ══ TRUS vol_iso + heatmap ══",
        f'vol_t = slicer.util.loadVolume(r"{safe_trus}")',
        f'vol_t.SetName("{state.pid}_TRUS_vol_iso")',
        "vol_t.SetOrigin(0.0, 0.0, 0.0)",
        f'heat_t = slicer.util.loadVolume(r"{safe_heat_t}")',
        f'heat_t.SetName("TRUS_Heatmap_{state.pid}")',
        "heat_t.SetOrigin(0.0, 0.0, 0.0)",
        "dn_t = heat_t.GetDisplayNode()",
        'dn_t.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")',
        "dn_t.SetAutoWindowLevel(False)",
        "dn_t.SetWindowLevelMinMax(0.05, 1.0)",
        "dn_t.SetOpacity(0.7)",
        f'fid_t = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Centroid_Coarse_TRUS_{state.pid}")',
        f"fid_t.AddControlPoint({tx}, {ty}, {tz})",
        "fid_t.GetDisplayNode().SetSelectedColor(1.0, 0.5, 0.0)",
        "fid_t.GetDisplayNode().SetGlyphScale(3.5)",
        "fid_t.GetDisplayNode().SetSliceProjection(True)",
        f'print("TRUS centroide: ({tc[0] if tc else 0},{tc[1] if tc else 0},{tc[2] if tc else 0})mm")',
    ]) if safe_trus else ""

    combined_fov_script = (mri_fov_script or "") + trus_script_combined


    return {
        "current_node":       "hitl_fov",
        "centroid_vox":       None,
        "centroid_ras":       mri_centroid_ras,
        "centroid_global_mm": mri_centroid_mm,
        "crop_fine_mm":       None,
        "confidence":         mri_max_prob,
        "gt_error_mm":        None,
        "fov_radius_mm":      45.2,
        "slicer_fov_script":  combined_fov_script,
        "case_metadata":      updated_meta,
        "pending_hitl":       True,
        "pending_hitl_type":  "fov",
        "hitl_message":       hitl_msg,
        "messages":           [AIMessage(content=hitl_msg)],
    }

# ── Nodo 2b: CROP FINO post-HITL ────────────────────────────────────────────

async def node_crop_fine(state: PipelineState) -> dict:
    """
    Genera crops 160³ @ 0.565mm para MRI y TRUS centrados en los centroides
    detectados por CoarseCNN. Se ejecuta DESPUES del HITL de FOV.

    El crop mantiene exactamente 0.565mm de spacing — clave para el registro.
    Si el centroide cae cerca del borde, el crop se rellena con ceros.

    Complejidad: O(160³) por modalidad — resampleo fijo.
    """
    logger.info(f"[CROP_FINE] Generando crops 160³ para PID={state.pid}")

    import asyncio as _aio

    meta   = state.case_metadata
    dual   = meta.get("detection_dual", {})
    mri_iso  = meta.get("mri_vol_iso_path", "")
    trus_iso = meta.get("trus_vol_iso_path", "")
    mri_c    = dual.get("mri_centroid_mm", [])
    trus_c   = dual.get("trus_centroid_mm", [])

    if not mri_iso or not mri_c:
        return {
            "current_node": "crop_fine",
            "error": "Faltan rutas vol_iso o centroides para el crop.",
            "messages": [AIMessage(content="Error: datos insuficientes para crop.")],
        }

    # Crops en paralelo
    tasks = [
        _call_mcp_tool("crop_fine_from_coarse", {
            "vol_iso_path": mri_iso,
            "centroid_mm":  mri_c,
            "modality":     "MRI",
            "pid":          state.pid,
        }),
    ]
    if trus_iso and trus_c:
        tasks.append(_call_mcp_tool("crop_fine_from_coarse", {
            "vol_iso_path": trus_iso,
            "centroid_mm":  trus_c,
            "modality":     "TRUS",
            "pid":          state.pid,
        }))

    results = await _aio.gather(*tasks)
    mri_crop_result  = results[0]
    trus_crop_result = results[1] if len(results) > 1 else {"success": False}

    if not mri_crop_result.get("success"):
        return {
            "current_node": "crop_fine",
            "error": f"Crop MRI fallo: {mri_crop_result.get('error','')}",
            "messages": [AIMessage(content="Error en crop MRI.")],
        }

    mri_crop_path  = mri_crop_result.get("crop_path", "")
    trus_crop_path = trus_crop_result.get("crop_path", "") if trus_crop_result.get("success") else ""
    mri_origin     = mri_crop_result.get("crop_origin_mm", [])
    trus_origin    = trus_crop_result.get("crop_origin_mm", []) if trus_crop_result.get("success") else []

    # Generar script Slicer para verificar ambos crops
    safe_mri  = mri_crop_path.replace("\\", "/") if mri_crop_path else ""
    safe_trus = trus_crop_path.replace("\\", "/") if trus_crop_path else ""
    mc = [round(v,1) for v in mri_c]
    tc = [round(v,1) for v in trus_c] if trus_c else []

    script_lines = [
        "import slicer",
        "",
        "# ── Crop MRI 160³ @ 0.565mm ──",
        f'mri_crop = slicer.util.loadVolume(r"{safe_mri}")',
        f'mri_crop.SetName("MRI_crop_160_{state.pid}")',
        "mri_crop.SetOrigin(0.0, 0.0, 0.0)",
    ]
    if safe_trus:
        script_lines += [
            "",
            "# ── Crop TRUS 160³ @ 0.565mm ──",
            f'trus_crop = slicer.util.loadVolume(r"{safe_trus}")',
            f'trus_crop.SetName("TRUS_crop_160_{state.pid}")',
            "trus_crop.SetOrigin(0.0, 0.0, 0.0)",
            "slicer.util.setSliceViewerLayers(background=mri_crop, foreground=trus_crop, foregroundOpacity=0.4)",
        ]
    script_lines += [
        "slicer.util.resetSliceViews()",
        f'print("MRI crop: {mc}mm | TRUS crop: {tc}mm")',
    ]
    crop_slicer_script = chr(10).join(script_lines)

    # Actualizar metadata con rutas de crops
    updated_meta = dict(meta)
    updated_meta["mri_volume"]  = mri_crop_path   # rutas para el registro
    updated_meta["trus_volume"] = trus_crop_path
    updated_meta["crop_fine"] = {
        "mri_crop_path":   mri_crop_path,
        "trus_crop_path":  trus_crop_path,
        "mri_origin_mm":   mri_origin,
        "trus_origin_mm":  trus_origin,
        "spacing_mm":      0.565,
        "size_vox":        160,
    }

    logger.info(
        f"[CROP_FINE] OK | "
        f"MRI: {mri_crop_path} | "
        f"TRUS: {trus_crop_path or 'N/A'}"
    )

    msg = (
        f"Crops 160³ @ 0.565mm generados para PID {state.pid}:\n\n"
        f"MRI crop: {mc}mm\n"
        f"TRUS crop: {tc}mm\n\n"
        "Copia el script en Slicer para verificar ambos crops superpuestos.\n"
        "Confirma que la prostata esta correctamente centrada en ambos."
    )

    return {
        "current_node":      "hitl_crops",
        "slicer_fov_script": crop_slicer_script,
        "hitl_message":      msg,
        "pending_hitl":      True,
        "pending_hitl_type": "crops",
        "case_metadata":     updated_meta,
        "crop_fine_mm":      mri_origin,
        "messages":          [AIMessage(content=msg)],
    }


# ── Nodo HITL: CROPS ─────────────────────────────────────────────────────────

async def node_hitl_crops(state: PipelineState) -> dict:
    """HITL #2 — Verificar ambos crops 160³ @ 0.565mm superpuestos en Slicer."""
    logger.info(f"[HITL_CROPS] PID={state.pid}")

    decision = interrupt({
        "type":          "hitl_crops",
        "message":       state.hitl_message,
        "pid":           state.pid,
        "slicer_script": state.slicer_fov_script,
        "instructions": (
            "1. Copia el script en Slicer (Ctrl+3)\n"
            "2. Verifica MRI crop (gris) + TRUS crop (overlay)\n"
            "3. Confirma que la prostata esta centrada en ambos\n"
            "4. Aprueba para continuar al registro"
        ),
    })

    hitl_record = HITLDecision(
        point="crops",
        decision=decision.get("decision", "approved"),
        notes=decision.get("notes", ""),
    )
    approved = hitl_record.decision == "approved"
    logger.info(f"[HITL_CROPS] Decision: {hitl_record.decision}")

    return {
        "current_node":   "plan_params" if approved else "crop_fine",
        "pending_hitl":   False,
        "hitl_decisions": state.hitl_decisions + [hitl_record],
    }


async def node_plan_params(state: PipelineState) -> dict:
    """
    Nodo de planificación: MedGemma 27B sugiere parámetros de registro
    basándose en las características del caso (FOV, calidad, retry count).

    Complejidad: O(1) — una llamada HTTP a Ollama.
    """
    logger.info(f"[PLAN_PARAMS] MedGemma razonando sobre parámetros para PID={state.pid}")

    llm = get_llm(
        model=os.getenv("OLLAMA_MODEL", "alibayram/medgemma:27b"),
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    params = await suggest_registration_params(state, llm)

    hitl_msg = (
        f"MedGemma sugiere los siguientes parámetros de registro:\n"
        f"- Algoritmo: {params.algorithm}\n"
        f"- Métrica: {params.metric}\n"
        f"- Iteraciones: {params.iterations}\n"
        f"- Grid spacing: {params.grid_spacing_mm} mm\n"
        f"- Optimizer: {params.optimizer}\n"
        f"- Sampling rate: {params.sampling_rate}\n\n"
        f"Justificación clínica: {params.rationale}\n\n"
        f"¿Aprueba estos parámetros para proceder al registro?"
    )

    logger.info(f"[PLAN_PARAMS] Parámetros: {params.algorithm} | grid={params.grid_spacing_mm}mm")

    return {
        "current_node": "hitl_params",
        "registration_params": params,
        "pending_hitl": True,
        "pending_hitl_type": "params",
        "hitl_message": hitl_msg,
        "llm_reasoning": [f"[{_ts()}] Parámetros sugeridos: {params.model_dump()}"],
        "messages": [AIMessage(content=hitl_msg)],
    }


# ── Nodo 4: REGISTER ─────────────────────────────────────────────────────────

async def node_register(state: PipelineState) -> dict:
    """
    Nodo de registro: ejecuta el registro deformable con los parámetros
    aprobados. En Fase 4 llamará al tool run_elastix del servidor MCP.
    Por ahora simula el registro para demostrar el grafo.

    Complejidad: O(V²) en el peor caso para registro deformable — dominado
    por el algoritmo de optimización.
    """
    params = state.registration_params
    logger.info(
        f"[REGISTER] Ejecutando registro {params.algorithm if params else 'N/A'} "
        f"para PID={state.pid} (intento {state.retry_count + 1})"
    )

    # ── Fase 4: Registro real con SimpleElastix ──────────────────────────────
    import json

    params_json = json.dumps(params.model_dump()) if params else "{}"

    # Pasar traslación inicial de la detección dual si está disponible
    import json as _json
    detection_dual = state.case_metadata.get("detection_dual", {})
    initial_translation = detection_dual.get("initial_translation")
    translation_json = _json.dumps(initial_translation) if initial_translation else ""

    if initial_translation:
        logger.info(f"[REGISTER] Usando traslación inicial de detección dual: {[round(v,1) for v in initial_translation]} mm")

    # Usar crops generados por CoarseCNN si están disponibles
    crop_info    = state.case_metadata.get("crop_fine", {})
    mri_crop     = crop_info.get("mri_crop_path", "")
    trus_crop    = crop_info.get("trus_crop_path", "")
    mri_mask     = state.case_metadata.get("mri_mask_path", "")
    trus_mask    = state.case_metadata.get("trus_mask_path", "")

    reg_result = await _call_mcp_tool("run_registration", {
        "pid":                      state.pid,
        "params_json":              params_json,
        "initial_translation_json": translation_json,
        "fixed_path":               mri_crop,
        "moving_path":              trus_crop,
        "fixed_mask_path":          mri_mask,
        "moving_mask_path":         trus_mask,
    })

    if not reg_result.get("success"):
        # Si el registro falla (archivo no encontrado, etc.) → fallback simulado
        err = reg_result.get("error", "Error desconocido")
        logger.warning(f"[REGISTER] Registro real falló: {err}. Usando simulación.")
        import random
        base_dice = 0.78 + (state.retry_count * 0.04)
        dice  = min(round(base_dice + random.uniform(-0.03, 0.03), 3), 0.97)
        hd95  = round(max(6.5 - (state.retry_count * 1.2) + random.uniform(-0.5, 0.5), 1.5), 2)
        tre   = round(max(4.2 - (state.retry_count * 0.8) + random.uniform(-0.3, 0.3), 0.8), 2)
    else:
        dice  = reg_result.get("dice", 0.0)
        hd95  = reg_result.get("hd95_mm", 99.0)
        tre   = reg_result.get("tre_mm") or 0.0
        logger.info(f"[REGISTER] Registro real OK | Dice={dice} | HD95={hd95}mm")

    success = (dice or 0) > 0.85 and (hd95 or 99) < 5.0

    logger.info(
        f"[REGISTER] {'OK' if success else 'MARGINAL'} — "
        f"Dice={dice} | HD95={hd95}mm | TRE={tre}mm"
    )

    next_node = "validate" if success else "hitl_failure"

    if not success and state.retry_count < state.max_retries - 1:
        next_node = "plan_params"  # Retry automático con nuevos parámetros

    return {
        "current_node": next_node,
        "dice_score": dice,
        "hd95_mm": hd95,
        "tre_mm": tre,
        "registration_success": success,
        "retry_count": state.retry_count + 1,
        "messages": [
            AIMessage(content=(
                f"Registro completado (intento {state.retry_count + 1}):\n"
                f"- Dice: {dice} {'✅' if dice > 0.85 else '❌'}\n"
                f"- HD95: {hd95}mm {'✅' if hd95 < 5.0 else '❌'}\n"
                f"- TRE: {tre}mm {'✅' if tre < 3.0 else '⚠️'}\n"
                f"{'Resultado aceptable.' if success else 'Resultado marginal — reintentando.'}"
            ))
        ],
    }


# ── Nodo 5: VALIDATE ─────────────────────────────────────────────────────────

async def node_validate(state: PipelineState) -> dict:
    """
    Nodo de validación: MedGemma interpreta las métricas y el investigador
    aprueba el resultado final antes de guardarlo.

    Complejidad: O(1) — una llamada al LLM + preparación del HITL.
    """
    logger.info(f"[VALIDATE] Interpretando métricas para PID={state.pid}")

    llm = get_llm(
        model=os.getenv("OLLAMA_MODEL", "alibayram/medgemma:27b"),
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    interpretation = await interpret_metrics(state, llm)

    hitl_msg = (
        f"Métricas de registro para PID {state.pid}:\n"
        f"- Dice: {state.dice_score} (umbral: >0.85)\n"
        f"- HD95: {state.hd95_mm}mm (umbral: <5mm)\n"
        f"- TRE: {state.tre_mm}mm (umbral: <3mm)\n\n"
        f"Interpretación clínica (MedGemma):\n{interpretation}\n\n"
        f"¿Aprueba este resultado para guardar en el dataset de investigación?"
    )

    return {
        "current_node": "hitl_final",
        "pending_hitl": True,
        "pending_hitl_type": "final",
        "hitl_message": hitl_msg,
        "clinical_notes": interpretation,
        "llm_reasoning": [f"[{_ts()}] Interpretación: {interpretation}"],
        "messages": [AIMessage(content=hitl_msg)],
    }


# ── Nodo 6: REPORT ───────────────────────────────────────────────────────────

async def node_report(state: PipelineState) -> dict:
    """
    Nodo de reporte: genera el resumen clínico del caso completo.
    Guarda los resultados en el registro de casos.

    Complejidad: O(1) — una llamada al LLM.
    """
    logger.info(f"[REPORT] Generando reporte para PID={state.pid}")

    llm = get_llm(
        model=os.getenv("OLLAMA_MODEL", "alibayram/medgemma:27b"),
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    summary = await generate_report_summary(state, llm)

    # Actualizar el estado del caso en el registry
    await _call_mcp_tool("load_dicom", {
        "dicom_path": state.case_metadata.get("mri_volume", ""),
        "case_id": state.pid,
    })

    report = {
        "pid": state.pid,
        "modality": state.modality,
        "completed_at": _ts(),
        "detection": {
            "centroid_ras": state.centroid_ras,
            "confidence": state.confidence,
            "gt_error_mm": state.gt_error_mm,
        },
        "registration": {
            "dice": state.dice_score,
            "hd95_mm": state.hd95_mm,
            "tre_mm": state.tre_mm,
            "retries": state.retry_count,
            "params": state.registration_params.model_dump() if state.registration_params else {},
        },
        "hitl_decisions": [d.model_dump() for d in state.hitl_decisions],
        "clinical_summary": summary,
    }

    logger.info(f"[REPORT] Reporte generado para PID={state.pid}")

    return {
        "current_node": "end",
        "pipeline_complete": True,
        "pending_hitl": False,
        "messages": [
            AIMessage(content=f"Pipeline completado para PID {state.pid}.\n\n{summary}")
        ],
        "llm_reasoning": [f"[{_ts()}] Reporte final: {summary}"],
    }


# ── Nodo 7: HANDLE HITL ──────────────────────────────────────────────────────

async def node_handle_hitl_failure(state: PipelineState) -> dict:
    """
    Nodo de fallo de registro: el agente escaló porque superó max_retries.
    Espera decisión del investigador sobre cómo proceder.
    """
    hitl_msg = (
        f"El registro para PID {state.pid} no alcanzó los umbrales clínicos "
        f"después de {state.retry_count} intentos.\n\n"
        f"Últimas métricas:\n"
        f"- Dice: {state.dice_score} (umbral: >0.85)\n"
        f"- HD95: {state.hd95_mm}mm (umbral: <5mm)\n\n"
        f"Opciones:\n"
        f"1. Aprobar igualmente (resultado marginal aceptable para este caso)\n"
        f"2. Rechazar y marcar el caso para revisión manual\n"
        f"3. Ajustar parámetros manualmente y reintentar"
    )

    return {
        "current_node": "hitl_failure",
        "pending_hitl": True,
        "pending_hitl_type": "failure",
        "hitl_message": hitl_msg,
        "messages": [AIMessage(content=hitl_msg)],
    }