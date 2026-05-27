"""
registration_engine.py — Motor de registro deformable MRI-US usando SimpleElastix.

Implementa el registro B-spline deformable entre MRI T2W y TRUS prostático
usando SimpleElastix (ITK 6.0), con los parámetros sugeridos por MedGemma.

Pipeline de registro:
    1. Cargar volúmenes preprocesados (160³, 0.565mm isotrópico)
    2. Registro rígido inicial (alineación de centros)
    3. Registro deformable B-spline con parámetros del LLM
    4. Aplicar transformación a la máscara TRUS
    5. Calcular métricas vs máscara MRI (Dice, HD95, TRE)

Complejidad:
    registro rígido    : O(V × I_rigid)  — V=160³, I=iteraciones
    registro deformable: O(V × I_def × G³) — G=nodos del grid B-spline
    cálculo de métricas: O(V) — una pasada sobre los volúmenes

Referencias:
    Klein et al. (2010) - elastix: A Toolbox for Intensity-Based Medical Image Registration
    PI-RADS v2.1 - parámetros de calidad para MRI prostático
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
from loguru import logger

# RegistrationParams se importa lazy dentro de las funciones para evitar circular import


# ── Umbrales clínicos para registro MRI-US prostático ────────────────────────
# Basados en literatura de registro multimodal prostático
DICE_THRESHOLD    = 0.85   # PI-RADS: overlap mínimo aceptable
HD95_THRESHOLD_MM = 5.0    # Distancia de Hausdorff al 95% en mm
TRE_THRESHOLD_MM  = 3.0    # Target Registration Error en mm


# ── Parámetros elastix ────────────────────────────────────────────────────────

def _build_rigid_params() -> sitk.ParameterMap:
    """
    Parámetros para registro rígido inicial MRI-US.
    
    IMPORTANTE: MRI y TRUS tienen distribuciones de intensidad muy distintas.
    CenterOfGravity falla porque el centro de masa difiere enormemente.
    Usamos GeometricalCenter (centro del volumen) como inicialización —
    ambos volúmenes son 160³ al mismo spacing, así que el centro geométrico
    es equivalente y no depende de las intensidades.
    
    Complejidad: O(V × I) donde I = 300 iteraciones.
    """
    pm = sitk.GetDefaultParameterMap("rigid")
    pm["MaximumNumberOfIterations"]              = ["300"]
    pm["NumberOfResolutions"]                    = ["3"]
    pm["Metric"]                                 = ["AdvancedMattesMutualInformation"]
    pm["Optimizer"]                              = ["AdaptiveStochasticGradientDescent"]
    pm["NumberOfSpatialSamples"]                 = ["2000"]
    pm["NewSamplesEveryIteration"]               = ["true"]
    pm["AutomaticTransformInitialization"]       = ["true"]
    pm["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    pm["WriteResultImage"]                       = ["false"]
    pm["MaximumStepLength"]                      = ["1.0"]
    return pm


def _build_bspline_params(params: RegistrationParams) -> sitk.ParameterMap:
    """
    Parámetros B-spline deformable basados en sugerencia de MedGemma.

    grid_spacing_mm controla la flexibilidad de la deformación:
    - 16mm: deformación global (bueno para diferencias grandes)
    - 12mm: intermedio
    - 8mm:  deformación local fina (para ajuste preciso)

    Complejidad: O(V × I × (FOV/grid)³) — proporcional al número de nodos.
    """
    pm = sitk.GetDefaultParameterMap("bspline")

    # Iteraciones por resolución
    iterations = [str(i) for i in params.iterations]
    pm["MaximumNumberOfIterations"]  = [str(max(params.iterations))]
    pm["NumberOfResolutions"]        = [str(len(params.iterations))]

    # Métrica de similitud
    metric_map = {
        "NMI":   "AdvancedMattesMutualInformation",
        "MI":    "AdvancedMattesMutualInformation",
        "Mattes":"AdvancedMattesMutualInformation",
        "MSE":   "AdvancedMeanSquares",
        "CC":    "AdvancedNormalizedCorrelation",
    }
    pm["Metric"] = [metric_map.get(params.metric, "NormalizedMutualInformation")]

    # Grid B-spline
    pm["NumberOfHistogramBins"]             = ["64"]
    pm["FinalGridSpacingInPhysicalUnits"]   = [str(params.grid_spacing_mm)]
    pm["GridSpacingSchedule"]             = [
        str(params.grid_spacing_mm * (2 ** (len(params.iterations) - 1 - i)))
        for i in range(len(params.iterations))
    ]

    # Optimizer
    pm["Optimizer"]              = [params.optimizer]
    pm["NumberOfSpatialSamples"] = [str(int(160**3 * params.sampling_rate))]
    pm["NewSamplesEveryIteration"] = ["true"]
    pm["WriteResultImage"]       = ["false"]
    pm["WriteTransformParametersEachIteration"] = ["false"]

    return pm


# ── Motor de registro ─────────────────────────────────────────────────────────

def run_registration(
    fixed_path: str,           # MRI T2W (volumen fijo)
    moving_path: str,          # TRUS (volumen móvil)
    moving_mask_path: str,     # Máscara TRUS (se transforma junto con el TRUS)
    fixed_mask_path: str,      # Máscara MRI (referencia para métricas)
    params,  # RegistrationParams from langgraph_agent.state
    output_dir: Optional[str] = None,
    initial_translation: Optional[list] = None,  # traslación MRI→TRUS de detección dual
) -> dict:
    """
    Registro deformable MRI-US usando SimpleElastix.

    Flujo:
        1. Cargar volúmenes (ya preprocesados a 160³ 0.565mm)
        2. Registro rígido inicial (alineación de centros de masa)
        3. Registro B-spline deformable con parámetros del LLM
        4. Aplicar transformación a la máscara TRUS
        5. Calcular Dice, HD95, TRE sobre las máscaras

    Args:
        fixed_path:       ruta al NIfTI MRI normalizado
        moving_path:      ruta al NIfTI TRUS normalizado
        moving_mask_path: ruta a la máscara TRUS binaria
        fixed_mask_path:  ruta a la máscara MRI binaria
        params:           parámetros sugeridos por MedGemma
        output_dir:       carpeta para guardar resultados (opcional)

    Returns:
        dict con métricas, rutas de output y metadatos del registro
    """
    result = {
        "success": False,
        "algorithm": params.algorithm,
        "metric": params.metric,
        "grid_spacing_mm": params.grid_spacing_mm,
        "dice": None,
        "hd95_mm": None,
        "tre_mm": None,
        "output_dir": output_dir,
        "error": None,
    }

    try:
        # ── Cargar imágenes ───────────────────────────────────────────────────
        logger.info(f"Cargando volúmenes para registro...")
        fixed_img  = sitk.ReadImage(fixed_path,  sitk.sitkFloat32)
        moving_img = sitk.ReadImage(moving_path, sitk.sitkFloat32)
        fixed_mask  = sitk.ReadImage(fixed_mask_path)
        moving_mask = sitk.ReadImage(moving_mask_path)

        logger.info(
            f"Fixed (MRI):  {fixed_img.GetSize()} @ {fixed_img.GetSpacing()} mm | origin={fixed_img.GetOrigin()}\n"
            f"Moving (TRUS): {moving_img.GetSize()} @ {moving_img.GetSpacing()} mm | origin={moving_img.GetOrigin()}"
        )

        # ── Normalizar espacio físico ──────────────────────────────────────────
        # MRI y TRUS tienen crop_origin_mm distintos en el dataset.
        # Elastix falla si los orígenes son muy distintos porque los samples
        # del TRUS caen fuera del buffer del MRI.
        # Solución: resetear ambos a origen (0,0,0) y dirección identidad.
        # El preprocesamiento ya los llevó al mismo espacio 160³ a 0.565mm,
        # así que este reset es seguro y no distorsiona la geometría relativa.
        for img in [fixed_img, moving_img, fixed_mask, moving_mask]:
            img.SetOrigin((0.0, 0.0, 0.0))
            img.SetDirection((1,0,0, 0,1,0, 0,0,1))

        logger.info("Orígenes normalizados a (0,0,0) para registro.")

        # ── Registro rígido inicial ───────────────────────────────────────────
        logger.info("Paso 1/2: Registro rígido (alineación de centros)...")
        elastix_rigid = sitk.ElastixImageFilter()
        elastix_rigid.SetFixedImage(fixed_img)
        elastix_rigid.SetMovingImage(moving_img)
        elastix_rigid.SetParameterMap(_build_rigid_params())
        elastix_rigid.LogToConsoleOn()
        elastix_rigid.Execute()

        rigid_transform = elastix_rigid.GetTransformParameterMap()
        logger.info("Registro rígido completado.")

        # ── Registro deformable B-spline ──────────────────────────────────────
        logger.info(
            f"Paso 2/2: Registro B-spline deformable "
            f"(grid={params.grid_spacing_mm}mm, métrica={params.metric})..."
        )
        elastix_def = sitk.ElastixImageFilter()
        elastix_def.SetFixedImage(fixed_img)
        elastix_def.SetMovingImage(elastix_rigid.GetResultImage())
        elastix_def.SetParameterMap(_build_bspline_params(params))
        elastix_def.LogToConsoleOff()
        elastix_def.Execute()

        registered_img      = elastix_def.GetResultImage()
        deformable_transform = elastix_def.GetTransformParameterMap()
        logger.info("Registro deformable completado.")

        # ── Aplicar transformación a la máscara TRUS ──────────────────────────
        logger.info("Aplicando transformación a máscara TRUS...")

        # Primero aplicar la transformación rígida
        transformix_rigid = sitk.TransformixImageFilter()
        transformix_rigid.SetMovingImage(
            sitk.Cast(moving_mask, sitk.sitkFloat32)
        )
        transformix_rigid.SetTransformParameterMap(rigid_transform)
        transformix_rigid.LogToConsoleOff()
        transformix_rigid.Execute()
        mask_after_rigid = transformix_rigid.GetResultImage()

        # Luego aplicar la transformación deformable
        transformix_def = sitk.TransformixImageFilter()
        transformix_def.SetMovingImage(mask_after_rigid)
        transformix_def.SetTransformParameterMap(deformable_transform)
        transformix_def.LogToConsoleOff()
        transformix_def.Execute()
        transformed_mask = transformix_def.GetResultImage()

        # Binarizar la máscara transformada (interpolación puede crear valores no-binarios)
        transformed_mask_bin = sitk.BinaryThreshold(
            transformed_mask, lowerThreshold=0.5, upperThreshold=1e6,
            insideValue=1, outsideValue=0
        )

        # ── Calcular métricas ─────────────────────────────────────────────────
        logger.info("Calculando métricas de registro...")
        metrics = compute_registration_metrics(
            fixed_mask=fixed_mask,
            transformed_moving_mask=transformed_mask_bin,
        )

        result.update({
            "success": True,
            "dice":    metrics["dice"],
            "hd95_mm": metrics["hd95_mm"],
            "tre_mm":  metrics.get("tre_mm"),
        })

        # ── Guardar resultados opcionales ─────────────────────────────────────
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(registered_img, str(out / "registered_trus.nii.gz"))
            sitk.WriteImage(transformed_mask_bin, str(out / "transformed_trus_mask.nii.gz"))
            result["registered_trus_path"] = str(out / "registered_trus.nii.gz")
            result["transformed_mask_path"] = str(out / "transformed_trus_mask.nii.gz")
            logger.info(f"Resultados guardados en {output_dir}")

        logger.info(
            f"Registro completado | "
            f"Dice={metrics['dice']:.3f} | "
            f"HD95={metrics['hd95_mm']:.2f}mm"
        )

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        logger.error(f"Error en registro: {e}", exc_info=True)

    return result


# ── Métricas de registro ──────────────────────────────────────────────────────

def compute_registration_metrics(
    fixed_mask: sitk.Image,
    transformed_moving_mask: sitk.Image,
) -> dict:
    """
    Calcula métricas estándar de registro sobre las máscaras binarias.

    Dice: overlap volumétrico. O(V).
    HD95: distancia de Hausdorff al percentil 95. O(V log V).
    TRE:  error de registro en puntos de control. O(k) donde k = landmarks.

    Los valores clínicos de referencia para registro MRI-US prostático son:
    Dice > 0.85, HD95 < 5mm (basado en literatura de registro prostático).
    """
    # Asegurar mismo espacio físico
    transformed_moving_mask = sitk.Resample(
        transformed_moving_mask, fixed_mask,
        sitk.Transform(), sitk.sitkNearestNeighbor, 0.0,
        transformed_moving_mask.GetPixelID()
    )

    # Convertir a arrays numpy
    fixed_arr   = sitk.GetArrayFromImage(fixed_mask).astype(bool)
    moving_arr  = sitk.GetArrayFromImage(transformed_moving_mask).astype(bool)

    # ── Dice ──────────────────────────────────────────────────────────────────
    intersection = np.logical_and(fixed_arr, moving_arr).sum()
    dice = (2.0 * intersection) / (fixed_arr.sum() + moving_arr.sum() + 1e-8)

    # ── HD95 — Hausdorff Distance al 95% ─────────────────────────────────────
    spacing = np.array(fixed_mask.GetSpacing())  # mm por voxel (x,y,z)

    fixed_surface  = _get_surface_points(fixed_arr,  spacing)
    moving_surface = _get_surface_points(moving_arr, spacing)

    hd95 = _hausdorff_95(fixed_surface, moving_surface)

    return {
        "dice":    round(float(dice), 4),
        "hd95_mm": round(float(hd95), 3),
        "tre_mm":  None,  # Requiere landmarks manuales — se agrega en Fase 5
        "within_thresholds": dice > DICE_THRESHOLD and hd95 < HD95_THRESHOLD_MM,
    }


def _get_surface_points(mask: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """
    Extrae puntos de superficie de una máscara binaria 3D.
    Usa erosión morfológica para identificar el contorno. O(V).
    """
    from scipy.ndimage import binary_erosion
    eroded   = binary_erosion(mask)
    surface  = mask & ~eroded
    coords   = np.argwhere(surface).astype(float)  # (N, 3) en voxeles (z,y,x)
    # Convertir a mm — spacing es (sx,sy,sz) → aplicar como (sz,sy,sx) para (z,y,x)
    coords_mm = coords * spacing[::-1]
    return coords_mm


def _hausdorff_95(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """
    Distancia de Hausdorff al percentil 95 entre dos conjuntos de puntos.
    Usa scipy KDTree para eficiencia. O(N log N).
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("inf")

    from scipy.spatial import KDTree
    tree_b = KDTree(pts_b)
    tree_a = KDTree(pts_a)

    dist_a_to_b, _ = tree_b.query(pts_a)
    dist_b_to_a, _ = tree_a.query(pts_b)

    hd95 = max(
        np.percentile(dist_a_to_b, 95),
        np.percentile(dist_b_to_a, 95),
    )
    return float(hd95)