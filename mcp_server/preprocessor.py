"""
preprocessor.py — Tool de preprocesamiento para volúmenes crudos.

Reproduce el Script 1 del doctorado:
  DICOM/NIfTI crudo → resampleo 0.565mm → LPS → crop 160³ → z-score → NIfTI

Separado de localizer_model.py para poder usarlo como tool MCP independiente
y para que el pipeline sea: preprocess → save → detect (dos pasos explícitos).

Complejidad:
    preprocess_and_save: O(V_orig) dominado por resampleo isotrópico
    V_orig típico MRI:  256×256×60  ≈ 3.9M voxeles
    V_orig típico TRUS: 360×360×227 ≈ 29M voxeles
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
from loguru import logger

# Reusar las funciones del pipeline del doctorado
from .localizer_model import (
    SPACING_ISO,
    CROP_SIZE,
    HALF_FOV_MM,
    _resample_isotropic,
    _normalize_geometry,
    _crop_from_fov_center,
    _normalize_intensity,
)


def preprocess_and_save(
    input_path: str,
    output_dir: str,
    pid: str,
    modality: str,          # "MRI" o "TRUS"
    is_dicom: bool = True,
    overwrite: bool = False,
) -> dict:
    """
    Ejecuta el pipeline de preprocesamiento completo y guarda el resultado.

    Salidas en output_dir:
        {pid}_{modality}_loc_160.nii.gz       ← volumen normalizado
        {pid}_{modality}_preproc_meta.json     ← metadatos de trazabilidad

    Retorna dict con rutas, metadatos y estadísticas de normalización.

    Complejidad: O(V_orig) — dominado por el resampleo isotrópico.
    """
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol_name  = f"{pid}_{modality}_loc_160.nii.gz"
    meta_name = f"{pid}_{modality}_preproc_meta.json"
    vol_path  = out_dir / vol_name
    meta_path = out_dir / meta_name

    if vol_path.exists() and not overwrite:
        logger.info(f"Ya existe {vol_path.name} — saltando (overwrite=False)")
        return {
            "success": True,
            "skipped": True,
            "volume_path": str(vol_path),
            "message": "Archivo ya existe. Usa overwrite=True para regenerar.",
        }

    result: dict = {
        "success": False,
        "pid": pid,
        "modality": modality,
        "input_path": input_path,
        "volume_path": str(vol_path),
        "meta_path": str(meta_path),
    }

    try:
        t0 = datetime.now(timezone.utc)

        # ── Paso 1: Cargar imagen ─────────────────────────────────────────────
        if is_dicom:
            reader = sitk.ImageSeriesReader()
            files  = reader.GetGDCMSeriesFileNames(input_path)
            if not files:
                raise ValueError(f"No se encontraron DICOMs en: {input_path}")
            reader.SetFileNames(files)
            img = reader.Execute()
            logger.info(f"DICOM cargado: {len(files)} slices | {pid}")
        else:
            if not Path(input_path).exists():
                raise FileNotFoundError(f"NIfTI no encontrado: {input_path}")
            img = sitk.ReadImage(input_path)
            logger.info(f"NIfTI cargado: {input_path} | {pid}")

        orig_size    = list(img.GetSize())
        orig_spacing = list(img.GetSpacing())
        result["original_size"]    = orig_size
        result["original_spacing"] = orig_spacing

        # ── Paso 2: Resampleo isotrópico ──────────────────────────────────────
        img_iso = _resample_isotropic(img)
        iso_size = list(img_iso.GetSize())
        result["iso_size"] = iso_size
        result["iso_fov_mm"] = [
            round(iso_size[i] * SPACING_ISO, 2) for i in range(3)
        ]

        # ── Paso 3: Normalización geométrica ──────────────────────────────────
        img_iso = _normalize_geometry(img_iso)

        # ── Paso 4: Crop 160³ desde centro FOV ────────────────────────────────
        img_crop, crop_origin_mm = _crop_from_fov_center(img_iso)
        result["crop_origin_mm"] = crop_origin_mm.tolist()

        # ── Paso 5: Normalización de intensidad ───────────────────────────────
        arr = sitk.GetArrayFromImage(img_crop).astype("float32")

        p1  = float(np.percentile(arr, 1))
        p99 = float(np.percentile(arr, 99))
        arr_clip = np.clip(arr, p1, p99)
        mean = float(arr_clip.mean())
        std  = float(arr_clip.std())
        arr_norm = ((arr_clip - mean) / (std + 1e-5)).astype("float32")

        result["normalization"] = {
            "p1": round(p1, 4), "p99": round(p99, 4),
            "mean": round(mean, 4), "std": round(std, 4),
        }

        # ── Paso 6: Guardar NIfTI normalizado ─────────────────────────────────
        img_out = sitk.GetImageFromArray(arr_norm)
        img_out.SetSpacing([SPACING_ISO] * 3)
        img_out.SetOrigin(crop_origin_mm.tolist())
        img_out.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        sitk.WriteImage(img_out, str(vol_path), useCompression=True)

        # ── Metadatos de trazabilidad ─────────────────────────────────────────
        elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
        meta = {
            "pid": pid,
            "modality": modality,
            "processed_at": t0.isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "input_path": input_path,
            "volume_path": str(vol_path),
            "original_size": orig_size,
            "original_spacing": orig_spacing,
            "iso_size": iso_size,
            "iso_fov_mm": result["iso_fov_mm"],
            "crop_origin_mm": crop_origin_mm.tolist(),
            "normalization": result["normalization"],
            "spacing_out": SPACING_ISO,
            "crop_size": CROP_SIZE,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        result["success"]  = True
        result["elapsed_s"] = round(elapsed, 2)
        result["shape_out"] = list(arr_norm.shape)

        logger.info(
            f"Preprocesamiento OK | {pid}/{modality} | "
            f"{elapsed:.1f}s | → {vol_path.name}"
        )

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        logger.error(f"Error en preprocesamiento {pid}/{modality}: {e}", exc_info=True)

    return result
