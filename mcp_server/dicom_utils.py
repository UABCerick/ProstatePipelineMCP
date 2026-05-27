"""
dicom_utils.py — Utilidades de procesamiento DICOM para próstata.

Encapsula la lógica médica de carga y validación de series.
Separado del servidor MCP para testear en aislamiento.

Complejidad:
  - scan_directory   : O(n_files)  — lectura de headers
  - validate_series  : O(s)        — s = num slices
  - extract_metadata : O(t)        — t = num tags solicitados
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
import SimpleITK as sitk
from loguru import logger

from .schemas import (
    DicomSeriesInfo,
    ImagingModality,
    MRISequence,
    SeriesQuality,
    ValidationResult,
    ImageMetadata,
)


# ── Heurísticas de detección de modalidad/secuencia ──────────────────────────

_SEQUENCE_KEYWORDS: dict[MRISequence, list[str]] = {
    MRISequence.T2W: ["t2", "t2w", "t2_tse", "t2 tse", "t2_tra", "tse"],
    MRISequence.DWI: ["dwi", "diffusion", "b0", "b1000", "b2000"],
    MRISequence.DCE: ["dce", "dynamic", "contrast", "perf"],
    MRISequence.ADC: ["adc", "apparent diffusion"],
}

def _detect_mri_sequence(series_description: str) -> MRISequence:
    """
    Infiere la secuencia MRI desde la descripción de la serie. O(k) donde
    k = número total de keywords — constante, efectivamente O(1).
    """
    desc_lower = series_description.lower()
    for seq, keywords in _SEQUENCE_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return seq
    return MRISequence.UNKNOWN


def _str_to_modality(dicom_modality: str) -> ImagingModality:
    """Convierte el tag Modality de DICOM a nuestro enum. O(1)."""
    mapping = {"MR": ImagingModality.MRI, "US": ImagingModality.US,
               "CT": ImagingModality.CT, "PT": ImagingModality.PET}
    return mapping.get(dicom_modality.upper(), ImagingModality.UNKNOWN)


# ── Carga de directorios DICOM ────────────────────────────────────────────────

def scan_dicom_directory(dicom_path: str) -> list[DicomSeriesInfo]:
    """
    Escanea un directorio DICOM y agrupa archivos por SeriesInstanceUID.
    Retorna una lista de DicomSeriesInfo.

    Complejidad: O(n) donde n = número de archivos DICOM en el directorio.
    Solo lee el header de cada archivo, no los píxeles.
    """
    path = Path(dicom_path)
    if not path.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {dicom_path}")

    # Agrupar por SeriesInstanceUID — O(n)
    series_map: dict[str, list[pydicom.Dataset]] = {}
    for f in path.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
            series_map.setdefault(uid, []).append(ds)
        except pydicom.errors.InvalidDicomError:
            logger.warning(f"Archivo no DICOM ignorado: {f.name}")
        except Exception as e:
            logger.warning(f"Error leyendo {f.name}: {e}")

    if not series_map:
        # Intentar sin extensión .dcm
        for f in path.rglob("*"):
            if f.is_file() and not f.suffix.lower() in (".json", ".txt", ".xml"):
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                    uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
                    series_map.setdefault(uid, []).append(ds)
                except Exception:
                    pass

    results: list[DicomSeriesInfo] = []
    for uid, slices in series_map.items():
        ds = slices[0]  # representante de la serie
        modality = _str_to_modality(str(getattr(ds, "Modality", "UNKNOWN")))
        description = str(getattr(ds, "SeriesDescription", "Sin descripción"))
        sequence = _detect_mri_sequence(description) if modality == ImagingModality.MRI else MRISequence.UNKNOWN

        # Extraer espaciado — puede estar ausente en US
        ps = getattr(ds, "PixelSpacing", None)
        pixel_spacing = (float(ps[0]), float(ps[1])) if ps and len(ps) >= 2 else None
        slice_thickness = float(getattr(ds, "SliceThickness", 0)) or None

        rows = int(getattr(ds, "Rows", 0))
        cols = int(getattr(ds, "Columns", 0))

        results.append(DicomSeriesInfo(
            series_uid=uid,
            modality=modality,
            sequence=sequence,
            description=description,
            num_slices=len(slices),
            slice_thickness_mm=slice_thickness,
            pixel_spacing_mm=pixel_spacing,
            dimensions=(rows, cols, len(slices)),
            file_count=len(slices),
        ))
        logger.debug(f"Serie detectada: {uid[:20]}... | {modality.value} | {len(slices)} slices | {description}")

    logger.info(f"Scan completo: {len(results)} series en {dicom_path}")
    return results


# ── Validación clínica de series ──────────────────────────────────────────────

# Umbrales clínicos para registro MRI-US prostático
# Basados en guías PI-RADS v2.1 y literatura de registro
_MIN_SLICES_MRI = 16        # T2W con <16 slices es insuficiente
_MIN_SLICES_US  = 10
_MAX_SLICE_THICKNESS_MRI_MM = 4.0   # >4mm compromete resolución axial
_MIN_IN_PLANE_RES_MM = 0.8          # Resolución mínima en plano

def validate_series_for_registration(
    series_info: DicomSeriesInfo,
    expected_modality: ImagingModality,
    expected_sequence: Optional[MRISequence],
    min_slices: int,
) -> ValidationResult:
    """
    Valida si una serie cumple requisitos para registro MRI-US prostático.

    Aplica criterios clínicos de calidad basados en PI-RADS v2.1.
    Complejidad: O(1) — solo comparaciones sobre metadatos ya cargados.
    """
    issues: list[str] = []

    # 1. Verificar modalidad
    modality_match = series_info.modality == expected_modality
    if not modality_match:
        issues.append(
            f"Modalidad incorrecta: se esperaba {expected_modality.value}, "
            f"se encontró {series_info.modality.value}."
        )

    # 2. Verificar secuencia (solo MRI)
    sequence_match = True
    if expected_sequence and series_info.modality == ImagingModality.MRI:
        sequence_match = series_info.sequence == expected_sequence
        if not sequence_match:
            issues.append(
                f"Secuencia incorrecta: se esperaba {expected_sequence.value}, "
                f"se detectó {series_info.sequence.value}."
            )

    # 3. Verificar número de slices
    effective_min = max(
        min_slices,
        _MIN_SLICES_MRI if series_info.modality == ImagingModality.MRI else _MIN_SLICES_US
    )
    slice_count_ok = series_info.num_slices >= effective_min
    if not slice_count_ok:
        issues.append(
            f"Insuficientes slices: {series_info.num_slices} < {effective_min} requeridos."
        )

    # 4. Verificar espaciado (si está disponible)
    spacing_ok = True
    if series_info.slice_thickness_mm and series_info.modality == ImagingModality.MRI:
        if series_info.slice_thickness_mm > _MAX_SLICE_THICKNESS_MRI_MM:
            spacing_ok = False
            issues.append(
                f"Grosor de slice {series_info.slice_thickness_mm}mm excede el máximo "
                f"recomendado de {_MAX_SLICE_THICKNESS_MRI_MM}mm para T2W prostático."
            )
    if series_info.pixel_spacing_mm:
        min_ps = min(series_info.pixel_spacing_mm)
        if min_ps > _MIN_IN_PLANE_RES_MM * 1.5:
            spacing_ok = False
            issues.append(
                f"Resolución en plano {min_ps:.2f}mm puede ser insuficiente para segmentación."
            )

    # Determinar calidad global
    if not modality_match or not slice_count_ok:
        quality = SeriesQuality.REJECTED
    elif issues:
        quality = SeriesQuality.MARGINAL
    else:
        quality = SeriesQuality.ACCEPTABLE

    # Recomendación en lenguaje natural
    if quality == SeriesQuality.ACCEPTABLE:
        recommendation = "Serie apta para registro MRI-US. Puede proceder al siguiente paso."
    elif quality == SeriesQuality.MARGINAL:
        recommendation = (
            "Serie marginalmente aceptable. Se recomienda revisión antes de proceder. "
            f"Problemas: {'; '.join(issues)}"
        )
    else:
        recommendation = (
            "Serie rechazada para registro. "
            "Se requiere adquirir una nueva serie con los parámetros correctos."
        )

    return ValidationResult(
        case_id="",   # se rellena en el tool
        series_uid=series_info.series_uid,
        quality=quality,
        modality_match=modality_match,
        sequence_match=sequence_match,
        slice_count_ok=slice_count_ok,
        spacing_ok=spacing_ok,
        issues=issues,
        recommendation=recommendation,
    )


# ── Extracción de metadatos ───────────────────────────────────────────────────

def extract_metadata(
    dicom_path: str,
    series_uid: str,
    fields: list[str],
) -> dict[str, str]:
    """
    Extrae tags DICOM específicos de la primera imagen de una serie. O(n_files).

    Solo extrae los tags solicitados (ya validados en GetMetadataInput).
    Retorna valores como strings para serialización JSON segura.
    """
    path = Path(dicom_path)
    target_file: Optional[Path] = None

    # Encontrar el primer archivo de la serie — O(n_files)
    for f in path.rglob("*"):
        if f.is_file():
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                if str(getattr(ds, "SeriesInstanceUID", "")) == series_uid:
                    target_file = f
                    break
            except Exception:
                continue

    if not target_file:
        raise FileNotFoundError(f"Serie {series_uid} no encontrada en {dicom_path}")

    ds = pydicom.dcmread(str(target_file), stop_before_pixels=True)
    result: dict[str, str] = {}
    for field in fields:
        val = getattr(ds, field, None)
        if val is None:
            result[field] = "N/A"
        elif hasattr(val, "__iter__") and not isinstance(val, str):
            result[field] = str(list(val))
        else:
            result[field] = str(val)

    return result
