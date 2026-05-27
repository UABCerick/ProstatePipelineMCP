"""
schemas.py — Modelos de validación Pydantic para el pipeline prostático.

Toda entrada al servidor MCP pasa primero por aquí.
Esto previene Prompt Injection y errores de dominio en origen.
"""
from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumeraciones de dominio ──────────────────────────────────────────────────

class ImagingModality(str, Enum):
    MRI  = "MRI"
    US   = "US"           # Ultrasonido
    CT   = "CT"
    PET  = "PET"
    UNKNOWN = "UNKNOWN"


class MRISequence(str, Enum):
    T2W   = "T2W"         # Morfología prostática
    DWI   = "DWI"         # Difusión
    DCE   = "DCE"         # Contraste dinámico
    ADC   = "ADC"         # Mapa de difusión aparente
    UNKNOWN = "UNKNOWN"


class SeriesQuality(str, Enum):
    ACCEPTABLE = "ACCEPTABLE"
    MARGINAL   = "MARGINAL"
    REJECTED   = "REJECTED"


# ── Input schemas (lo que recibe cada tool MCP) ───────────────────────────────

# Regex para bloquear path traversal y caracteres peligrosos
_SAFE_PATH_RE = re.compile(r'^[a-zA-Z0-9_\-/\\.\\: ]+$')
_SAFE_ID_RE   = re.compile(r'^[a-zA-Z0-9_\-]{1,64}$')


class LoadDicomInput(BaseModel):
    """
    Input para load_dicom.

    Seguridad: valida que la ruta no contenga path traversal (../../),
    caracteres de shell, ni apunte fuera del directorio de datos.
    """
    dicom_path: str = Field(
        ...,
        description="Ruta absoluta o relativa al directorio DICOM del caso.",
        examples=["/data/cases/patient_001/MRI"]
    )
    modality: ImagingModality = Field(
        ImagingModality.UNKNOWN,
        description="Modalidad esperada. Si es UNKNOWN se detecta automáticamente."
    )
    case_id: Optional[str] = Field(
        None,
        description="ID único del caso. Si no se provee, se genera desde el DICOM."
    )

    @field_validator("dicom_path")
    @classmethod
    def validate_path_safety(cls, v: str) -> str:
        # Bloquear path traversal
        if ".." in v:
            raise ValueError("Path traversal detectado en dicom_path. Operación bloqueada.")
        # Solo caracteres seguros
        if not _SAFE_PATH_RE.match(v):
            raise ValueError(
                "dicom_path contiene caracteres no permitidos. "
                "Solo se aceptan letras, números, guiones, puntos y separadores de ruta."
            )
        return v

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _SAFE_ID_RE.match(v):
            raise ValueError(
                "case_id inválido. Solo alfanumérico, guiones y underscores, máx 64 chars."
            )
        return v


class ValidateSeriesInput(BaseModel):
    """
    Input para validate_series.

    Valida que una serie DICOM cargada cumple requisitos mínimos
    de calidad para registro MRI-US prostático.
    """
    case_id: str = Field(..., description="ID del caso previamente cargado.")
    series_uid: str = Field(..., description="UID de la serie DICOM a validar.")
    expected_modality: ImagingModality = Field(
        ...,
        description="Modalidad esperada para esta serie."
    )
    expected_sequence: Optional[MRISequence] = Field(
        None,
        description="Secuencia MRI esperada (solo aplica si modality=MRI)."
    )
    min_slices: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Número mínimo de slices requeridos para aceptar la serie."
    )

    @field_validator("case_id", "series_uid")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        # UIDs DICOM pueden tener puntos además de alfanuméricos
        if not re.match(r'^[a-zA-Z0-9_\-\.]{1,128}$', v):
            raise ValueError(f"ID '{v}' contiene caracteres no permitidos.")
        return v

    @model_validator(mode="after")
    def validate_sequence_only_for_mri(self) -> "ValidateSeriesInput":
        if self.expected_sequence and self.expected_modality != ImagingModality.MRI:
            raise ValueError(
                "expected_sequence solo es válido cuando expected_modality=MRI."
            )
        return self


class GetMetadataInput(BaseModel):
    """
    Input para get_image_metadata.

    Extrae metadatos DICOM relevantes para el pipeline de registro.
    """
    case_id: str = Field(..., description="ID del caso.")
    series_uid: str = Field(..., description="UID de la serie.")
    fields: list[str] = Field(
        default_factory=lambda: [
            "PatientID", "Modality", "SeriesDescription",
            "SliceThickness", "PixelSpacing", "Rows", "Columns",
            "NumberOfSlices", "StudyDate", "SeriesInstanceUID"
        ],
        description="Tags DICOM a extraer. Solo tags de la whitelist son permitidos."
    )

    # Whitelist de tags DICOM permitidos (previene lectura de datos sensibles no necesarios)
    _ALLOWED_TAGS: set[str] = {
        "PatientID", "Modality", "SeriesDescription", "StudyDescription",
        "SliceThickness", "PixelSpacing", "Rows", "Columns",
        "NumberOfSlices", "StudyDate", "SeriesDate", "SeriesTime",
        "SeriesInstanceUID", "SOPClassUID", "ImageOrientationPatient",
        "ImagePositionPatient", "SliceLocation", "SpacingBetweenSlices",
        "ProtocolName", "MagneticFieldStrength", "RepetitionTime", "EchoTime",
    }

    @field_validator("fields")
    @classmethod
    def validate_allowed_fields(cls, v: list[str]) -> list[str]:
        blocked = [f for f in v if f not in cls._ALLOWED_TAGS]
        if blocked:
            raise ValueError(
                f"Tags DICOM no permitidos: {blocked}. "
                f"Solo se pueden solicitar tags de la whitelist clínica."
            )
        return v

    @field_validator("case_id", "series_uid")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_\-\.]{1,128}$', v):
            raise ValueError(f"ID inválido: '{v}'")
        return v


# ── Output schemas (lo que devuelve cada tool) ────────────────────────────────

class DicomSeriesInfo(BaseModel):
    """Información de una serie DICOM dentro de un caso."""
    series_uid: str
    modality: ImagingModality
    sequence: MRISequence
    description: str
    num_slices: int
    slice_thickness_mm: Optional[float]
    pixel_spacing_mm: Optional[tuple[float, float]]
    dimensions: tuple[int, int, int]   # (rows, cols, slices)
    file_count: int


class LoadDicomResult(BaseModel):
    """Resultado de cargar un directorio DICOM."""
    success: bool
    case_id: str
    dicom_path: str
    series_found: list[DicomSeriesInfo]
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class ValidationResult(BaseModel):
    """Resultado de validar una serie para registro prostático."""
    case_id: str
    series_uid: str
    quality: SeriesQuality
    modality_match: bool
    sequence_match: bool
    slice_count_ok: bool
    spacing_ok: bool
    issues: list[str] = Field(default_factory=list)
    recommendation: str = ""


class ImageMetadata(BaseModel):
    """Metadatos DICOM extraídos de una serie."""
    case_id: str
    series_uid: str
    tags: dict[str, str]
    extracted_at: str
