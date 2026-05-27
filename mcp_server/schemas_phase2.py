"""
schemas_phase2.py — Schemas de validación para la Fase 2.

Extiende schemas.py con los tipos de la Fase 2:
  - DetectProstateInput  : entrada al tool detect_prostate_center
  - DetectionResult      : resultado estructurado de la detección
"""
from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

_SAFE_PATH_RE = re.compile(r'^[a-zA-Z0-9_\-/\\.\\: ]+$')


class ImageModalityDetect(str, Enum):
    MRI  = "MRI"
    TRUS = "TRUS"


class DetectProstateInput(BaseModel):
    """
    Input para detect_prostate_center.

    Validaciones:
      - image_path: no path traversal, solo caracteres seguros
      - modality: solo MRI o TRUS
      - threshold: rango clínico [0.001, 0.5] — valores fuera de rango
        indican error de configuración, no intención maliciosa
    """
    image_path: str = Field(
        ...,
        description="Ruta al directorio DICOM (MRI) o archivo .nii.gz (TRUS).",
        examples=[
            "C:/data/cases/patient_001/MRI",
            "C:/data/cases/patient_001/trus.nii.gz",
        ],
    )
    modality: ImageModalityDetect = Field(
        ...,
        description="Modalidad de imagen: MRI (DICOM) o TRUS (NIfTI).",
    )
    case_id: Optional[str] = Field(
        None,
        description="ID del caso. Si no se provee, se genera automáticamente.",
    )
    threshold: float = Field(
        default=0.01,
        ge=0.001,
        le=0.5,
        description=(
            "Umbral mínimo del heatmap para considerar activación válida. "
            "Rango clínico: [0.001, 0.5]. Default=0.01 (probado en el dataset)."
        ),
    )
    visualize_in_slicer: bool = Field(
        default=True,
        description="Si True, muestra el FOV como esfera ROI en 3D Slicer.",
    )

    @field_validator("image_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if ".." in v:
            raise ValueError("Path traversal detectado. Operación bloqueada.")
        if not _SAFE_PATH_RE.match(v):
            raise ValueError(
                "image_path contiene caracteres no permitidos. "
                "Solo letras, números, guiones, puntos y separadores de ruta."
            )
        return v

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', v):
            raise ValueError("case_id inválido: solo alfanumérico, guiones y underscores.")
        return v


class HITLDecision(str, Enum):
    APPROVED  = "APPROVED"
    REJECTED  = "REJECTED"
    ADJUSTED  = "ADJUSTED"


class DetectionResult(BaseModel):
    """
    Resultado de detect_prostate_center.

    Incluye toda la información necesaria para el HITL:
    el investigador ve el FOV propuesto y decide aprobarlo,
    rechazarlo, o ajustar manualmente el centro.
    """
    success: bool
    case_id: str = ""
    modality: str = ""

    # Resultados de la detección
    centroid_vox: Optional[list[float]] = Field(
        None,
        description="Centro detectado (z, y, x) en voxeles del cubo 160³.",
    )
    centroid_global_mm: Optional[list[float]] = Field(
        None,
        description="Centro detectado (x, y, z) en mm — espacio global LPS.",
    )
    centroid_ras: Optional[list[float]] = Field(
        None,
        description="Centro detectado (x, y, z) en mm — espacio RAS (Slicer).",
    )
    crop_fine_mm: Optional[list[float]] = Field(
        None,
        description="Origen del crop fino para FineCNN (x, y, z) en mm.",
    )
    confidence: Optional[float] = Field(
        None,
        description="Valor máximo del heatmap ∈ [0,1]. Proxy de confianza.",
    )
    fov_radius_mm: float = Field(
        default=45.2,
        description="Radio del FOV en mm (fijo = HALF_FOV_MM = 45.2mm).",
    )

    # Estado HITL
    hitl_required: bool = Field(
        default=True,
        description="Siempre True — el investigador debe confirmar el FOV.",
    )
    hitl_message: str = Field(
        default="",
        description="Mensaje para el investigador en el punto HITL.",
    )
    slicer_visualization: bool = Field(
        default=False,
        description="True si el FOV fue visualizado en 3D Slicer.",
    )
    slicer_fov_script: str = Field(
        default="",
        description="Script Python listo para pegar en Slicer Python Interactor.",
    )

    # Metadatos de procesamiento
    model_path: str = ""
    device_used: str = ""
    preprocessing_notes: list[str] = Field(default_factory=list)
    error: Optional[str] = None