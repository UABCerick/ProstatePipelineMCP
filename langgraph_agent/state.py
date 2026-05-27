"""
state.py — Estado del grafo LangGraph para el pipeline prostático.

El estado es la "memoria de trabajo" del agente en cada sesión.
El checkpointer SQLite persiste este estado entre sesiones completas.

Cada campo tiene un reducer que define cómo se actualiza cuando
un nodo retorna un valor parcial — por defecto reemplaza el valor anterior.
Los campos de tipo list usan el reducer add_messages o append.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RegistrationParams(BaseModel):
    """
    Parámetros de registro deformable sugeridos por el LLM.
    Basados en las características del caso (FOV, calidad de imagen, etc.)
    """
    algorithm: str = "elastix"              # elastix | ants
    metric: str = "NMI"                     # NMI | MI | MSE | CC
    iterations: list[int] = Field(
        default_factory=lambda: [500, 250, 100]
    )
    grid_spacing_mm: float = 16.0           # spacing del B-spline grid
    optimizer: str = "AdaptiveStochasticGradientDescent"
    sampling_rate: float = 0.1             # fracción de vóxeles usados
    rationale: str = ""                    # justificación del LLM


class HITLDecision(BaseModel):
    """Decisión del investigador en un punto HITL."""
    point: str                             # "fov" | "params" | "failure" | "final"
    decision: str                          # "approved" | "rejected" | "adjusted"
    notes: str = ""
    adjusted_value: Optional[Any] = None  # valor ajustado manualmente si aplica


class PipelineState(BaseModel):
    """
    Estado completo del pipeline de registro MRI-US.

    Campos persistidos por el checkpointer SQLite entre sesiones.
    Campos con Annotated[list, add_messages] acumulan historial.
    """

    # ── Identificación del caso ───────────────────────────────────────────────
    case_id: str = ""
    pid: str = ""
    modality: str = "MRI"
    session_id: str = ""

    # ── Resultados de detección ───────────────────────────────────────────────
    centroid_vox: Optional[list[float]] = None
    centroid_ras: Optional[list[float]] = None
    centroid_global_mm: Optional[list[float]] = None
    crop_fine_mm: Optional[list[float]] = None
    confidence: Optional[float] = None
    fov_radius_mm: float = 45.2
    heatmap_path: str = ""
    slicer_fov_script: str = ""
    gt_error_mm: Optional[float] = None

    # ── Parámetros de registro ────────────────────────────────────────────────
    registration_params: Optional[RegistrationParams] = None
    retry_count: int = 0
    max_retries: int = 3

    # ── Métricas de validación ────────────────────────────────────────────────
    dice_score: Optional[float] = None
    hd95_mm: Optional[float] = None
    tre_mm: Optional[float] = None
    registration_success: bool = False

    # ── Decisiones HITL ──────────────────────────────────────────────────────
    hitl_decisions: list[HITLDecision] = Field(default_factory=list)
    pending_hitl: bool = False
    pending_hitl_type: str = ""        # "fov" | "params" | "failure" | "final"
    hitl_message: str = ""

    # ── Estado del pipeline ───────────────────────────────────────────────────
    current_node: str = "start"
    pipeline_complete: bool = False
    error: Optional[str] = None

    # ── Historial de mensajes LLM (acumulativo) ───────────────────────────────
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    # ── Notas del LLM (razonamiento clínico) ─────────────────────────────────
    llm_reasoning: list[str] = Field(default_factory=list)
    clinical_notes: str = ""

    # ── Metadatos del caso (del dataset) ─────────────────────────────────────
    case_metadata: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
