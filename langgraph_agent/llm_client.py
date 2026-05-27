"""
llm_client.py — Cliente MedGemma 27B vía Ollama para razonamiento clínico.

MedGemma 27B está entrenado en datos médicos incluyendo imágenes de radiología,
histopatología prostática, y registros clínicos — ideal para razonar sobre
parámetros de registro MRI-US y justificar decisiones clínicas.

El cliente es stateless — el estado del pipeline se mantiene en LangGraph,
no en el LLM. Cada llamada es independiente y recibe el contexto completo.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from .state import PipelineState, RegistrationParams


# ── System prompt del dominio ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MedGemma, a specialized medical AI assistant integrated 
into a prostate MRI-US registration pipeline for doctoral research.

Your role is to reason about:
1. Deformable registration parameters (elastix/ANTs) based on image characteristics
2. Clinical quality assessment of MRI T2W and TRUS volumes  
3. Interpretation of registration metrics (Dice, HD95, TRE)
4. Prostate localization and FOV validation

Key clinical context:
- Images are preprocessed to 160³ voxels at 0.565mm isotropic spacing
- Prostate centroid detected by LocalizerUNet with typical error 1-4mm
- Registration targets: Dice > 0.85, HD95 < 5mm, TRE < 3mm (clinical thresholds)
- Dataset: TCIA prostate cases, T2W MRI + TRUS pairs

Always provide concise, clinically grounded reasoning.
When suggesting registration parameters, justify each choice based on image properties.
Respond in the same language as the user (Spanish or English)."""


def get_llm(
    model: str = "alibayram/medgemma:27b",
    host: str = "http://localhost:11434",
    temperature: float = 0.1,
    num_ctx: int = 4096,
) -> ChatOllama:
    """
    Instancia el cliente MedGemma 27B.
    temperature=0.1 para respuestas deterministas en decisiones clínicas.
    """
    return ChatOllama(
        model=model,
        base_url=host,
        temperature=temperature,
        num_ctx=num_ctx,
    )


# ── Prompts especializados ────────────────────────────────────────────────────

def prompt_suggest_registration_params(state: PipelineState) -> str:
    """
    Genera el prompt para que el LLM sugiera parámetros de registro.
    Incluye toda la información del caso disponible.
    """
    ras = [round(v, 1) for v in (state.centroid_ras or [])]
    conf = round((state.confidence or 0) * 100, 1)
    error = round(state.gt_error_mm or 0, 2) if state.gt_error_mm else "N/A"

    meta = state.case_metadata
    fov_mri  = meta.get("mri_iso_fov_mm", "N/A")
    fov_trus = meta.get("trus_iso_fov_mm", "N/A")
    norm_mri = meta.get("mri_normalization", {})
    margin   = meta.get("mri_min_margin_vox", "N/A")

    return f"""
Analyze this prostate registration case and suggest optimal elastix/ANTs parameters.

CASE INFORMATION:
- PID: {state.pid}
- Modality: {state.modality}
- Detected centroid (RAS): {ras} mm
- Detection confidence: {conf}%
- Error vs ground truth: {error} mm
- MRI FOV: {fov_mri} mm
- TRUS FOV: {fov_trus} mm  
- MRI normalization: p1={norm_mri.get('p1','N/A')}, p99={norm_mri.get('p99','N/A')}, mean={norm_mri.get('mean') or 'N/A'}, std={norm_mri.get('std') or 'N/A'}
- Min margin (vox): {margin}
- FOV radius: {state.fov_radius_mm} mm
- Retry attempt: {state.retry_count}/{state.max_retries}

Based on these characteristics, suggest deformable registration parameters.
FIXED CONSTRAINTS (do not change these):
- Voxel size: 160³ (fixed)
- Spacing: 0.565mm isotropic (fixed)
- Keep registration fast: iterations [100-200, 50-100, 25-50]

You may suggest any values for: metric, grid_spacing_mm (8-16mm), optimizer, sampling_rate.

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
  "algorithm": "elastix",
  "metric": "AdvancedMattesMutualInformation",
  "iterations": [150, 75, 50],
  "grid_spacing_mm": 12.0,
  "optimizer": "AdaptiveStochasticGradientDescent",
  "sampling_rate": 0.1,
  "rationale": "Brief clinical justification in 2-3 sentences"
}}
"""


def prompt_interpret_metrics(state: PipelineState) -> str:
    """
    Prompt para que el LLM interprete las métricas de registro
    y decida si el resultado es clínicamente aceptable.
    """
    return f"""
Interpret these prostate MRI-US registration metrics for case PID {state.pid}:

REGISTRATION RESULTS:
- Dice score: {state.dice_score or 'N/A'}
- HD95: {state.hd95_mm or 'N/A'} mm  
- TRE: {state.tre_mm or 'N/A'} mm
- Retry attempt: {state.retry_count}/{state.max_retries}
- Parameters used: {state.registration_params.model_dump() if state.registration_params else 'N/A'}

Clinical thresholds for prostate registration:
- Dice > 0.85: acceptable overlap
- HD95 < 5mm: acceptable surface distance
- TRE < 3mm: acceptable target registration error

Provide a brief clinical interpretation (2-3 sentences) of whether this registration
is acceptable for clinical use and research purposes.
Respond in Spanish.
"""


def prompt_generate_report_summary(state: PipelineState) -> str:
    """
    Prompt para generar el resumen clínico del caso.
    """
    decisions_text = "\n".join([
        f"- HITL {d.point}: {d.decision} — {d.notes}"
        for d in state.hitl_decisions
    ])

    return f"""
Generate a brief clinical summary (3-4 sentences in Spanish) for this 
prostate MRI-US registration case:

PID: {state.pid}
Modality: {state.modality}
Detection error: {round(state.gt_error_mm or 0, 2)} mm
Registration Dice: {state.dice_score or 'pendiente'}
HD95: {state.hd95_mm or 'pendiente'} mm
HITL decisions:
{decisions_text or 'Ninguna registrada'}
Clinical notes: {state.clinical_notes or 'N/A'}

Summarize the pipeline execution, quality of results, and any recommendations
for the research dataset.
"""


# ── Funciones de llamada al LLM ───────────────────────────────────────────────

async def suggest_registration_params(
    state: PipelineState,
    llm: Optional[ChatOllama] = None,
) -> RegistrationParams:
    """
    Llama a MedGemma para sugerir parámetros de registro.
    Parsea la respuesta JSON de forma segura.
    Retorna parámetros default si el LLM falla.

    Complejidad: O(1) — una llamada HTTP al servidor Ollama.
    """
    if llm is None:
        llm = get_llm()

    prompt = prompt_suggest_registration_params(state)

    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await llm.ainvoke(messages)
        raw = response.content.strip()

        # Limpiar markdown si viene con ```json
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)

        data = json.loads(raw)
        params = RegistrationParams(**data)
        logger.info(
            f"LLM sugirió parámetros: {params.algorithm} | "
            f"metric={params.metric} | grid={params.grid_spacing_mm}mm"
        )
        return params

    except json.JSONDecodeError as e:
        logger.warning(f"LLM retornó JSON inválido: {e}. Usando defaults.")
        return _default_params(state)
    except Exception as e:
        logger.error(f"Error llamando al LLM: {e}")
        return _default_params(state)


async def interpret_metrics(
    state: PipelineState,
    llm: Optional[ChatOllama] = None,
) -> str:
    """
    Interpreta métricas de registro con contexto clínico.
    Retorna texto en español con la interpretación.
    """
    if llm is None:
        llm = get_llm()

    prompt = prompt_interpret_metrics(state)

    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await llm.ainvoke(messages)
        interpretation = response.content.strip()
        logger.info(f"LLM interpretó métricas para PID {state.pid}")
        return interpretation
    except Exception as e:
        logger.error(f"Error interpretando métricas: {e}")
        return f"Error al interpretar métricas: {e}"


async def generate_report_summary(
    state: PipelineState,
    llm: Optional[ChatOllama] = None,
) -> str:
    """Genera resumen clínico del caso para el reporte."""
    if llm is None:
        llm = get_llm()

    prompt = prompt_generate_report_summary(state)

    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await llm.ainvoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generando reporte: {e}")
        return f"Error al generar resumen: {e}"


def _default_params(state: PipelineState) -> RegistrationParams:
    """
    Parámetros default basados en literatura de registro MRI-US prostático.
    Se usan cuando el LLM falla o retorna JSON inválido.
    """
    # Ajustar grid spacing según retry count — estrategia de refinamiento
    grid = {0: 16.0, 1: 12.0, 2: 8.0}.get(state.retry_count, 8.0)

    return RegistrationParams(
        algorithm="elastix",
        metric="AdvancedMattesMutualInformation",
        iterations=[150, 75, 50],
        grid_spacing_mm=grid,
        optimizer="AdaptiveStochasticGradientDescent",
        sampling_rate=0.1,
        rationale=(
            f"Parámetros optimizados para demo (intento {state.retry_count + 1}). "
            f"Grid spacing {grid}mm, iteraciones reducidas para completar en ~2 minutos."
        ),
    )