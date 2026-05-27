"""graph.py — Grafo LangGraph para pipeline prostático MRI-TRUS."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from loguru import logger

from .state import PipelineState, HITLDecision
from .nodes import _call_slicer_tool
from .nodes import (
    node_load,
    node_preprocess,
    node_detect,
    node_crop_fine,
    node_plan_params,
    node_register,
    node_validate,
    node_report,
    node_handle_hitl_failure,
)


# ── HITL: FOV (MRI + TRUS juntos) ────────────────────────────────────────────

async def node_hitl_fov(state: PipelineState) -> dict:
    """HITL #1 — Verificar centroides MRI y TRUS en vol_iso completo."""
    logger.info(f"[HITL_FOV] PID={state.pid}")

    meta  = state.case_metadata
    dual  = meta.get("detection_dual", {})
    trus_c    = dual.get("trus_centroid_mm", [])
    trus_prob = dual.get("trus_max_prob", 0)
    trus_iso  = meta.get("trus_vol_iso_path", "")
    trus_heat = dual.get("trus_heatmap", "")

    safe_trus   = trus_iso.replace("\\\\", "/").replace("\\", "/") if trus_iso else ""
    safe_heat_t = trus_heat.replace("\\\\", "/").replace("\\", "/") if trus_heat else ""
    tc = [round(v,1) for v in trus_c] if trus_c else []
    tx = tc[0] if tc else 0
    ty = tc[1] if tc else 0
    tz = tc[2] if tc else 0

    trus_script = chr(10).join([
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
        "slicer.util.setSliceViewerLayers(background=vol_t, foreground=heat_t, foregroundOpacity=0.5)",
        f'fid_t = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Centroid_Coarse_TRUS_{state.pid}")',
        f"fid_t.AddControlPoint({-tx}, {-ty}, {tz})",
        "fid_t.GetDisplayNode().SetSelectedColor(1.0, 0.5, 0.0)",
        "fid_t.GetDisplayNode().SetGlyphScale(3.5)",
        "fid_t.GetDisplayNode().SetSliceProjection(True)",
        "slicer.util.resetSliceViews()",
        f'print("TRUS centroide: ({tx},{ty},{tz})mm | prob={trus_prob:.3f}")',
    ]) if safe_trus else 'print("TRUS no disponible")'

    mri_c    = dual.get("mri_centroid_mm", [])
    mri_prob = dual.get("mri_max_prob", 0)
    mri_iso  = meta.get("mri_vol_iso_path", "").replace("\\\\", "/").replace("\\", "/")
    mri_heat = dual.get("mri_heatmap", "").replace("\\\\", "/").replace("\\", "/")
    mc = [round(v,1) for v in mri_c] if mri_c else []
    rx = -mc[0] if mc else 0
    ry = -mc[1] if mc else 0
    rz =  mc[2] if mc else 0

    mri_script = chr(10).join([
        "import slicer",
        f'vol_m = slicer.util.loadVolume(r"{mri_iso}")',
        f'vol_m.SetName("{state.pid}_MRI_vol_iso")',
        "vol_m.SetOrigin(0.0, 0.0, 0.0)",
        f'heat_m = slicer.util.loadVolume(r"{mri_heat}")',
        f'heat_m.SetName("MRI_Heatmap_{state.pid}")',
        "heat_m.SetOrigin(0.0, 0.0, 0.0)",
        "dn_m = heat_m.GetDisplayNode()",
        'dn_m.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")',
        "dn_m.SetAutoWindowLevel(False)",
        "dn_m.SetWindowLevelMinMax(0.05, 1.0)",
        "dn_m.SetOpacity(0.7)",
        f'fid_m = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Centroid_Coarse_MRI_{state.pid}")',
        f"fid_m.AddControlPoint({rx}, {ry}, {rz})",
        "fid_m.GetDisplayNode().SetSelectedColor(0.2, 1.0, 0.3)",
        "fid_m.GetDisplayNode().SetGlyphScale(3.5)",
        "fid_m.GetDisplayNode().SetSliceProjection(True)",
        f'print("MRI centroide: ({mc[0] if mc else 0},{mc[1] if mc else 0},{mc[2] if mc else 0})mm | prob={mri_prob:.3f}")',
    ]) if mri_iso else 'print("MRI no disponible")'

    combined = chr(10).join([
        mri_script, "",
        "# ══ TRUS ══",
        trus_script, "",
        "try:",
        "    slicer.util.setSliceViewerLayers(background=vol_m, foreground=heat_m, foregroundOpacity=0.5)",
        "    slicer.util.resetSliceViews()",
        "except: pass",
    ])

    hitl_msg = (
        f"Centroides detectados para PID {state.pid}:\n"
        f"MRI: {mc}mm | prob={mri_prob:.3f}\n"
        f"TRUS: {tc}mm | prob={trus_prob:.3f}\n"
        f"Traslacion: {dual.get('translation_norm_mm', 0):.1f}mm\n\n"
        "Copia el script en Slicer (Ctrl+3).\n"
        "Verifica centroide MRI (verde) y TRUS (naranja).\n"
        "El crop 160³ se aplica DESPUES de aprobar."
    )

    # Intentar cargar volúmenes en Slicer automáticamente
    slicer_loaded = False
    slicer_error  = ""
    try:
        # Limpiar escena anterior
        await _call_slicer_tool("execute_python_code", {
            "code": "slicer.mrmlScene.Clear(0)"
        })
        # Cargar MRI vol_iso + heatmap
        await _call_slicer_tool("execute_python_code", {
            "code": combined.replace('"', '\"')
        })
        slicer_loaded = True
        logger.info(f"[HITL_FOV] Volúmenes cargados en Slicer automáticamente")
    except RuntimeError as e:
        slicer_error = str(e)
        logger.warning(f"[HITL_FOV] Slicer no disponible: {e}")
    except Exception as e:
        slicer_error = str(e)
        logger.warning(f"[HITL_FOV] Error al cargar en Slicer: {e}")

    if not slicer_loaded:
        hitl_msg += f"\n\n⚠️ {slicer_error}\nUsa el script de abajo manualmente."

    decision = interrupt({
        "type":              "hitl_fov",
        "message":           hitl_msg,
        "pid":               state.pid,
        "slicer_loaded":     slicer_loaded,
        "slicer_fov_script": combined,
    })

    hitl_record = HITLDecision(
        point="fov",
        decision=decision.get("decision", "approved"),
        notes=decision.get("notes") or "",
    )
    approved = hitl_record.decision == "approved"
    logger.info(f"[HITL_FOV] Decision: {hitl_record.decision}")

    return {
        "current_node":      "crop_fine" if approved else "detect",
        "pending_hitl":      False,
        "slicer_fov_script": combined,
        "hitl_message":      hitl_msg,
        "hitl_decisions":    state.hitl_decisions + [hitl_record],
    }


# ── HITL: CROPS ──────────────────────────────────────────────────────────────

async def node_hitl_crops(state: PipelineState) -> dict:
    """HITL #2 — Verificar crops 160³ @ 0.565mm superpuestos en Slicer."""
    logger.info(f"[HITL_CROPS] PID={state.pid}")

    # Intentar cargar crops en Slicer automáticamente
    slicer_loaded = False
    slicer_error  = ""
    crop_script   = state.slicer_fov_script or ""
    if crop_script:
        try:
            await _call_slicer_tool("execute_python_code", {
                "code": "slicer.mrmlScene.Clear(0)"
            })
            await _call_slicer_tool("execute_python_code", {
                "code": crop_script.replace('"', '\"')
            })
            slicer_loaded = True
            logger.info(f"[HITL_CROPS] Crops cargados en Slicer automáticamente")
        except RuntimeError as e:
            slicer_error = str(e)
            logger.warning(f"[HITL_CROPS] Slicer no disponible: {e}")
        except Exception as e:
            slicer_error = str(e)
            logger.warning(f"[HITL_CROPS] Error: {e}")

    crops_msg = state.hitl_message or ""
    if not slicer_loaded and slicer_error:
        crops_msg += f"\n\n⚠️ {slicer_error}\nUsa el script manualmente."

    decision = interrupt({
        "type":          "hitl_crops",
        "message":       crops_msg,
        "pid":           state.pid,
        "slicer_loaded": slicer_loaded,
        "slicer_script": crop_script,
        "instructions":  "Verifica MRI+TRUS crops superpuestos en Slicer, aprueba.",
    })

    hitl_record = HITLDecision(
        point="crops",
        decision=decision.get("decision", "approved"),
        notes=decision.get("notes") or "",
    )
    approved = hitl_record.decision == "approved"
    logger.info(f"[HITL_CROPS] Decision: {hitl_record.decision}")

    return {
        "current_node":      "end" if approved else "crop_fine",
        "pipeline_complete": approved,
        "pending_hitl":      False,
        "hitl_decisions":    state.hitl_decisions + [hitl_record],
        "messages":          [AIMessage(content=(
            f"Crops 160³ @ 0.565mm aprobados para PID {state.pid}. "
            "Pipeline completado hasta crop. Registro pendiente — Fase 4."
        ))],
    }


# ── HITL: PARAMS ─────────────────────────────────────────────────────────────

async def node_hitl_params(state: PipelineState) -> dict:
    """HITL #3 — Aprobar parámetros de registro sugeridos por MedGemma."""
    logger.info(f"[HITL_PARAMS] PID={state.pid}")

    decision = interrupt({
        "type":    "hitl_params",
        "message": state.hitl_message,
        "pid":     state.pid,
        "suggested_params": state.registration_params.model_dump() if state.registration_params else {},
    })

    hitl_record = HITLDecision(
        point="params",
        decision=decision.get("decision", "approved"),
        notes=decision.get("notes") or "",
    )
    approved = hitl_record.decision in ("approved", "adjusted")

    from .state import RegistrationParams
    params = state.registration_params
    if hitl_record.decision == "adjusted" and decision.get("params_override"):
        try:
            params = RegistrationParams(**decision["params_override"])
        except Exception as e:
            logger.warning(f"[HITL_PARAMS] Override error: {e}")

    logger.info(f"[HITL_PARAMS] Decision: {hitl_record.decision}")
    return {
        "current_node":        "register" if approved else "plan_params",
        "pending_hitl":        False,
        "hitl_decisions":      state.hitl_decisions + [hitl_record],
        "registration_params": params,
    }


# ── HITL: FINAL ──────────────────────────────────────────────────────────────

async def node_hitl_final(state: PipelineState) -> dict:
    """HITL #4 — Validación final antes de guardar."""
    logger.info(f"[HITL_FINAL] PID={state.pid}")
    decision = interrupt({
        "type": "hitl_final", "message": state.hitl_message, "pid": state.pid,
    })
    hitl_record = HITLDecision(
        point="final",
        decision=decision.get("decision", "approved"),
        notes=decision.get("notes") or "",
    )
    approved = hitl_record.decision == "approved"
    return {
        "current_node": "report" if approved else "end",
        "pending_hitl": False,
        "hitl_decisions": state.hitl_decisions + [hitl_record],
        "pipeline_complete": not approved,
    }


# ── HITL: FAILURE ─────────────────────────────────────────────────────────────

async def node_hitl_failure_decision(state: PipelineState) -> dict:
    """HITL #3b — Decisión cuando el registro falla."""
    logger.info(f"[HITL_FAILURE] PID={state.pid}")
    decision = interrupt({
        "type": "hitl_failure", "message": state.hitl_message, "pid": state.pid,
    })
    hitl_record = HITLDecision(
        point="failure",
        decision=decision.get("decision", "rejected"),
        notes=decision.get("notes") or "",
    )
    dec = hitl_record.decision
    next_node = {"approved": "validate", "rejected": "end", "retry": "plan_params"}.get(dec, "end")
    return {
        "current_node": next_node,
        "pending_hitl": False,
        "hitl_decisions": state.hitl_decisions + [hitl_record],
        "retry_count": 0 if dec == "retry" else state.retry_count,
        "pipeline_complete": dec == "rejected",
    }


# ── Routing functions ─────────────────────────────────────────────────────────

def _route(state, node_name):
    return state.current_node if state.current_node in (node_name, "detect", END, "end") else node_name

def route_after_load(s):
    return END if s.error else "preprocess"

def route_after_preprocess(s):
    return END if s.error else "detect"

def route_after_detect(s):
    return END if s.error else "hitl_fov"

def route_after_hitl_fov(s):
    last = s.hitl_decisions[-1] if s.hitl_decisions else None
    if last and last.point == "fov" and last.decision == "rejected":
        return "detect"
    return "crop_fine"

def route_after_hitl_crops(s):
    last = s.hitl_decisions[-1] if s.hitl_decisions else None
    if last and last.point == "crops" and last.decision == "approved":
        return END
    return "crop_fine"

def route_after_hitl_params(s):
    last = s.hitl_decisions[-1] if s.hitl_decisions else None
    if last and last.point == "params" and last.decision == "rejected":
        return "plan_params"
    return "register"

def route_after_register(s):
    if s.registration_success:
        return "validate"
    if s.retry_count < s.max_retries:
        return "plan_params"
    return "hitl_failure"

def route_after_hitl_failure(s):
    last = s.hitl_decisions[-1] if s.hitl_decisions else None
    if not last:
        return END
    return {"approved": "validate", "rejected": END, "retry": "plan_params"}.get(last.decision, END)

def route_after_hitl_final(s):
    last = s.hitl_decisions[-1] if s.hitl_decisions else None
    return "report" if (last and last.decision == "approved") else END


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph(db_path: Optional[str] = None) -> Any:
    if db_path is None:
        db_path = os.getenv("LANGGRAPH_DB_PATH", "./data/langgraph_checkpoints.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    builder = StateGraph(PipelineState)

    # Nodes
    builder.add_node("load",          node_load)
    builder.add_node("preprocess",    node_preprocess)
    builder.add_node("detect",        node_detect)
    builder.add_node("hitl_fov",      node_hitl_fov)
    builder.add_node("crop_fine",     node_crop_fine)
    builder.add_node("hitl_crops",    node_hitl_crops)
    builder.add_node("plan_params",   node_plan_params)
    builder.add_node("hitl_params",   node_hitl_params)
    builder.add_node("register",      node_register)
    builder.add_node("validate",      node_validate)
    builder.add_node("hitl_final",    node_hitl_final)
    builder.add_node("hitl_failure",  node_hitl_failure_decision)
    builder.add_node("report",        node_report)

    # Fixed edges
    builder.add_edge(START,        "load")
    builder.add_edge("crop_fine",  "hitl_crops")   # ← crop_fine → hitl_crops (not plan_params)
    builder.add_edge("plan_params","hitl_params")
    builder.add_edge("validate",   "hitl_final")
    builder.add_edge("report",     END)

    # Conditional edges
    builder.add_conditional_edges("load",       route_after_load)
    builder.add_conditional_edges("preprocess",   route_after_preprocess)
    builder.add_conditional_edges("detect",       route_after_detect)
    builder.add_conditional_edges("hitl_fov",     route_after_hitl_fov)
    builder.add_conditional_edges("hitl_crops",   route_after_hitl_crops)
    builder.add_conditional_edges("hitl_params",  route_after_hitl_params)
    builder.add_conditional_edges("register",     route_after_register)
    builder.add_conditional_edges("hitl_failure", route_after_hitl_failure)
    builder.add_conditional_edges("hitl_final",   route_after_hitl_final)

    # Checkpointer
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        memory = SqliteSaver.from_conn_string(db_path)
        graph  = builder.compile(checkpointer=memory, interrupt_before=[])
        logger.info(f"Grafo compilado con SQLite | DB: {db_path}")
    except Exception as e:
        logger.warning(f"SQLite no disponible ({e}), usando MemorySaver")
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
        graph  = builder.compile(checkpointer=memory, interrupt_before=[])
        logger.info("Grafo compilado con MemorySaver")

    return graph


# ── Resume helper ─────────────────────────────────────────────────────────────

async def resume_with_hitl_decision(
    graph: Any,
    thread_id: str,
    decision: str,
    hitl_type: str,
    notes: str = "",
    adjusted_value: Any = None,
) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    resume_data = {"decision": decision, "hitl_type": hitl_type, "notes": notes}
    if adjusted_value:
        if hitl_type == "fov":
            resume_data["adjusted_center"] = adjusted_value
        elif hitl_type == "params":
            resume_data["params_override"] = adjusted_value

    logger.info(f"Resumiendo thread={thread_id} | HITL={hitl_type} | decision={decision}")
    result = await graph.ainvoke(Command(resume=resume_data), config)
    return result