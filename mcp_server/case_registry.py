"""
case_registry.py — Registro persistente de casos en JSON.

Actúa como el MCP Resource 'case_registry': guarda qué casos
han sido cargados, sus series, y el estado del pipeline.

Complejidad:
  - get/set por case_id : O(1) — dict lookup
  - list_cases          : O(n) — n casos registrados
  - save a disco        : O(n) — serialización JSON
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class CaseRegistry:
    """
    Registro en memoria con persistencia JSON.

    Cada entrada tiene la estructura:
    {
        "case_id": str,
        "created_at": ISO8601,
        "dicom_path": str,
        "series": { series_uid: {modality, quality, ...} },
        "pipeline_state": str,   # "loaded" | "detected" | "registered" | "validated"
        "notes": str
    }
    """

    def __init__(self, registry_path: str = "./data/case_registry.json"):
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    # ── Persistencia ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Carga el registro desde disco. O(n)."""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info(f"Registry cargado: {len(self._data)} casos desde {self._path}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"No se pudo cargar registry ({e}). Iniciando vacío.")
                self._data = {}
        else:
            logger.info("Registry nuevo iniciado.")

    def _save(self) -> None:
        """Persiste el registro a disco. O(n)."""
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.error(f"Error guardando registry: {e}")

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def register_case(
        self,
        dicom_path: str,
        case_id: Optional[str] = None,
    ) -> str:
        """
        Registra un caso nuevo. Retorna el case_id asignado. O(1).
        Si ya existe un caso con el mismo dicom_path, retorna el existente.
        """
        # Idempotencia: evitar duplicados por path
        for cid, data in self._data.items():
            if data.get("dicom_path") == dicom_path:
                logger.info(f"Caso ya registrado para {dicom_path}: {cid}")
                return cid

        cid = case_id or f"case_{uuid.uuid4().hex[:8]}"
        self._data[cid] = {
            "case_id": cid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dicom_path": dicom_path,
            "series": {},
            "pipeline_state": "loaded",
            "notes": "",
        }
        self._save()
        logger.info(f"Caso registrado: {cid}")
        return cid

    def get_case(self, case_id: str) -> Optional[dict[str, Any]]:
        """Retorna un caso por ID. O(1)."""
        return self._data.get(case_id)

    def update_series(
        self,
        case_id: str,
        series_uid: str,
        series_info: dict[str, Any],
    ) -> None:
        """Agrega o actualiza una serie dentro de un caso. O(1)."""
        if case_id not in self._data:
            raise KeyError(f"Caso '{case_id}' no encontrado en el registry.")
        self._data[case_id]["series"][series_uid] = series_info
        self._save()

    def update_pipeline_state(self, case_id: str, state: str) -> None:
        """Actualiza el estado del pipeline para un caso. O(1)."""
        if case_id not in self._data:
            raise KeyError(f"Caso '{case_id}' no encontrado.")
        self._data[case_id]["pipeline_state"] = state
        self._data[case_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def list_cases(self) -> list[dict[str, Any]]:
        """Lista todos los casos registrados. O(n)."""
        return list(self._data.values())

    def case_exists(self, case_id: str) -> bool:
        """Verifica si un caso existe. O(1)."""
        return case_id in self._data

    def to_resource_dict(self) -> dict[str, Any]:
        """
        Serialización para el MCP Resource 'case_registry'.
        Retorna un resumen sin datos sensibles.
        """
        return {
            "total_cases": len(self._data),
            "cases": [
                {
                    "case_id": v["case_id"],
                    "pipeline_state": v["pipeline_state"],
                    "series_count": len(v.get("series", {})),
                    "created_at": v["created_at"],
                }
                for v in self._data.values()
            ],
        }
