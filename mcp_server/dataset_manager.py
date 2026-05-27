"""
dataset_manager.py — Gestor del dataset de casos preprocesados.

Lee el índice JSON (dataset_index_160_localizer_641.json) y expone
los casos como una "base de datos" simulada para el pipeline.

El JSON ya contiene rutas absolutas, ground truth de centroides,
parámetros de normalización y flags de calidad — no hay que
recalcular nada para los casos preprocesados.

Complejidad:
    load_index     : O(n) — n = casos en el JSON
    get_case       : O(1) — dict lookup
    list_cases     : O(n)
    get_ready_tensor: O(V) — V = 160³ voxeles, lectura NIfTI
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch
from loguru import logger


# ── Tipos de datos ────────────────────────────────────────────────────────────

@dataclass
class CaseEntry:
    """Un caso del dataset con todos sus metadatos."""
    pid: str

    # Rutas
    mri_volume:  str = ""
    trus_volume: str = ""
    mri_mask:    str = ""
    trus_mask:   str = ""

    # Ground truth — centroide de la máscara (z,y,x) voxeles cubo 160³
    mri_centroid_vox_zyx:  list[float] = field(default_factory=list)
    trus_centroid_vox_zyx: list[float] = field(default_factory=list)

    # Ground truth — centroide en mm espacio local
    mri_centroid_mm_xyz:  list[float] = field(default_factory=list)
    trus_centroid_mm_xyz: list[float] = field(default_factory=list)

    # Ground truth — centroide en mm espacio isotrópico global
    mri_centroid_iso_mm:  list[float] = field(default_factory=list)
    trus_centroid_iso_mm: list[float] = field(default_factory=list)

    # Orígenes del crop
    mri_crop_origin_mm:  list[float] = field(default_factory=list)
    trus_crop_origin_mm: list[float] = field(default_factory=list)

    # FOV isotrópico en mm
    mri_iso_fov_mm:  list[float] = field(default_factory=list)
    trus_iso_fov_mm: list[float] = field(default_factory=list)

    # Calidad
    mri_usable:       bool  = True
    trus_usable:      bool  = True
    mri_min_margin:   float = 0.0
    trus_min_margin:  float = 0.0

    # Normalización original (p1/p99/mean/std)
    mri_norm:  dict = field(default_factory=dict)
    trus_norm: dict = field(default_factory=dict)

    status: str = "ok"

    def is_valid(self, modality: str = "MRI") -> bool:
        """Verifica que el caso tiene archivos y es usable."""
        if modality == "MRI":
            return (
                self.mri_usable and
                bool(self.mri_volume) and
                Path(self.mri_volume).exists()
            )
        return (
            self.trus_usable and
            bool(self.trus_volume) and
            Path(self.trus_volume).exists()
        )

    def volume_path(self, modality: str) -> str:
        return self.mri_volume if modality == "MRI" else self.trus_volume

    def mask_path(self, modality: str) -> str:
        return self.mri_mask if modality == "MRI" else self.trus_mask

    def gt_centroid_vox(self, modality: str) -> list[float]:
        return self.mri_centroid_vox_zyx if modality == "MRI" else self.trus_centroid_vox_zyx

    def gt_centroid_mm(self, modality: str) -> list[float]:
        return self.mri_centroid_mm_xyz if modality == "MRI" else self.trus_centroid_mm_xyz

    def crop_origin(self, modality: str) -> list[float]:
        return self.mri_crop_origin_mm if modality == "MRI" else self.trus_crop_origin_mm

    def summary(self) -> dict:
        """Resumen para mostrar en la UI."""
        return {
            "pid": self.pid,
            "mri_usable": self.mri_usable,
            "trus_usable": self.trus_usable,
            "mri_min_margin_vox": self.mri_min_margin,
            "trus_min_margin_vox": self.trus_min_margin,
            "mri_volume_exists": Path(self.mri_volume).exists() if self.mri_volume else False,
            "trus_volume_exists": Path(self.trus_volume).exists() if self.trus_volume else False,
            "status": self.status,
        }


# ── Dataset Manager ───────────────────────────────────────────────────────────

class DatasetManager:
    """
    Carga y gestiona el índice JSON del dataset preprocesado.

    Actúa como la "base de datos" simulada del pipeline:
    - Indexa todos los casos disponibles
    - Expone rutas y metadatos sin recargar NIfTIs
    - Provee tensores listos para el modelo bajo demanda
    - Permite comparar predicción vs ground truth

    El JSON ya tiene todo precalculado — esta clase solo lo organiza.
    """

    def __init__(self, index_path: str):
        self._path  = Path(index_path)
        self._cases: dict[str, CaseEntry] = {}
        self._load()

    def _load(self) -> None:
        """Carga el JSON índice. O(n) donde n = casos."""
        if not self._path.exists():
            raise FileNotFoundError(f"Índice no encontrado: {self._path}")

        with open(self._path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for pid, data in raw.items():
            if data.get("status") != "ok":
                logger.debug(f"Caso {pid} omitido (status={data.get('status')})")
                continue

            gt_mri  = data.get("MRI_gt_check", {})
            gt_trus = data.get("TRUS_gt_check", {})

            entry = CaseEntry(
                pid=pid,
                mri_volume  = data.get("MRI_volume_loc", ""),
                trus_volume = data.get("TRUS_volume_loc", ""),
                mri_mask    = data.get("MRI_mask_loc", ""),
                trus_mask   = data.get("TRUS_mask_loc", ""),
                mri_centroid_vox_zyx  = data.get("MRI_centroid_local_vox_zyx", []),
                trus_centroid_vox_zyx = data.get("TRUS_centroid_local_vox_zyx", []),
                mri_centroid_mm_xyz   = data.get("MRI_centroid_local_mm_xyz", []),
                trus_centroid_mm_xyz  = data.get("TRUS_centroid_local_mm_xyz", []),
                mri_centroid_iso_mm   = data.get("MRI_centroid_iso_mm", []),
                trus_centroid_iso_mm  = data.get("TRUS_centroid_iso_mm", []),
                mri_crop_origin_mm    = data.get("MRI_crop_origin_mm", []),
                trus_crop_origin_mm   = data.get("TRUS_crop_origin_mm", []),
                mri_iso_fov_mm        = data.get("MRI_iso_fov_mm", []),
                trus_iso_fov_mm       = data.get("TRUS_iso_fov_mm", []),
                mri_usable    = data.get("MRI_usable", False),
                trus_usable   = data.get("TRUS_usable", False),
                mri_min_margin  = gt_mri.get("min_margin_vox", 0.0),
                trus_min_margin = gt_trus.get("min_margin_vox", 0.0),
                mri_norm  = data.get("MRI_normalization", {}),
                trus_norm = data.get("TRUS_normalization", {}),
                status = data.get("status", "ok"),
            )
            self._cases[pid] = entry

        logger.info(
            f"Dataset cargado: {len(self._cases)} casos desde {self._path.name}"
        )

    # ── Consultas ─────────────────────────────────────────────────────────────

    def get_case(self, pid: str) -> Optional[CaseEntry]:
        """O(1)."""
        return self._cases.get(pid)

    def list_cases(
        self,
        modality: str = "MRI",
        only_usable: bool = True,
    ) -> list[CaseEntry]:
        """
        Lista casos filtrados por modalidad y usabilidad. O(n).
        Ordenados por pid para reproducibilidad.
        """
        cases = sorted(self._cases.values(), key=lambda c: c.pid)
        if only_usable:
            cases = [c for c in cases if c.is_valid(modality)]
        return cases

    def list_pids(self, modality: str = "MRI", only_usable: bool = True) -> list[str]:
        """Lista de PIDs disponibles. O(n)."""
        return [c.pid for c in self.list_cases(modality, only_usable)]

    def summary_table(self) -> list[dict]:
        """Resumen de todos los casos para mostrar en la UI. O(n)."""
        return [c.summary() for c in sorted(self._cases.values(), key=lambda c: c.pid)]

    def to_resource_dict(self) -> dict:
        """Serialización para MCP Resource. O(n)."""
        usable_mri  = sum(1 for c in self._cases.values() if c.mri_usable)
        usable_trus = sum(1 for c in self._cases.values() if c.trus_usable)
        return {
            "total_cases": len(self._cases),
            "usable_mri":  usable_mri,
            "usable_trus": usable_trus,
            "index_path":  str(self._path),
            "pids": self.list_pids("MRI"),
        }

    # ── Carga de tensores ─────────────────────────────────────────────────────

    def get_ready_tensor(
        self,
        pid: str,
        modality: str = "MRI",
    ) -> tuple[torch.Tensor, CaseEntry]:
        """
        Carga el NIfTI preprocesado y lo convierte a tensor listo para el modelo.

        El volumen YA está normalizado (z-score aplicado en el Script 1).
        Solo hacemos: NIfTI → numpy → tensor (1,1,160,160,160).

        NO se aplica preprocesamiento adicional — el archivo es el output
        del Script 1, con z-score ya aplicado.

        Complejidad: O(V) = O(160³) — lectura del NIfTI.
        """
        case = self.get_case(pid)
        if case is None:
            raise KeyError(f"Caso '{pid}' no encontrado en el dataset.")

        vol_path = case.volume_path(modality)
        if not vol_path:
            raise ValueError(f"Caso {pid} no tiene volumen {modality}.")
        if not Path(vol_path).exists():
            raise FileNotFoundError(f"Archivo no encontrado: {vol_path}")

        # Leer NIfTI — ya normalizado, solo convertir
        img = sitk.ReadImage(vol_path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (z, y, x)

        if arr.shape != (160, 160, 160):
            raise ValueError(
                f"Shape inesperado {arr.shape} para {pid}/{modality}. "
                f"Se esperaba (160, 160, 160)."
            )

        tensor = torch.tensor(arr[np.newaxis, np.newaxis].copy())
        logger.debug(f"Tensor cargado: {pid}/{modality} | shape={tensor.shape}")
        return tensor, case

    # ── Métricas de evaluación ────────────────────────────────────────────────

    def compute_error_mm(
        self,
        pid: str,
        modality: str,
        pred_vox_zyx: list[float],
    ) -> dict:
        """
        Calcula el error entre la predicción del modelo y el ground truth.

        Error en voxeles y en mm (spacing = 0.565 mm).
        Complejidad: O(1).
        """
        case = self.get_case(pid)
        if case is None:
            return {"error": f"Caso {pid} no encontrado"}

        gt_zyx  = np.array(case.gt_centroid_vox(modality))
        pr_zyx  = np.array(pred_vox_zyx)
        SPACING = 0.565

        diff_vox = pr_zyx - gt_zyx
        error_vox = float(np.linalg.norm(diff_vox))
        error_mm  = error_vox * SPACING

        return {
            "pid": pid,
            "modality": modality,
            "gt_vox_zyx":   gt_zyx.tolist(),
            "pred_vox_zyx": pr_zyx.tolist(),
            "diff_vox_zyx": diff_vox.tolist(),
            "error_vox":    round(error_vox, 3),
            "error_mm":     round(error_mm, 3),
            "clinical_threshold_mm": 10.0,
            "within_threshold": error_mm <= 10.0,
        }
