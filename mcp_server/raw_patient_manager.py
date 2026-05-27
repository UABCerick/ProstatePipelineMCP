"""
raw_patient_manager.py — Gestor de pacientes crudos desde stage1_selected_pairs.json.

Provee acceso a las rutas raw (DICOM MRI + NIfTI TRUS exportado)
para cualquier paciente del dataset, listo para pasar por el pipeline
de preprocesamiento + CoarseCNN.

Complejidad:
    _load   : O(n) primera vez — carga el JSON completo
    get_case: O(1) — lookup en dict
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


STAGE1_JSON_DEFAULT = r"C:\Codes\Us_MRI_Fusion\PRE-TCIA\Preprocessing\pipeline_v3\stage1_selected_pairs.json"


@dataclass
class RawPatientEntry:
    """Rutas y metadatos de un paciente crudo."""
    pid: str

    # Rutas de entrada
    mri_dicom_path:    str = ""   # carpeta DICOM MRI
    trus_nifti_path:   str = ""   # NIfTI TRUS exportado
    mri_mask_path:     str = ""   # máscara MRI (para registro)
    trus_mask_path:    str = ""   # máscara TRUS (para registro)
    mri_stl_path:      str = ""   # STL MRI (opcional)
    trus_stl_path:     str = ""   # STL TRUS (opcional)

    # Metadatos MRI
    mri_technique:     str = ""
    mri_manufacturer:  str = ""
    mri_n_files:       int = 0
    mri_tr:            float = 0.0
    mri_te:            float = 0.0
    mri_score:         int = 0
    mri_series_desc:   str = ""
    mri_study_date:    str = ""

    # Metadatos TRUS
    trus_size_mb:      float = 0.0
    trus_study_date:   str = ""

    # Flags de calidad
    stage1_ok:         bool = False
    stage1_flags:      list = field(default_factory=list)
    in_old_641:        bool = False

    # Vol_iso preprocesados (si ya existen)
    mri_vol_iso_path:  str = ""
    trus_vol_iso_path: str = ""

    def has_raw(self) -> bool:
        """Verifica que existen los archivos crudos."""
        return (
            bool(self.mri_dicom_path) and Path(self.mri_dicom_path).exists() and
            bool(self.trus_nifti_path) and Path(self.trus_nifti_path).exists()
        )

    def has_masks(self) -> bool:
        """Verifica que existen las máscaras para registro."""
        return (
            bool(self.mri_mask_path) and Path(self.mri_mask_path).exists() and
            bool(self.trus_mask_path) and Path(self.trus_mask_path).exists()
        )

    def has_vol_iso(self) -> bool:
        """Verifica que ya existen los vol_iso preprocesados."""
        return (
            bool(self.mri_vol_iso_path) and Path(self.mri_vol_iso_path).exists() and
            bool(self.trus_vol_iso_path) and Path(self.trus_vol_iso_path).exists()
        )


class RawPatientManager:
    """
    Gestor de pacientes desde stage1_selected_pairs.json.

    Carga el JSON en memoria y provee acceso O(1) por PID.
    Opcionalmente verifica si ya existen los vol_iso preprocesados
    en un directorio de cache.
    """

    def __init__(
        self,
        json_path: Optional[str] = None,
        preproc_dir: Optional[str] = None,
    ):
        self._json_path   = json_path or os.getenv("STAGE1_JSON_PATH", STAGE1_JSON_DEFAULT)
        self._preproc_dir = preproc_dir or os.getenv("PREPROC_DIR", "")
        self._data: dict[str, RawPatientEntry] = {}
        self._loaded = False

    def _load(self) -> None:
        """Carga el JSON. O(n) primera vez."""
        if not Path(self._json_path).exists():
            logger.warning(f"stage1 JSON no encontrado: {self._json_path}")
            return

        with open(self._json_path) as f:
            raw = json.load(f)

        patients = raw.get("patients", {})
        for pid, entry in patients.items():
            # Calcular rutas de vol_iso si existe el directorio de cache
            mri_iso = ""
            trus_iso = ""
            if self._preproc_dir:
                mri_iso  = os.path.join(self._preproc_dir, pid, "MRI_vol_iso.nii.gz")
                trus_iso = os.path.join(self._preproc_dir, pid, "TRUS_vol_iso.nii.gz")

            self._data[pid] = RawPatientEntry(
                pid=pid,
                mri_dicom_path   = entry.get("MRI_dicom_path", ""),
                trus_nifti_path  = entry.get("TRUS_exported_path", ""),
                mri_mask_path    = entry.get("MRI_mask_path", ""),
                trus_mask_path   = entry.get("TRUS_mask_path", ""),
                mri_stl_path     = entry.get("MRI_STL_path", ""),
                trus_stl_path    = entry.get("TRUS_STL_path", ""),
                mri_technique    = entry.get("MRI_technique", ""),
                mri_manufacturer = entry.get("MRI_manufacturer", ""),
                mri_n_files      = entry.get("MRI_n_files", 0),
                mri_tr           = entry.get("MRI_TR", 0.0),
                mri_te           = entry.get("MRI_TE", 0.0),
                mri_score        = entry.get("MRI_score", 0),
                mri_series_desc  = entry.get("MRI_series_desc", ""),
                mri_study_date   = entry.get("MRI_study_date", ""),
                trus_size_mb     = entry.get("TRUS_size_mb", 0.0),
                trus_study_date  = entry.get("TRUS_study_date", ""),
                stage1_ok        = entry.get("stage1_ok", False),
                stage1_flags     = entry.get("stage1_flags", []),
                in_old_641       = entry.get("in_old_641", False),
                mri_vol_iso_path = mri_iso,
                trus_vol_iso_path= trus_iso,
            )

        self._loaded = True
        logger.info(f"RawPatientManager: {len(self._data)} pacientes desde {Path(self._json_path).name}")

    def get_case(self, pid: str) -> Optional[RawPatientEntry]:
        """Retorna el entry del paciente. O(1)."""
        if not self._loaded:
            self._load()
        return self._data.get(pid)

    def list_cases(self, only_ok: bool = True) -> list[str]:
        """Lista PIDs disponibles. O(n)."""
        if not self._loaded:
            self._load()
        if only_ok:
            return [p for p, e in self._data.items() if e.stage1_ok]
        return list(self._data.keys())

    def case_exists(self, pid: str) -> bool:
        if not self._loaded:
            self._load()
        return pid in self._data

    @property
    def total(self) -> int:
        if not self._loaded:
            self._load()
        return len(self._data)