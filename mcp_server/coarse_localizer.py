"""
coarse_localizer.py — CoarseCNN para localización de próstata en volúmenes crudos.

Reproduce exactamente el pipeline de inferencia del stage1 del doctorado:
    vol_iso → pad_to_multiple(32) → grid patches 32³ → CoarseCNN → probs → centroide

Arquitectura:
    ConvBlock3D ×4 → GlobalAvgPool → FC(128→64) → Dropout → FC(64→1) → Sigmoid
    Input : (B, 1, 32, 32, 32) float32
    Output: (B, 1) probabilidad ∈ [0,1]

Pipeline de inferencia:
    1. Preprocesar volumen crudo: reset_geometry → resample 0.565mm iso
    2. Pad al múltiplo de 32
    3. Extraer grid de patches 32³ (sin overlap)
    4. Predecir probabilidad por patch
    5. Centroide ponderado → centro coarse en mm
    6. Crop 160³ centrado en el centroide

Complejidad:
    preprocesamiento : O(V_orig) — dominado por resampleo
    inferencia grid  : O(N_patches × 32³) — N = (Xp/32)×(Yp/32)×(Zp/32)
    centroide        : O(N_patches) — suma ponderada
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from loguru import logger


# ── Constantes ────────────────────────────────────────────────────────────────
PATCH_SIZE  = 32
DST_SPACING = 0.565
CROP_SIZE   = 160
HALF_FOV_MM = CROP_SIZE / 2 * DST_SPACING   # = 45.2mm


# ── Arquitectura CoarseCNN ────────────────────────────────────────────────────

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CoarseCNN(nn.Module):
    """
    CNN coarse para localización de próstata por patches 32³.
    Input : (B, 1, 32, 32, 32) float32
    Output: (B, 1) logit — Sigmoid para p ∈ [0,1]
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock3D(1,    16),   # → (B, 16, 16, 16, 16)
            ConvBlock3D(16,   32),   # → (B, 32,  8,  8,  8)
            ConvBlock3D(32,   64),   # → (B, 64,  4,  4,  4)
            ConvBlock3D(64,  128),   # → (B,128,  2,  2,  2)
        )
        self.gap  = nn.AdaptiveAvgPool3d(1)   # → (B,128,1,1,1)
        self.head = nn.Sequential(
            nn.Flatten(),                      # → (B,128)
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),                  # → (B,1) logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gap(self.encoder(x)))

    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def load_coarse_model(
    model_path: str,
    device: Optional[torch.device] = None,
) -> tuple[CoarseCNN, torch.device]:
    """Carga CoarseCNN desde .pth con state_dict. O(P) parámetros."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo coarse no encontrado: {model_path}")

    model = CoarseCNN().to(device)
    state = torch.load(str(path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"CoarseCNN cargado: {path.name} | {n_params:,} params | {device}")
    return model, device


# ── Preprocesamiento para inferencia (sin STL/VTK) ───────────────────────────

IDENTITY_DIR = [1, 0, 0, 0, 1, 0, 0, 0, 1]


def preprocess_raw_volume(
    input_path: str,
    is_dicom: bool = True,
) -> sitk.Image:
    """
    Preprocesa un volumen crudo para inferencia con CoarseCNN.

    Reproduce el pipeline de Preprocessing.py SIN STL/VTK
    (no hay GT disponible en inferencia — solo necesitamos vol_iso).

    Pasos:
        1. Cargar DICOM o NIfTI
        2. reset_geometry: DICOMOrient(LPS) + origen(0,0,0) + dir=identidad
        3. resample_isotropic: B-spline a 0.565mm preservando el FOV completo

    Complejidad: O(V_orig) — dominado por el resampleo.
    """
    # Paso 1: cargar
    if is_dicom:
        # MRI: leer serie DICOM completa
        reader = sitk.ImageSeriesReader()
        files  = reader.GetGDCMSeriesFileNames(input_path)
        if not files:
            raise ValueError(f"No se encontraron DICOMs en: {input_path}")
        reader.SetFileNames(files)
        img = reader.Execute()
        logger.debug(f"DICOM cargado: {len(files)} slices")
    else:
        # TRUS: NIfTI ya exportado desde Slicer — leer directamente
        # TRUS_exported_path ya tiene el spacing y orientación correctos
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Archivo no encontrado: {input_path}")
        img = sitk.ReadImage(input_path)
        logger.debug(f"NIfTI TRUS exportado cargado: {Path(input_path).name} | "
                     f"size={img.GetSize()} spacing={[round(s,3) for s in img.GetSpacing()]}")

    orig_size    = img.GetSize()
    orig_spacing = img.GetSpacing()
    logger.debug(f"Original: {orig_size} @ {orig_spacing} mm")

    # Paso 2: reset_geometry (idéntico a Preprocessing.py)
    img = sitk.DICOMOrient(img, "LPS")
    img.SetOrigin([0.0, 0.0, 0.0])
    img.SetDirection(IDENTITY_DIR)

    # Paso 3: resample isotrópico a 0.565mm
    orig_sp_arr   = np.array(img.GetSpacing())
    orig_size_arr = np.array(img.GetSize())
    new_size      = np.ceil(orig_size_arr * orig_sp_arr / DST_SPACING).astype(int)

    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing([DST_SPACING] * 3)
    r.SetSize(new_size.tolist())
    r.SetOutputOrigin([0.0, 0.0, 0.0])
    r.SetOutputDirection(IDENTITY_DIR)
    r.SetInterpolator(sitk.sitkBSpline)
    r.SetDefaultPixelValue(0)
    vol_iso = r.Execute(img)

    logger.info(
        f"vol_iso: {vol_iso.GetSize()} @ {vol_iso.GetSpacing()} mm | "
        f"FOV: {[round(s*DST_SPACING, 1) for s in vol_iso.GetSize()]} mm"
    )
    return vol_iso


# ── Inferencia grid de patches ────────────────────────────────────────────────

def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """
    Z-score por patch — replica exactamente stage1.py.
    Si std ≈ 0 (padding), retorna zeros.
    """
    mu  = patch.mean()
    std = patch.std()
    if std < 1e-8:
        return np.zeros_like(patch, dtype=np.float32)
    return ((patch - mu) / std).astype(np.float32)


def run_coarse_inference(
    vol_iso: sitk.Image,
    model: CoarseCNN,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """
    Ejecuta CoarseCNN sobre el grid completo de patches 32³.

    Replica exactamente stage1_test.py:
        1. Pad al múltiplo de 32
        2. Grid de patches sin overlap
        3. Batch inference con CoarseCNN
        4. Centroide ponderado en mm

    Retorna:
        centroid_mm    : (x,y,z) en mm espacio vol_iso (origen 0,0,0)
        prob_map       : array (nK,nJ,nI) float32 de probabilidades
        grid_shape     : (nI, nJ, nK) — número de patches por eje
        pad_shape      : (Xp, Yp, Zp) — tamaño del volumen paddeado
        max_prob       : probabilidad máxima en el grid

    Complejidad: O(N_patches × 32³) donde N = (Xp/32)×(Yp/32)×(Zp/32).
    """
    P   = PATCH_SIZE
    arr = sitk.GetArrayFromImage(vol_iso).astype(np.float32)  # (Z,Y,X)
    Z, Y, X = arr.shape

    # Pad al múltiplo de P
    Zp = int(np.ceil(Z / P) * P)
    Yp = int(np.ceil(Y / P) * P)
    Xp = int(np.ceil(X / P) * P)
    vol_pad = np.zeros((Zp, Yp, Xp), dtype=np.float32)
    vol_pad[:Z, :Y, :X] = arr

    nI = Xp // P   # eje X
    nJ = Yp // P   # eje Y
    nK = Zp // P   # eje Z

    total_patches = nI * nJ * nK
    logger.info(
        f"Grid: {nI}×{nJ}×{nK} = {total_patches} patches | "
        f"Vol padded: ({Xp},{Yp},{Zp}) | "
        f"FOV padded: {Xp*DST_SPACING:.1f}×{Yp*DST_SPACING:.1f}×{Zp*DST_SPACING:.1f} mm"
    )

    # Extraer todos los patches
    patches_list = []
    centers      = []   # centros en mm (x,y,z)

    for k in range(nK):
        for j in range(nJ):
            for i in range(nI):
                patch = vol_pad[
                    k*P:(k+1)*P,
                    j*P:(j+1)*P,
                    i*P:(i+1)*P
                ].copy()
                patches_list.append(normalize_patch(patch))
                centers.append([
                    (i + 0.5) * P * DST_SPACING,   # x mm
                    (j + 0.5) * P * DST_SPACING,   # y mm
                    (k + 0.5) * P * DST_SPACING,   # z mm
                ])

    centers = np.array(centers)   # (N,3)

    # Inferencia en batches
    all_probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, total_patches, batch_size):
            batch_np = np.array(patches_list[start:start+batch_size])   # (B,32,32,32)
            batch_t  = torch.tensor(
                batch_np[:, None], dtype=torch.float32
            ).to(device)
            logits = model(batch_t)
            probs  = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs.tolist())

    probs_arr = np.array(all_probs, dtype=np.float32)  # (N,)

    # Centroide ponderado (replica stage1_test.py exactamente)
    w_sum = probs_arr.sum()
    if w_sum < 1e-8:
        centroid_mm = centers.mean(axis=0)
        logger.warning("Todas las probabilidades ~0 — usando centroide geométrico")
    else:
        centroid_mm = (centers * probs_arr[:, None]).sum(axis=0) / w_sum

    # Reshape a grid 3D para visualización
    prob_map = probs_arr.reshape(nK, nJ, nI)   # (Z,Y,X) en patches

    logger.info(
        f"Centroide coarse: ({centroid_mm[0]:.1f},{centroid_mm[1]:.1f},{centroid_mm[2]:.1f}) mm | "
        f"max_prob={probs_arr.max():.3f}"
    )

    return {
        "centroid_mm":  centroid_mm.tolist(),      # (x,y,z) mm
        "prob_map":     prob_map,                  # (nK,nJ,nI) float32
        "grid_shape":   (nI, nJ, nK),
        "pad_shape":    (Xp, Yp, Zp),
        "orig_shape":   (X, Y, Z),
        "centers_mm":   centers.tolist(),
        "probs":        probs_arr.tolist(),
        "max_prob":     float(probs_arr.max()),
        "total_patches": total_patches,
    }


# ── Crop 160³ centrado en el centroide coarse ─────────────────────────────────

def crop_fine_volume(
    vol_iso: sitk.Image,
    centroid_mm: list[float],
    crop_size: int = CROP_SIZE,
    spacing: float = DST_SPACING,
) -> tuple[sitk.Image, list[float]]:
    """
    Crop 160³ centrado en el centroide coarse.

    Si el centroide está cerca del borde, el crop se ajusta con padding negro.
    Retorna (vol_crop, crop_origin_mm) para trazabilidad.

    Complejidad: O(160³) — tamaño fijo del crop de salida.
    """
    half_mm = crop_size / 2 * spacing   # = 45.2mm

    cx, cy, cz = centroid_mm
    crop_origin_mm = [
        cx - half_mm,
        cy - half_mm,
        cz - half_mm,
    ]

    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing([spacing] * 3)
    r.SetSize([crop_size] * 3)
    r.SetOutputOrigin(crop_origin_mm)
    r.SetOutputDirection(IDENTITY_DIR)
    r.SetInterpolator(sitk.sitkBSpline)
    r.SetDefaultPixelValue(0)   # padding negro si sale del FOV

    vol_crop = r.Execute(vol_iso)

    logger.info(
        f"Crop 160³ | origen: {[round(v,1) for v in crop_origin_mm]} mm | "
        f"centro: {[round(v,1) for v in centroid_mm]} mm"
    )
    return vol_crop, crop_origin_mm


def save_prob_heatmap_nifti(
    prob_map: np.ndarray,
    grid_shape: tuple,
    orig_vol_iso: sitk.Image,
    output_path: str,
) -> None:
    """
    Guarda el mapa de probabilidades como NIfTI para visualización en Slicer.

    Upsamplea el mapa de patches (nK,nJ,nI) al espacio original del vol_iso
    usando interpolación de vecino más próximo para preservar los valores
    discretos por patch.

    Complejidad: O(V_iso) — interpolación sobre el volumen isotrópico.
    """
    nK, nJ, nI = prob_map.shape
    orig_size = orig_vol_iso.GetSize()   # (X, Y, Z)

    # Crear imagen SimpleITK del prob_map en espacio de patches
    pm_img = sitk.GetImageFromArray(prob_map.astype(np.float32))
    pm_img.SetSpacing([PATCH_SIZE * DST_SPACING] * 3)   # 32 × 0.565 = 18.08mm
    pm_img.SetOrigin([PATCH_SIZE * DST_SPACING * 0.5] * 3)
    pm_img.SetDirection(IDENTITY_DIR)

    # Upsamplear al espacio de vol_iso
    # Crear imagen de referencia con origen forzado a 0,0,0
    ref = sitk.Image(orig_vol_iso.GetSize(), sitk.sitkFloat32)
    ref.SetSpacing(orig_vol_iso.GetSpacing())
    ref.SetOrigin([0.0, 0.0, 0.0])   # forzar mismo origen que preprocess_raw_volume
    ref.SetDirection(orig_vol_iso.GetDirection())

    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetDefaultPixelValue(0.0)
    prob_upsampled = r.Execute(pm_img)

    sitk.WriteImage(prob_upsampled, output_path, useCompression=True)
    logger.info(f"Heatmap guardado: {output_path}")