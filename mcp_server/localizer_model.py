"""
localizer_model.py — Arquitectura LocalizerUNet y pipeline de preprocesamiento.

Réplica exacta del código de investigación del doctorado.
Este archivo NO modifica la arquitectura ni el preprocesamiento —
los encapsula tal como están para integrarlos al servidor MCP.

Arquitectura:
    LocalizerUNet — Encoder-decoder 3D con CenterNet head.
    Input : (B, 1, 160, 160, 160) float32
    Output: (B, 1, 160, 160, 160) float32 heatmap ∈ [0,1]

Pipeline de preprocesamiento (reproduce Script 1):
    DICOM/NIfTI → resampleo 0.565mm → LPS → crop 160³ → z-score → tensor

Complejidad:
    preprocess_volume : O(V) donde V = voxeles del volumen original
    run_inference     : O(V) forward pass — dominado por las conv3D
    extract_centroid  : O(V) sobre el heatmap 160³ — V = 4,096,000
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger


# ── Arquitectura ──────────────────────────────────────────────────────────────

class ConvBnLReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LocalizerUNet(nn.Module):
    """
    Encoder-decoder 3D con CenterNet head.
    Input : (B, 1, 160, 160, 160) float32
    Output: (B, 1, 160, 160, 160) float32, valores en [0,1]
    """

    def __init__(self):
        super().__init__()
        self.enc0       = ConvBnLReLU(1,    16, stride=2)
        self.enc1       = ConvBnLReLU(16,   32, stride=2)
        self.enc2       = ConvBnLReLU(32,   64, stride=2)
        self.enc3       = ConvBnLReLU(64,  128, stride=2)
        self.bottleneck = ConvBnLReLU(128, 128, stride=1)
        self.dec3       = ConvBnLReLU(128 + 64, 64)
        self.dec2       = ConvBnLReLU(64  + 32, 32)
        self.dec1       = ConvBnLReLU(32  + 16, 16)
        self.dec0       = ConvBnLReLU(16,        16)
        self.head       = nn.Sequential(
            nn.Conv3d(16, 8, 3, padding=1, bias=False),
            nn.InstanceNorm3d(8, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(8, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bottleneck(e3)
        d3 = self.dec3(torch.cat([
            F.interpolate(b,  scale_factor=2, mode="trilinear", align_corners=False), e2], 1))
        d2 = self.dec2(torch.cat([
            F.interpolate(d3, scale_factor=2, mode="trilinear", align_corners=False), e1], 1))
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, scale_factor=2, mode="trilinear", align_corners=False), e0], 1))
        d0 = self.dec0(F.interpolate(d1, scale_factor=2, mode="trilinear", align_corners=False))
        return self.head(d0)


# ── Carga de pesos ────────────────────────────────────────────────────────────

def load_localizer(
    model_path: str,
    device: Optional[torch.device] = None,
) -> tuple[LocalizerUNet, torch.device]:
    """
    Carga LocalizerUNet desde un .pth con state_dict.
    Selecciona CUDA automáticamente si está disponible. O(P) donde P = parámetros.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    model = LocalizerUNet().to(device)
    state = torch.load(str(path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Modelo cargado: {path.name} | {n_params:,} parámetros | device={device}")
    return model, device


# ── Preprocesamiento (reproduce Script 1 exactamente) ─────────────────────────

SPACING_ISO  = 0.565          # mm — spacing isotrópico objetivo
CROP_SIZE    = 160            # voxeles — tamaño del cubo
HALF_FOV_MM  = CROP_SIZE / 2 * SPACING_ISO   # = 45.2 mm


def _resample_isotropic(img: sitk.Image) -> sitk.Image:
    """
    Paso 1: Resampleo isotrópico a 0.565 mm preservando el FOV completo.
    Interpolador B-spline para el volumen. O(V_out).
    """
    orig_spacing = np.array(img.GetSpacing())          # (sx, sy, sz)
    orig_size    = np.array(img.GetSize())             # (nx, ny, nz)

    new_spacing  = np.array([SPACING_ISO] * 3)
    new_size     = np.ceil(orig_size * orig_spacing / new_spacing).astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize(new_size.tolist())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkBSpline)

    return resampler.Execute(img)


def _normalize_geometry(img: sitk.Image) -> sitk.Image:
    """
    Paso 2: Orientación LPS, origen (0,0,0), dirección identidad.
    Reproduce exactamente el Script 1. O(1).
    """
    img = sitk.DICOMOrient(img, "LPS")
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1,0,0, 0,1,0, 0,0,1))
    return img


def _crop_from_fov_center(
    img: sitk.Image,
) -> tuple[sitk.Image, np.ndarray]:
    """
    Paso 3: Crop 160³ centrado en el centro geométrico del FOV.
    No usa máscara — solo geometría del volumen.
    Retorna (imagen_crop, crop_origin_mm) para trazabilidad.

    crop_origin_mm: coordenada mm del vóxel (0,0,0) del cubo 160³
    en el espacio del volumen isotrópico.
    O(V_crop) = O(160³) = O(4M).
    """
    size_iso    = np.array(img.GetSize())              # (nx, ny, nz) voxeles
    spacing_arr = np.array(img.GetSpacing())           # (sx, sy, sz)

    # Centro geométrico del FOV en mm
    fov_center_mm = size_iso * spacing_arr / 2.0      # (cx, cy, cz) mm

    # Origen del crop en mm
    crop_origin_mm = fov_center_mm - HALF_FOV_MM      # (ox, oy, oz)

    # Origen en voxeles (puede ser negativo — se rellena con 0)
    crop_origin_vox = np.round(crop_origin_mm / spacing_arr).astype(int)

    # Resamplear al cubo 160³ con el origen calculado
    new_origin = img.GetOrigin() + crop_origin_mm * np.array([1, 1, 1])

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([SPACING_ISO] * 3)
    resampler.SetSize([CROP_SIZE] * 3)
    resampler.SetOutputDirection((1,0,0, 0,1,0, 0,0,1))
    resampler.SetOutputOrigin(crop_origin_mm.tolist())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkBSpline)

    crop = resampler.Execute(img)
    return crop, crop_origin_mm


def _normalize_intensity(arr: np.ndarray) -> np.ndarray:
    """
    Paso 4: Clip p1-p99 + z-score por volumen.
    O(V) = O(160³).
    """
    p1  = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)
    arr = np.clip(arr, p1, p99)
    mean = arr.mean()
    std  = arr.std()
    arr  = (arr - mean) / (std + 1e-5)
    return arr.astype(np.float32)


def preprocess_volume(
    image_path: str,
    is_dicom: bool = True,
) -> tuple[torch.Tensor, np.ndarray, sitk.Image]:
    """
    Pipeline completo de preprocesamiento (reproduce Script 1).

    Retorna:
        tensor       : (1, 1, 160, 160, 160) float32 — listo para el modelo
        crop_origin  : (3,) float32 array en mm — para trazabilidad
        img_iso      : SimpleITK Image isotrópica — para visualización en Slicer

    Complejidad: O(V_orig) dominado por el resampleo isotrópico.
    """
    # Cargar imagen
    if is_dicom:
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(image_path)
        if not dicom_files:
            raise ValueError(f"No se encontraron archivos DICOM en: {image_path}")
        reader.SetFileNames(dicom_files)
        img = reader.Execute()
        logger.debug(f"DICOM cargado: {len(dicom_files)} slices")
    else:
        img = sitk.ReadImage(image_path)
        logger.debug(f"NIfTI cargado: {image_path}")

    orig_size    = img.GetSize()
    orig_spacing = img.GetSpacing()
    logger.debug(f"Original: size={orig_size} spacing={orig_spacing}")

    # Paso 1: Resampleo isotrópico
    img_iso = _resample_isotropic(img)
    logger.debug(f"Tras resampleo: size={img_iso.GetSize()}")

    # Paso 2: Normalización de geometría
    img_iso = _normalize_geometry(img_iso)

    # Paso 3: Crop 160³ desde centro geométrico
    img_crop, crop_origin_mm = _crop_from_fov_center(img_iso)
    logger.debug(f"Crop 160³ | origin_mm={crop_origin_mm}")

    # Paso 4: Normalización de intensidad
    arr = sitk.GetArrayFromImage(img_crop).astype(np.float32)  # (z, y, x)
    arr = _normalize_intensity(arr)

    # Paso 5: Tensor (1, 1, 160, 160, 160)
    tensor = torch.tensor(arr[np.newaxis, np.newaxis].copy())

    return tensor, crop_origin_mm, img_iso


# ── Inferencia ────────────────────────────────────────────────────────────────

def run_inference(
    model: LocalizerUNet,
    tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Forward pass con torch.no_grad().
    Retorna heatmap numpy (1, 1, 160, 160, 160) float32. O(V).
    """
    with torch.no_grad():
        heatmap = model(tensor.to(device))
    return heatmap.cpu().numpy()


# ── Extracción del centroide ──────────────────────────────────────────────────

def extract_centroid(
    heatmap: np.ndarray,
    threshold: float = 0.01,
) -> tuple[np.ndarray, float]:
    """
    Centroide de masa ponderado sobre el heatmap.

    Retorna:
        c_vox : (z, y, x) en voxeles del cubo 160³
        confidence : valor máximo del heatmap ∈ [0,1]

    Si el heatmap está por debajo del umbral → retorna el centro (80,80,80).

    Complejidad: O(V) = O(160³) = O(4M) — una pasada sobre el volumen.
    """
    h = heatmap[0, 0]                    # (160, 160, 160)
    confidence = float(h.max())

    mask   = h > threshold
    if not mask.any():
        logger.warning(f"Heatmap por debajo del umbral {threshold}. Usando centro.")
        return np.array([80.0, 80.0, 80.0]), confidence

    coords  = np.argwhere(mask).astype(float)   # (N, 3) — (z, y, x)
    weights = h[mask]                            # (N,)
    c_vox   = (coords * weights[:, None]).sum(axis=0) / weights.sum()

    logger.debug(f"Centroide vox (z,y,x): {c_vox} | confidence={confidence:.4f}")
    return c_vox, confidence


def centroid_to_global_mm(
    c_vox: np.ndarray,
    crop_origin_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte centroide en voxeles del cubo 160³ a coordenadas globales en mm.

    Paso 7 del pipeline (reproduce Script 1):
        c_pred_mm_xyz  = c_pred_vox[::-1] × 0.565     ← (z,y,x) → (x,y,z)
        c_pred_global  = crop_origin_mm + c_pred_mm_xyz
        crop_fine_mm   = c_pred_global - 45.2 mm       ← origen del FineCNN

    Retorna:
        center_global_mm : (3,) array (x,y,z) en mm — coordenadas globales
        crop_fine_mm     : (3,) array (x,y,z) en mm — origen del cubo fino

    Complejidad: O(1).
    """
    # (z,y,x) voxeles → (x,y,z) mm en el cubo 160³
    c_mm_xyz = c_vox[::-1] * SPACING_ISO           # (x, y, z)

    # Coordenadas globales en el espacio del volumen isotrópico
    center_global_mm = crop_origin_mm + c_mm_xyz

    # Origen del crop fino para el FineCNN
    crop_fine_mm = center_global_mm - HALF_FOV_MM

    return center_global_mm, crop_fine_mm


def voxel_to_ras(
    center_mm: np.ndarray,
    img_iso: sitk.Image,
) -> tuple[float, float, float]:
    """
    Convierte coordenadas mm (LPS) a RAS para visualización en 3D Slicer.
    Slicer usa RAS internamente — LPS es el estándar DICOM.

    LPS → RAS: invertir los dos primeros ejes (x, y).
    Complejidad: O(1).
    """
    lps = center_mm.tolist()
    ras = (-lps[0], -lps[1], lps[2])   # LPS → RAS
    return ras
