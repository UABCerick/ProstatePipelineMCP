"""
test_coarse_inference.py — Prueba rápida del CoarseCNN sobre un volumen crudo.

USO:
    python test_coarse_inference.py --pid 0001 --modality MRI

Requiere:
    - pip install torch SimpleITK matplotlib
    - Los modelos coarse en sus rutas configuradas
    - Un volumen vol_iso disponible (del dataset preprocesado o crudo)

Salida:
    ./test_output/{pid}_{modality}_coarse_result.png  ← figura matplotlib
    ./test_output/{pid}_{modality}_heatmap.nii.gz     ← heatmap para Slicer
"""
import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import torch

# ── Agregar el proyecto al path ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcp_server.coarse_localizer import (
    CoarseCNN, load_coarse_model,
    preprocess_raw_volume, run_coarse_inference,
    crop_fine_volume, save_prob_heatmap_nifti,
    PATCH_SIZE, DST_SPACING, CROP_SIZE,
)

# ── Argumentos ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--pid",      type=str, default="0001")
parser.add_argument("--modality", type=str, default="MRI", choices=["MRI","TRUS"])
parser.add_argument("--use_preproc", action="store_true",
                    help="Usar vol_iso preprocesado en lugar de DICOM crudo")
args = parser.parse_args()

PID      = args.pid
MODALITY = args.modality
P        = PATCH_SIZE

# ── Rutas ─────────────────────────────────────────────────────────────────────
COARSE_MODELS = {
    "MRI":  r"C:\Codes\Us_MRI_Fusion\PRE-TCIA\Training\step1_coarse\MRI\models\coarse_MRI_fold0_best.pth",
    "TRUS": r"C:\Codes\Us_MRI_Fusion\PRE-TCIA\Training\step1_coarse\TRUS\models\coarse_TRUS_fold0_best.pth",
}

# Dataset preprocesado (para prueba rápida)
PREPROC_DIR = r"C:\Codes\Us_MRI_Fusion\PRE-TCIA\Preprocessing\pipeline_v3\preprocessed"

# Dataset Localizer (vol_iso ya a 0.565mm)
LOCALIZER_DIR = r"C:\Codes\Us_MRI_Fusion\TCIA\CLEAN_COUD_P_REGISTER\Preprocessed\Localizer_160_0p565"

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Cargar vol_iso ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"CoarseCNN Test | PID={PID} | Modality={MODALITY}")
print(f"{'='*60}\n")

# Intentar cargar desde diferentes fuentes en orden de prioridad
vol_iso      = None
gt_centroid  = None
source_used  = ""

# Opción 1: vol_iso del pipeline de preprocesamiento
preproc_path = os.path.join(PREPROC_DIR, PID, f"{MODALITY}_vol_iso.nii.gz")
if os.path.exists(preproc_path):
    print(f"✓ Usando vol_iso preprocesado: {preproc_path}")
    vol_iso = sitk.ReadImage(preproc_path)
    source_used = "preprocessed"

    # Intentar cargar GT
    import json
    preproc_json = os.path.join(PREPROC_DIR, "preprocessed_dataset.json")
    if os.path.exists(preproc_json):
        with open(preproc_json) as f:
            preproc_data = json.load(f)
        patient_data = preproc_data.get("patients", {}).get(PID, {})
        gt_key = f"{MODALITY}_c_GT_mm"
        if gt_key in patient_data:
            gt_centroid = patient_data[gt_key]
            print(f"✓ GT centroide: {gt_centroid} mm")

# Opción 2: vol del dataset Localizer (ya a 0.565mm, 160³)
if vol_iso is None:
    localizer_path = os.path.join(LOCALIZER_DIR, f"{PID}_{MODALITY}_loc_160.nii.gz")
    if os.path.exists(localizer_path):
        print(f"✓ Usando vol del dataset Localizer: {localizer_path}")
        vol_iso = sitk.ReadImage(localizer_path)
        source_used = "localizer_dataset"

        # GT del dataset Localizer
        import json
        ds_json = os.path.join(LOCALIZER_DIR, "dataset_index_160_localizer_641.json")
        if os.path.exists(ds_json):
            with open(ds_json) as f:
                ds_data = json.load(f)
            case = ds_data.get(PID, {})
            key = f"{MODALITY}_centroid_local_mm_xyz"
            if key in case:
                gt_centroid = case[key]
                print(f"✓ GT centroide (local): {gt_centroid} mm")

if vol_iso is None:
    print("❌ No se encontró vol_iso. Verifica las rutas.")
    sys.exit(1)

print(f"\nVol ISO: {vol_iso.GetSize()} @ {vol_iso.GetSpacing()[0]:.4f}mm")

# ── Cargar modelo ─────────────────────────────────────────────────────────────
print(f"\nCargando CoarseCNN {MODALITY}...")
model_path = COARSE_MODELS[MODALITY]
if not os.path.exists(model_path):
    print(f"❌ Modelo no encontrado: {model_path}")
    sys.exit(1)

model, device = load_coarse_model(model_path)
print(f"✓ Modelo cargado en {device}")

# ── Inferencia ────────────────────────────────────────────────────────────────
print(f"\nCorriendo inferencia CoarseCNN...")
infer_result = run_coarse_inference(vol_iso, model, device)

centroid_mm  = infer_result["centroid_mm"]
prob_map     = np.array(infer_result["prob_map"])   # (nK,nJ,nI)
grid_shape   = infer_result["grid_shape"]            # (nI,nJ,nK)
nI, nJ, nK   = grid_shape
max_prob     = infer_result["max_prob"]

print(f"\n✓ Centroide predicho: ({centroid_mm[0]:.1f}, {centroid_mm[1]:.1f}, {centroid_mm[2]:.1f}) mm")
print(f"✓ Max prob: {max_prob:.3f}")
print(f"✓ Grid: {nI}×{nJ}×{nK} = {infer_result['total_patches']} patches")

if gt_centroid:
    error = np.linalg.norm(np.array(centroid_mm) - np.array(gt_centroid))
    diff  = np.array(centroid_mm) - np.array(gt_centroid)
    print(f"✓ Error vs GT: {error:.1f}mm  (X={diff[0]:.1f} Y={diff[1]:.1f} Z={diff[2]:.1f}mm)")

# ── Guardar heatmap NIfTI ─────────────────────────────────────────────────────
heatmap_path = os.path.join(OUT_DIR, f"{PID}_{MODALITY}_coarse_heatmap.nii.gz")
save_prob_heatmap_nifti(prob_map, grid_shape, vol_iso, heatmap_path)
print(f"\n✓ Heatmap guardado: {heatmap_path}")

# ── Crop 160³ ─────────────────────────────────────────────────────────────────
vol_crop, crop_origin = crop_fine_volume(vol_iso, centroid_mm)
crop_path = os.path.join(OUT_DIR, f"{PID}_{MODALITY}_loc_160_new.nii.gz")
sitk.WriteImage(vol_crop, crop_path, useCompression=True)
print(f"✓ Crop 160³ guardado: {crop_path}")

# ── Visualización ─────────────────────────────────────────────────────────────
print(f"\nGenerando figura...")

# Índices del centroide predicho
cx_vox = int(np.clip(round(centroid_mm[0] / DST_SPACING), 0, vol_iso.GetSize()[0]-1))
cy_vox = int(np.clip(round(centroid_mm[1] / DST_SPACING), 0, vol_iso.GetSize()[1]-1))
cz_vox = int(np.clip(round(centroid_mm[2] / DST_SPACING), 0, vol_iso.GetSize()[2]-1))

vol_arr = sitk.GetArrayFromImage(vol_iso).astype(np.float32)  # (Z,Y,X)
Z, Y, X = vol_arr.shape

# Normalizar para display
p1, p99 = np.percentile(vol_arr, 1), np.percentile(vol_arr, 99)
vol_norm = np.clip((vol_arr - p1) / (p99 - p1 + 1e-8), 0, 1)

# Colormap patches
cmap = plt.cm.RdYlBu_r
norm_c = Normalize(vmin=0, vmax=1)
sm   = ScalarMappable(cmap=cmap, norm=norm_c)

def get_probs_at_slice(axis, slice_idx):
    """Retorna lista de (rect_xy, rect_wh, prob) para los patches en ese slice."""
    items = []
    probs_flat = np.array(infer_result["probs"])
    centers    = np.array(infer_result["centers_mm"])
    P_mm       = P * DST_SPACING

    for idx, (cx_m, cy_m, cz_m) in enumerate(centers):
        # Origen del patch en voxeles
        i = int(round((cx_m / DST_SPACING - P/2) / P))
        j = int(round((cy_m / DST_SPACING - P/2) / P))
        k = int(round((cz_m / DST_SPACING - P/2) / P))
        x0, y0, z0 = i*P, j*P, k*P

        prob = probs_flat[idx]

        if axis == 'z' and z0 <= slice_idx < z0 + P:
            items.append(((x0, y0), P, prob))
        elif axis == 'y' and y0 <= slice_idx < y0 + P:
            items.append(((x0, z0), P, prob))
        elif axis == 'x' and x0 <= slice_idx < x0 + P:
            items.append(((y0, z0), P, prob))
    return items

# ── Figura: 2 filas × 3 columnas (como en tu ejemplo) ────────────────────────
fig = plt.figure(figsize=(22, 13), facecolor='#0d1117')

err_str = f"Error: {error:.1f}mm  (X={diff[0]:.1f} Y={diff[1]:.1f} Z={diff[2]:.1f}mm)  ✓ <15mm" \
    if gt_centroid else f"Centroide: ({centroid_mm[0]:.1f},{centroid_mm[1]:.1f},{centroid_mm[2]:.1f})mm"

fig.suptitle(
    f"PID {PID} — {MODALITY} | Coarse Inference\n{err_str}",
    color='white', fontsize=13, fontweight='bold', y=0.98
)

axes_top = [fig.add_subplot(2, 3, i+1) for i in range(3)]
axes_bot = [fig.add_subplot(2, 3, i+4) for i in range(3)]

slices = [
    ('z', cz_vox, vol_norm[cz_vox,:,:], f"Axial z={cz_vox}",
     lambda s: (cx_vox, cy_vox)),
    ('y', cy_vox, vol_norm[:,cy_vox,:], f"Coronal y={cy_vox}",
     lambda s: (cx_vox, cz_vox)),
    ('x', cx_vox, vol_norm[:,:,cx_vox], f"Sagital x={cx_vox}",
     lambda s: (cy_vox, cz_vox)),
]

for col, (axis, sl_idx, vol_sl, title, get_cross) in enumerate(slices):

    # ── Fila superior: volumen + máscara si existe ────────────────────────────
    ax = axes_top[col]
    ax.set_facecolor('#111111')
    ax.imshow(vol_sl, cmap='gray', aspect='equal', origin='upper')
    ax.set_title(title, color='white', fontsize=10)
    ax.axis('off')

    # Cruz centroide predicho (azul)
    cx_c, cy_c = get_cross(sl_idx)
    ax.scatter([cx_c], [cy_c], c='#00aaff', marker='+',
               s=300, linewidths=3, zorder=10, label='c_coarse')

    # Cruz GT si existe (verde)
    if gt_centroid:
        gx = int(round(gt_centroid[0] / DST_SPACING))
        gy = int(round(gt_centroid[1] / DST_SPACING))
        gz = int(round(gt_centroid[2] / DST_SPACING))
        if axis == 'z':
            ax.scatter([gx], [gy], c='#00ff88', marker='+',
                       s=300, linewidths=3, zorder=11, label='c_GT')
        elif axis == 'y':
            ax.scatter([gx], [gz], c='#00ff88', marker='+',
                       s=300, linewidths=3, zorder=11, label='c_GT')
        else:
            ax.scatter([gy], [gz], c='#00ff88', marker='+',
                       s=300, linewidths=3, zorder=11, label='c_GT')

    if col == 0:
        from matplotlib.lines import Line2D
        legend_els = [
            Line2D([0],[0], marker='+', color='w', markerfacecolor='#00ff88',
                   markersize=10, label='c_GT', linewidth=0),
            Line2D([0],[0], marker='+', color='w', markerfacecolor='#00aaff',
                   markersize=10, label='c_coarse', linewidth=0),
        ]
        ax.legend(handles=legend_els, loc='upper left',
                  fontsize=7, facecolor='#1a1a1a', labelcolor='white')

    # ── Fila inferior: patches con probabilidades ─────────────────────────────
    ax2 = axes_bot[col]
    ax2.set_facecolor('#111111')
    ax2.imshow(vol_sl, cmap='gray', aspect='equal', origin='upper', alpha=0.7)
    ax2.set_title(f"patches p_pred | {title}", color='white', fontsize=9)
    ax2.axis('off')

    items = get_probs_at_slice(axis, sl_idx)
    for (rx, ry), rp, prob in items:
        color = cmap(norm_c(prob))
        lw    = 2.5 if prob > 0.1 else 0.8
        rect  = mpatches.Rectangle(
            (rx, ry), rp, rp,
            linewidth=lw,
            edgecolor=color,
            facecolor=(*color[:3], 0.25 if prob > 0.1 else 0.05),
        )
        ax2.add_patch(rect)
        if prob > 0.05:
            ax2.text(
                rx + rp/2, ry + rp/2, f"{prob:.2f}",
                ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1',
                          facecolor='black', alpha=0.5, edgecolor='none')
            )

    # Cruz predicha
    ax2.scatter([cx_c], [cy_c], c='#00aaff', marker='+',
                s=300, linewidths=3, zorder=10)
    if gt_centroid:
        if axis == 'z':
            ax2.scatter([gx], [gy], c='#00ff88', marker='+',
                        s=300, linewidths=3, zorder=11)
        elif axis == 'y':
            ax2.scatter([gx], [gz], c='#00ff88', marker='+',
                        s=300, linewidths=3, zorder=11)
        else:
            ax2.scatter([gy], [gz], c='#00ff88', marker='+',
                        s=300, linewidths=3, zorder=11)

# Colorbar
cax = fig.add_axes([0.92, 0.12, 0.013, 0.75])
cb  = fig.colorbar(sm, cax=cax)
cb.set_label('p_pred', color='white', fontsize=10)
cb.ax.yaxis.set_tick_params(color='white')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

# Info texto lateral
info_text = (
    f"PID {PID}\n{MODALITY}\n"
    f"vol_iso + centroides\n\n"
    f"c_GT:\n({gt_centroid[0]:.1f},{gt_centroid[1]:.1f},{gt_centroid[2]:.1f})mm\n"
    f"c_coarse:\n({centroid_mm[0]:.1f},{centroid_mm[1]:.1f},{centroid_mm[2]:.1f})mm\n\n"
    f"✓ <15mm"
) if gt_centroid else (
    f"PID {PID}\n{MODALITY}\n\n"
    f"c_coarse:\n({centroid_mm[0]:.1f},{centroid_mm[1]:.1f},{centroid_mm[2]:.1f})mm\n"
    f"max_prob: {max_prob:.3f}"
)

fig.text(0.01, 0.5, info_text, color='#00ff88', fontsize=8,
         va='center', fontfamily='monospace',
         bbox=dict(facecolor='#111111', alpha=0.7, edgecolor='none', pad=6))

plt.subplots_adjust(left=0.10, right=0.91, top=0.93, bottom=0.05,
                    wspace=0.05, hspace=0.12)

out_fig = os.path.join(OUT_DIR, f"inference_{PID}_{MODALITY}.png")
fig.savefig(out_fig, dpi=130, bbox_inches='tight', facecolor='#0d1117')
plt.close(fig)
print(f"✓ Figura guardada: {out_fig}")

# ── Script de Slicer ──────────────────────────────────────────────────────────
safe_vol  = str(os.path.join(
    PREPROC_DIR if source_used == "preprocessed" else LOCALIZER_DIR,
    f"{PID}_{MODALITY}_vol_iso.nii.gz" if source_used == "preprocessed"
    else f"{PID}_{MODALITY}_loc_160.nii.gz"
)).replace("\\", "/")
safe_heat = heatmap_path.replace("\\", "/")
cx, cy, cz = [round(v,1) for v in centroid_mm]

slicer_script = f"""import slicer
vol  = slicer.util.loadVolume(r"{safe_vol}")
heat = slicer.util.loadVolume(r"{safe_heat}")
heat.SetName("Coarse_Heatmap_{PID}")
dn = heat.GetDisplayNode()
dn.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileHotToColdRainbow.txt")
dn.SetAutoWindowLevel(False)
dn.SetWindowLevelMinMax(0.05, 1.0)
dn.SetOpacity(0.6)
slicer.util.setSliceViewerLayers(background=vol, foreground=heat, foregroundOpacity=0.5)
fid = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Centroid_Coarse_{PID}")
fid.AddControlPoint({cx}, {cy}, {cz})
fid.GetDisplayNode().SetSelectedColor(0.2, 1.0, 0.3)
fid.GetDisplayNode().SetGlyphScale(3.0)
slicer.util.resetSliceViews()
print("Centroide coarse: ({cx},{cy},{cz}) mm")
"""

script_path = os.path.join(OUT_DIR, f"slicer_viz_{PID}_{MODALITY}.py")
with open(script_path, 'w') as f:
    f.write(slicer_script)
print(f"✓ Script Slicer: {script_path}")

print(f"\n{'='*60}")
print(f"RESULTADO FINAL")
print(f"{'='*60}")
print(f"  Centroide predicho : ({centroid_mm[0]:.1f}, {centroid_mm[1]:.1f}, {centroid_mm[2]:.1f}) mm")
if gt_centroid:
    print(f"  Centroide GT       : ({gt_centroid[0]:.1f}, {gt_centroid[1]:.1f}, {gt_centroid[2]:.1f}) mm")
    print(f"  Error              : {error:.1f} mm")
print(f"  Max probabilidad   : {max_prob:.3f}")
print(f"  Figura             : {out_fig}")
print(f"  Heatmap NIfTI      : {heatmap_path}")
print(f"  Crop 160³          : {crop_path}")
print(f"  Script Slicer      : {script_path}")
print(f"{'='*60}\n")