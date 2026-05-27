# ProstatePipelineMCP

Pipeline agéntico para registro deformable MRI-TRUS de próstata. Combina modelos de deep learning (CoarseCNN), un motor de registro elastix, y el LLM clínico MedGemma 27B orquestados mediante LangGraph con intervención humana explícita (HITL) en cada decisión crítica.

> Proyecto de investigación doctoral — Facultad de Ingeniería, UABC  
> Autor: [@UABCerick](https://github.com/UABCerick)

---

## Descripción general

El pipeline automatiza el flujo clínico MRI-TRUS completo:

1. **Detección y localización** — carga datos del paciente, preprocesa volúmenes a 0.565 mm isotrópico, detecta la próstata con CoarseCNN dual (MRI + TRUS en paralelo), genera crops 160³ centrados en el centroide.
2. **Registro y validación** — MedGemma 27B sugiere parámetros elastix, ejecuta registro rígido + B-spline deformable, calcula métricas clínicas (Dice, HD95, TRE).

En cada etapa crítica el grafo LangGraph pausa con `interrupt()` para que el investigador verifique visualmente en 3D Slicer antes de continuar.

El chat **3D Slicer AI** permite interactuar con los volúmenes en lenguaje natural — cargar heatmaps, ejecutar registro rígido por centroides, visualizar la salida del modelo — con aprobación humana antes de ejecutar cualquier acción.

---

## Stack tecnológico

| Componente | Tecnología | Versión |
|---|---|---|
| Orquestador | LangGraph | 0.2.x |
| Servidor MCP | FastMCP | 3.3.1 |
| LLM clínico | MedGemma 27B vía Ollama | alibayram/medgemma:27b |
| Detección | CoarseCNN (PyTorch) | custom |
| Registro | SimpleElastix | ITK 6.0 |
| Visualización | 3D Slicer | 5.10 |
| MCP-Slicer | uvx mcp-slicer | Python 3.13 |
| UI | Gradio | 4.x |
| GPU | RTX 5090 / cualquier CUDA | ≥ 16 GB VRAM recomendado |

---

## Requisitos de sistema

- Windows 10/11 (probado) o Linux
- Python 3.11
- CUDA 12.x con PyTorch compatible
- 3D Slicer 5.10 con módulo **Web Server** activo en puerto `2016`
- Ollama corriendo localmente con `alibayram/medgemma:27b`
- `uvx` instalado (`pip install uv` o desde [astral.sh/uv](https://astral.sh/uv))

---

## Instalación

```bash
git clone https://github.com/UABCerick/ProstatePipelineMCP.git
cd ProstatePipelineMCP

python -m venv AIagents311
# Windows
AIagents311\Scripts\activate
# Linux/Mac
source AIagents311/bin/activate

pip install -r requirements.txt
```

Instalar MCP-Slicer en entorno separado (requiere Python 3.13):

```bash
# Instalar uv si no lo tienes (puede ser que requieras instalarlo tambien en el entorno de python 3.11)
pip install uv

# mcp-slicer se ejecuta automáticamente vía uvx — no requiere instalación manual
# Verificar que funciona:
uvx mcp-slicer --help
```

---

## Configuración — archivo `.env`

Crea un archivo `.env` en la raíz del proyecto con el siguiente contenido. **Este archivo nunca debe subirse al repositorio** (está en `.gitignore`).

```ini
# Rutas del dataset TCIA
STAGE1_JSON_PATH=C:/ruta/a/stage1_selected_pairs.json
PREPROC_DIR=C:/ruta/a/preprocessed

# Modelos CoarseCNN (ver sección de modelos más abajo)
COARSE_MRI_PATH=C:/ruta/a/coarse_MRI_fold0_best.pth
COARSE_TRUS_PATH=C:/ruta/a/coarse_TRUS_fold0_best.pth

# Ejecutable uvx para MCP-Slicer
UVX_PATH=C:/Users/tu_usuario/.local/bin/uvx.exe

# Directorio de datos del pipeline
DATA_DIR=./data

# Servidor MCP
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8765

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=alibayram/medgemma:27b

# Contraseña de la UI (opcional, default: prostate2026)
UI_PASSWORD=prostate2026
```

---

## Modelos CoarseCNN

Los modelos CoarseCNN entrenados sobre el dataset TCIA Prostate-MRI-US-Biopsy siguen en desarrollo activo.

Si deseas acceder a los modelos para reproducir los experimentos, puedes solicitarlos por:
- **GitHub**: mensaje directo a [@UABCerick](https://github.com/UABCerick)
- **Correo institucional UABC**: erick.martinez@uabc.edu.mx

Para entrenar tus propios modelos, el pipeline de entrenamiento utiliza validación cruzada de 5 folds sobre el dataset TCIA. Los modelos esperan volúmenes NIfTI isotrópicos a **0.565 mm** con patches de **32³ voxeles**.

---

## Dataset — estructura esperada

### Datos TCIA (Prostate-MRI-US-Biopsy)

Descarga el dataset desde [TCIA](https://www.cancerimagingarchive.net/collection/prostate-mri-us-biopsy/).

El pipeline espera un archivo `stage1_selected_pairs.json` con la siguiente estructura:

```json
{
  "patients": {
    "0001": {
      "MRI_dicom_path": "C:/ruta/TCIA/Paciente0001/MRI_DICOM/",
      "TRUS_exported_path": "C:/ruta/TCIA/Paciente0001/TRUS.nii.gz",
      "MRI_study_date": "20091219"
    }
  }
}
```

El campo `MRI_dicom_path` debe apuntar a una **carpeta** con archivos `.dcm`. El campo `TRUS_exported_path` debe apuntar a un **archivo NIfTI** exportado desde 3D Slicer.

### Estructura de directorios esperada

```
ProstatePipelineMCP/
├── .env                          ← debes crear este archivo
├── data/
│   ├── preprocessed/             ← generado automáticamente por el pipeline
│   │   └── {pid}/
│   │       ├── {pid}_MRI_vol_iso.nii.gz
│   │       └── {pid}_TRUS_vol_iso.nii.gz
│   ├── coarse/                   ← generado automáticamente
│   │   └── {pid}/
│   │       ├── {pid}_MRI_loc_160.nii.gz
│   │       ├── {pid}_TRUS_loc_160.nii.gz
│   │       ├── {pid}_MRI_coarse_heatmap.nii.gz
│   │       ├── {pid}_TRUS_coarse_heatmap.nii.gz
│   │       ├── {pid}_MRI_coarse_heatmap_crop160.nii.gz
│   │       └── {pid}_TRUS_coarse_heatmap_crop160.nii.gz
│   ├── registrations/            ← generado en Fase 3 (pendiente)
│   ├── custom_patients.json      ← generado al registrar pacientes custom
│   └── last_pipeline_meta.json   ← generado tras cada detección CoarseCNN
├── mcp_server/
│   ├── server.py                 ← 16 tools MCP, FastMCP HTTP/SSE :8765
│   ├── schemas.py                ← validación Pydantic de inputs (anti prompt-injection)
│   ├── schemas_phase2.py         ← schemas DetectProstateInput, DetectionResult
│   ├── coarse_localizer.py       ← inferencia CoarseCNN, centroide ponderado, heatmap NIfTI
│   ├── preprocessor.py           ← resampleo B-spline isotrópico 0.565mm
│   ├── registration_engine.py    ← motor elastix rígido + B-spline deformable
│   ├── raw_patient_manager.py    ← lookup O(1) en stage1_selected_pairs.json (690 pacientes)
│   ├── dicom_utils.py            ← DICOMOrient, reset geometría, extracción de series
│   ├── slicer_bridge.py          ← bridge HTTP → Slicer Web Server :2016
│   ├── dataset_manager.py        ← gestor del dataset Localizer (641 casos preprocesados)
│   ├── case_registry.py          ← registro persistente de casos en JSON
│   └── localizer_model.py        ← arquitectura LocalizerUNet (legacy, mantenido por compatibilidad)
├── langgraph_agent/
│   ├── graph.py                  ← grafo LangGraph + nodos HITL
│   ├── nodes.py                  ← lógica de cada nodo del pipeline
│   ├── state.py                  ← PipelineState Pydantic
│   └── llm_client.py             ← cliente Ollama
├── UI.py                         ← interfaz Gradio (:7860)
├── list_slicer_tools.py          ← diagnóstico: lista tools MCP-Slicer
└── test_coarse_inference.py      ← prueba standalone CoarseCNN
```

### Pacientes propios (datos custom)

Si tus datos no provienen del dataset TCIA, puedes registrarlos desde la UI en la pestaña **Registrar Paciente**. Necesitas:

- Una **carpeta con archivos `.dcm`** para el MRI
- Un **archivo `.nii` o `.nii.gz`** exportado desde 3D Slicer para el TRUS

Los datos custom se guardan en `data/custom_patients.json` y son accesibles automáticamente desde el pipeline.

---

## Ejecución

### Prerequisitos antes de iniciar

1. 3D Slicer 5.10 abierto con el módulo **Web Server** activo (puerto 2016)
2. Ollama corriendo: `ollama serve` y modelo cargado: `ollama run alibayram/medgemma:27b`
3. Entorno virtual activado

### Terminal 1 — Servidor MCP

```bash
python -m mcp_server.server --http
```

El servidor queda en `http://localhost:8765`.

### Terminal 2 — Interfaz web

```bash
python UI.py
```

Accede en `http://127.0.0.1:7860`. La contraseña por defecto es `prostate2026` (configurable en `.env`).

---

## Herramientas de diagnóstico

```bash
# Listar herramientas disponibles en MCP-Slicer
python list_slicer_tools.py

# Probar inferencia CoarseCNN standalone
python test_coarse_inference.py
```

---

## Flujo del pipeline

### Detección y localización (activo)

```
START → load → preprocess → detect → [HITL: fov] → crop_fine → [HITL: crops] → END
```

### Registro y validación (implementado, pendiente de activación)

```
... → plan_params → [HITL: params] → register → validate → [HITL: final] → report → END
```

Para activar el registro, cambiar en `langgraph_agent/graph.py`:

```python
# Línea actual:
builder.add_edge("hitl_crops", END)

# Cambiar a:
builder.add_edge("hitl_crops", "plan_params")
```

---

## Chat 3D Slicer AI

El tab **3D Slicer AI** permite interactuar con los volúmenes en lenguaje natural. Ejemplos de prompts:

- `"Realiza el registro rígido del caso 0001"` — ejecuta `rigid_registration_by_centroid`
- `"Muéstrame cómo el modelo detectó la próstata en el caso 0001"` — ejecuta `load_heatmaps_with_grid`
- `"Carga el MRI del caso 0001 y ajusta la visualización"` — genera código Python para Slicer

Requiere contraseña de sesión. El modelo propone las acciones y el investigador aprueba antes de ejecutar.

---

## Seguridad

- La contraseña de sesión protege todas las acciones de la UI
- El sistema prompt del chat incluye reglas explícitas contra ejecución de comandos del sistema y acceso a archivos fuera del pipeline
- El panel de aprobación HITL requiere confirmación humana antes de ejecutar cualquier código en 3D Slicer
- El archivo `.env` y los datos de pacientes están excluidos del repositorio vía `.gitignore`

---

## Citación

Si utilizas este trabajo en tu investigación:

```
@misc{prostatepipelinemcp2026,
  author = {UABCerick},
  title  = {ProstatePipelineMCP: Agentic Pipeline for MRI-TRUS Rigid/Deformable Registration},
  year   = {2026},
  url    = {https://github.com/UABCerick/ProstatePipelineMCP}
}
```

---

## Licencia

Este proyecto es de uso educativo y de investigación. Los modelos CoarseCNN y el dataset TCIA tienen sus propias licencias — consultar los términos en [TCIA](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/).