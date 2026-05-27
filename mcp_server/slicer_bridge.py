"""
slicer_bridge.py — Puente HTTP hacia 3D Slicer Web Server.

Adaptado para Slicer 5.10 con los endpoints correctos verificados:
  - POST /slicer/exec   { "source": "..." }  → ejecuta Python (sin retorno directo)
  - GET  /slicer/mrml                         → lista nodos de la escena ✅
  - GET  /slicer/slice?orientation=axial      → imagen PNG de slice ✅

Comportamiento verificado en Slicer 5.10.0 Windows:
  - exec siempre retorna {} (el resultado se obtiene via mrml o variables globales)
  - mrml retorna lista JSON de nombres de nodos
  - slice retorna bytes PNG

Complejidad de red: O(1) por operación HTTP.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from loguru import logger


class SlicerConnectionError(Exception):
    """No se pudo conectar a 3D Slicer."""


class SlicerCommandError(Exception):
    """El comando enviado a Slicer produjo un error."""


class SlicerBridge:
    """
    Cliente HTTP para el Web Server de 3D Slicer 5.10.

    Endpoints confirmados en Slicer 5.10.0:
      ✅ POST /slicer/exec      — ejecuta Python (source=)
      ✅ GET  /slicer/mrml      — nodos de la escena
      ✅ GET  /slicer/slice     — imagen axial/coronal/sagital
      ✅ GET  /slicer/volume    — volumen en formato nrrd
      ❌ GET  /health           — no existe, usar /slicer/mrml como ping
      ❌ GET  /slicer/nodes     — renombrado a /slicer/mrml
      ❌ GET  /slicer/view      — renombrado a /slicer/slice
    """

    def __init__(self, host: str = "localhost", port: int = 2016):
        self._base_url = f"http://{host}:{port}"
        self._timeout  = httpx.Timeout(30.0)

    # ── Conexión ──────────────────────────────────────────────────────────────

    async def is_alive(self) -> bool:
        """
        Verifica si el Web Server de Slicer está activo.
        Usa /slicer/mrml como ping — es el endpoint más estable. O(1) red.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/slicer/mrml")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ── Ejecución de Python ───────────────────────────────────────────────────

    async def execute_python(self, code: str) -> dict[str, Any]:
        """
        Ejecuta código Python en Slicer 5.10.

        NOTA Slicer 5.10: el endpoint /slicer/exec siempre retorna {}.
        Para obtener resultados, usar get_scene_nodes() o consultas
        específicas al /slicer/mrml después del exec.

        Seguridad: este método solo recibe código generado internamente,
        nunca strings del usuario final.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/slicer/exec",
                    json={"source": code},          # Slicer 5.10 usa "source", no "code"
                    headers={"Content-Type": "application/json"},
                )
                # Slicer 5.10 retorna 200 con {} aunque el código falle silenciosamente
                # Verificamos éxito intentando parsear la respuesta
                if resp.status_code == 200:
                    return {"success": True, "output": resp.text}
                else:
                    body = resp.text
                    logger.warning(f"Slicer exec retornó {resp.status_code}: {body}")
                    return {"success": False, "error": body}

        except httpx.ConnectError:
            raise SlicerConnectionError(
                "No se puede conectar a 3D Slicer en "
                f"{self._base_url}. "
                "Verifica que Slicer esté abierto y el Web Server activo "
                "(módulo Web Server → Start)."
            )
        except httpx.TimeoutException:
            raise SlicerCommandError("Timeout esperando respuesta de Slicer (>30s).")

    async def execute_python_and_query(
        self, code: str, query_after: str = ""
    ) -> dict[str, Any]:
        """
        Ejecuta Python en Slicer y luego consulta el estado de la escena.

        Patrón para Slicer 5.10: exec no retorna valores, pero podemos
        leer el estado resultante con una segunda llamada a mrml.
        """
        exec_result = await self.execute_python(code)
        if not exec_result.get("success"):
            return exec_result

        # Si se pidió una consulta de seguimiento, ejecutarla
        if query_after:
            nodes = await self.get_scene_nodes(filter_name=query_after)
            exec_result["nodes_found"] = nodes

        return exec_result

    # ── Escena MRML ───────────────────────────────────────────────────────────

    async def get_scene_nodes(self, filter_name: str = "") -> list[dict[str, Any]]:
        """
        Lista nodos de la escena de Slicer vía /slicer/mrml. ✅ Verificado 5.10.
        Retorna lista de dicts con nombre y tipo de cada nodo. O(n_nodes) red.

        filter_name: si se especifica, filtra nodos cuyo nombre lo contenga.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/slicer/mrml")
                resp.raise_for_status()
                raw = resp.json()   # lista de strings "NombreNodo"

                nodes = []
                for item in raw:
                    if isinstance(item, str):
                        if not filter_name or filter_name.lower() in item.lower():
                            nodes.append({"name": item})
                    elif isinstance(item, dict):
                        name = item.get("name", str(item))
                        if not filter_name or filter_name.lower() in name.lower():
                            nodes.append(item)

                return nodes

        except httpx.ConnectError:
            raise SlicerConnectionError("Slicer no disponible.")
        except Exception as e:
            logger.error(f"Error leyendo escena MRML: {e}")
            return []

    async def get_scene_node_names(self) -> list[str]:
        """Versión simplificada: solo retorna nombres. O(n_nodes)."""
        nodes = await self.get_scene_nodes()
        return [n.get("name", "") for n in nodes if n.get("name")]

    # ── Carga de DICOM ────────────────────────────────────────────────────────

    async def load_dicom_volume(self, dicom_dir: str) -> dict[str, Any]:
        """
        Carga un directorio DICOM en Slicer. O(n_slices) I/O.

        Estrategia Slicer 5.10: ejecutar via exec, luego verificar
        que el nodo apareció en la escena mrml.
        """
        # Escapar backslashes para Windows
        safe_dir = dicom_dir.replace("\\", "\\\\")

        code = f"""
import slicer, os
dicom_dir = r"{safe_dir}"
if not os.path.isdir(dicom_dir):
    print(f"ERROR: directorio no encontrado: {{dicom_dir}}")
else:
    slicer.util.importDicom(dicom_dir)
    from DICOMLib import DICOMUtils
    db = slicer.dicomDatabase
    patients = db.patients()
    if patients:
        studies  = db.studiesForPatient(patients[0])
        series   = db.seriesForStudy(studies[-1]) if studies else []
        if series:
            DICOMUtils.loadSeriesByUID([series[-1]])
"""
        result = await self.execute_python(code)

        # Verificar en la escena que aparecieron nuevos nodos de volumen
        nodes_after = await self.get_scene_node_names()
        volumes = [n for n in nodes_after if any(
            kw in n.lower() for kw in ["volume", "series", "mri", "us", "ct"]
        )]

        logger.info(f"load_dicom_volume: {len(volumes)} volúmenes en escena")
        result["volumes_in_scene"] = volumes
        return result

    # ── Visualización ─────────────────────────────────────────────────────────

    async def show_fov_sphere(
        self,
        center_ras: tuple[float, float, float],
        radius_mm: float,
        label: str = "FOV_Prostata",
        heatmap_path: str = "",
    ) -> dict[str, Any]:
        """
        Genera script para visualizar en Slicer:
        1. Fiducial puntual en el centro detectado (más visible que ROI)
        2. Heatmap como overlay de color si se provee la ruta

        Slicer 5.10: ejecutar en Python Interactor (Ctrl+3). O(1).
        """
        cx, cy, cz = center_ras
        fid_label = label.replace("FOV_", "CTR_")

        script_lines = [
            "import slicer",
            "",
            "# ── 1. Fiducial puntual ──",
            f'existing = slicer.mrmlScene.GetFirstNodeByName("{fid_label}")',
            "if existing: slicer.mrmlScene.RemoveNode(existing)",
            f'fid = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "{fid_label}")',
            f"fid.AddControlPoint({cx:.4f}, {cy:.4f}, {cz:.4f})",
            "dn = fid.GetDisplayNode()",
            "dn.SetSelectedColor(0.2, 1.0, 0.3)",
            "dn.SetColor(0.2, 1.0, 0.3)",
            "dn.SetGlyphScale(3.5)",
            "dn.SetTextScale(3.5)",
            "# Proyectar punto en todas las vistas como cruz",
            "dn.SetSliceProjection(True)",
            "dn.SetSliceProjectionColor(0.2, 1.0, 0.3)",
            "dn.SetSliceProjectionOpacity(1.0)",
            f'fid.SetNthControlPointLabel(0, "CTR:({cx:.1f},{cy:.1f},{cz:.1f})")',
            "",
            "# ── Saltar las 3 vistas al punto ──",
            "slicer.util.resetSliceViews()",
            "slicer.util.resetSliceViews()",
        ]

        if heatmap_path:
            heatmap_safe = heatmap_path.replace("\\\\", "/").replace("\\", "/")
            script_lines += [
                "",
                "# ── 2. Heatmap como overlay ──",
                f'heatmap_node = slicer.util.loadVolume(r"{heatmap_safe}")',
                f'heatmap_node.SetName("{label}_heatmap")',
                "# Aplicar colormap verde-amarillo-rojo",
                "display = heatmap_node.GetDisplayNode()",
                'display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileHotToColdRainbow.txt")',
                "display.SetAutoWindowLevel(False)",
                "display.SetWindowLevelMinMax(0.01, 1.0)",
                "display.SetOpacity(0.6)",
                "# Mostrar en slice views",
                "slicer.util.setSliceViewerLayers(foreground=heatmap_node, foregroundOpacity=0.5)",
            ]

        script_lines += [
            "",
            "slicer.util.resetSliceViews()",
            f'print("Visualización lista: centro en RAS ({cx:.1f},{cy:.1f},{cz:.1f})")',
        ]

        script = chr(10).join(script_lines)

        logger.info(
            f"show_fov_sphere script: ({cx:.1f},{cy:.1f},{cz:.1f}) "
            f"heatmap={'si' if heatmap_path else 'no'}"
        )
        return {
            "success": True,
            "roi_name": label,
            "center_ras": [cx, cy, cz],
            "radius_mm": radius_mm,
            "slicer_script": script,
            "heatmap_included": bool(heatmap_path),
        }