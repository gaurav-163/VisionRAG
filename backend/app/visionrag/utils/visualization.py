"""
Visualization Utilities
Renders bounding boxes on images and scene graph diagrams.
"""

import io
import logging
from pathlib import Path

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

BBOX_COLORS = [
    (255, 87, 34),
    (33, 150, 243),
    (76, 175, 80),
    (156, 39, 176),
    (255, 193, 7),
    (0, 188, 212),
    (244, 67, 54),
    (63, 81, 181),
    (255, 152, 0),
    (96, 125, 139),
]


class VisualizationService:
    def __init__(self, output_dir: str = "static/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def draw_detections(self, image: np.ndarray, detections: list[dict], line_thickness: int = 2, font_scale: float = 0.6) -> np.ndarray:
        annotated = image.copy()
        label_color_map: dict[str, tuple] = {}
        color_idx = 0
        for detection in detections:
            label = detection["label"]
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            if label not in label_color_map:
                label_color_map[label] = BBOX_COLORS[color_idx % len(BBOX_COLORS)]
                color_idx += 1
            color = label_color_map[label]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)
            label_text = f"{label}: {confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            label_bg_y1 = max(y1 - text_h - baseline - 4, 0)
            cv2.rectangle(annotated, (x1, label_bg_y1), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            center = detection["center"]
            cv2.circle(annotated, (center["x"], center["y"]), 4, color, -1)
        return annotated

    def draw_scene_graph(self, scene_graph: dict, figsize: tuple = (14, 10)) -> bytes:
        nodes = scene_graph.get("nodes", [])
        edges = scene_graph.get("edges", [])
        if not nodes:
            return self._empty_graph_image()
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node["id"], label=node["label"])
        for edge in edges:
            G.add_edge(edge["subject"], edge["object"], relationship=edge["relationship"])
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("VisionRAG Scene Graph", fontsize=16, fontweight="bold", pad=20)
        pos = nx.spring_layout(G, seed=42, k=2.0)
        relationship_colors = {"left_of": "#2196F3", "right_of": "#FF9800", "above": "#4CAF50", "below": "#F44336", "near": "#9C27B0", "overlapping": "#FF5722"}
        for u, v, data in G.edges(data=True):
            rel = data.get("relationship", "")
            color = relationship_colors.get(rel, "#607D8B")
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, arrows=True, arrowsize=20, width=2, ax=ax, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_nodes(G, pos, node_size=1800, node_color="#263238", ax=ax)
        nx.draw_networkx_labels(G, pos, font_color="white", font_size=9, font_weight="bold", ax=ax)
        edge_labels = {(u, v): data["relationship"] for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color="#37474F", ax=ax)
        legend_patches = [mpatches.Patch(color=color, label=rel) for rel, color in relationship_colors.items()]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8, title="Relationships", title_fontsize=9)
        ax.axis("off")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def image_to_bytes(self, image: np.ndarray, format: str = ".jpg") -> bytes:
        encode_params = []
        if format == ".jpg":
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 92]
        elif format == ".png":
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
        success, buffer = cv2.imencode(format, image, encode_params)
        if not success:
            raise RuntimeError(f"Failed to encode image as {format}")
        return buffer.tobytes()

    def save_annotated_image(self, image: np.ndarray, filename: str) -> Path:
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), image)
        return output_path

    def _empty_graph_image(self) -> bytes:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No objects detected — scene graph empty", ha="center", va="center", fontsize=14, color="#607D8B", transform=ax.transAxes)
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
