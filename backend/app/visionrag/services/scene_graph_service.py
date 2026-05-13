"""
Scene Graph Service
Analyzes spatial relationships between detected objects and builds
a structured scene graph representation for VisionRAG queries.
"""

import logging
from itertools import combinations
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

NEAR_DISTANCE_RATIO = 0.25
OVERLAP_THRESHOLD = 0.05


class SceneGraphService:
    def __init__(self, near_ratio: float = NEAR_DISTANCE_RATIO, overlap_threshold: float = OVERLAP_THRESHOLD):
        self.near_ratio = near_ratio
        self.overlap_threshold = overlap_threshold

    def build_scene_graph(self, detections: list[dict], image_width: int, image_height: int) -> dict:
        if not detections:
            return self._empty_scene_graph()
        nodes = self._build_nodes(detections)
        image_diagonal = self._compute_diagonal(image_width, image_height)
        edges = self._build_edges(nodes, image_diagonal)
        nx_graph = self._build_networkx_graph(nodes, edges)
        triplets = self._edges_to_triplets(edges)
        return {"nodes": nodes, "edges": edges, "triplets": triplets, "summary": self._build_summary(nodes, edges, nx_graph)}

    def _build_nodes(self, detections: list[dict]) -> list[dict]:
        label_counts: dict[str, int] = {}
        nodes = []
        for detection in detections:
            label = detection["label"]
            count = label_counts.get(label, 0)
            label_counts[label] = count + 1
            node_id = f"{label}_{count}" if count > 0 else label
            nodes.append({"id": node_id, "label": label, "confidence": detection["confidence"], "bbox": detection["bbox"], "center": detection["center"], "area": detection["area"]})
        return nodes

    def _build_edges(self, nodes: list[dict], image_diagonal: float) -> list[dict]:
        edges = []
        near_threshold = image_diagonal * self.near_ratio
        for node_a, node_b in combinations(nodes, 2):
            relationships = self._compute_relationships(node_a, node_b, near_threshold)
            for relationship in relationships:
                edges.append({"subject": node_a["id"], "subject_label": node_a["label"], "relationship": relationship, "object": node_b["id"], "object_label": node_b["label"]})
                inverse = self._get_inverse_relationship(relationship)
                if inverse:
                    edges.append({"subject": node_b["id"], "subject_label": node_b["label"], "relationship": inverse, "object": node_a["id"], "object_label": node_a["label"]})
        return edges

    def _compute_relationships(self, node_a: dict, node_b: dict, near_threshold: float) -> list[str]:
        relationships = []
        center_a = node_a["center"]
        center_b = node_b["center"]
        bbox_a = node_a["bbox"]
        bbox_b = node_b["bbox"]
        iou = self._compute_iou(bbox_a, bbox_b)
        if iou >= self.overlap_threshold:
            relationships.append("overlapping")
            return relationships
        distance = self._euclidean_distance(center_a, center_b)
        if distance < near_threshold:
            relationships.append("near")
        dx = center_a["x"] - center_b["x"]
        dy = center_a["y"] - center_b["y"]
        if abs(dx) >= abs(dy):
            relationships.append("left_of" if dx < 0 else "right_of")
        else:
            relationships.append("above" if dy < 0 else "below")
        return relationships

    def _compute_iou(self, bbox_a: dict, bbox_b: dict) -> float:
        inter_x1 = max(bbox_a["x1"], bbox_b["x1"])
        inter_y1 = max(bbox_a["y1"], bbox_b["y1"])
        inter_x2 = min(bbox_a["x2"], bbox_b["x2"])
        inter_y2 = min(bbox_a["y2"], bbox_b["y2"])
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (bbox_a["x2"] - bbox_a["x1"]) * (bbox_a["y2"] - bbox_a["y1"])
        area_b = (bbox_b["x2"] - bbox_b["x1"]) * (bbox_b["y2"] - bbox_b["y1"])
        union_area = area_a + area_b - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    def _euclidean_distance(self, center_a: dict, center_b: dict) -> float:
        dx = center_a["x"] - center_b["x"]
        dy = center_a["y"] - center_b["y"]
        return (dx ** 2 + dy ** 2) ** 0.5

    def _compute_diagonal(self, width: int, height: int) -> float:
        return (width ** 2 + height ** 2) ** 0.5

    def _get_inverse_relationship(self, relationship: str) -> Optional[str]:
        inverse_map = {"left_of": "right_of", "right_of": "left_of", "above": "below", "below": "above", "near": None, "overlapping": None}
        return inverse_map.get(relationship)

    def _edges_to_triplets(self, edges: list[dict]) -> list[dict]:
        return [{"subject": edge["subject_label"], "relationship": edge["relationship"], "object": edge["object_label"]} for edge in edges]

    def _build_networkx_graph(self, nodes: list[dict], edges: list[dict]) -> nx.DiGraph:
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node["id"], label=node["label"], confidence=node["confidence"], center=node["center"])
        for edge in edges:
            G.add_edge(edge["subject"], edge["object"], relationship=edge["relationship"])
        return G

    def _build_summary(self, nodes: list[dict], edges: list[dict], nx_graph: nx.DiGraph) -> dict:
        unique_labels = list({node["label"] for node in nodes})
        most_connected = None
        if nx_graph.nodes:
            degrees = dict(nx_graph.degree())
            most_connected = max(degrees, key=degrees.get) if degrees else None
        relationship_counts: dict[str, int] = {}
        for edge in edges:
            rel = edge["relationship"]
            relationship_counts[rel] = relationship_counts.get(rel, 0) + 1
        return {"total_objects": len(nodes), "unique_categories": unique_labels, "total_relationships": len(edges), "relationship_distribution": relationship_counts, "most_connected_object": most_connected, "scene_complexity": ("simple" if len(nodes) <= 3 else "moderate" if len(nodes) <= 7 else "complex")}

    def _empty_scene_graph(self) -> dict:
        return {"nodes": [], "edges": [], "triplets": [], "summary": {"total_objects": 0, "unique_categories": [], "total_relationships": 0, "relationship_distribution": {}, "most_connected_object": None, "scene_complexity": "empty"}}
