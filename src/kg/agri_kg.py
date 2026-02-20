"""
agri_kg.py
----------
Lightweight Agricultural Knowledge Graph using NetworkX.

Node types:
  - CROP      : Agricultural crops
  - DISEASE   : Plant diseases
  - SYMPTOM   : Visible symptoms
  - TREATMENT : Recommended treatments/chemicals
  - WEATHER   : Weather conditions that trigger disease
  - PEST      : Common agricultural pests

Edge types:
  - HAS_SYMPTOM   : CROP → SYMPTOM (or DISEASE → SYMPTOM)
  - CAUSES        : WEATHER/PEST → DISEASE
  - TREATED_BY    : DISEASE → TREATMENT
  - AFFECTS       : DISEASE/PEST → CROP
  - WORSENED_BY   : DISEASE → WEATHER condition

Persistence: Saved to JSON for lightweight deployment.
Optional: Neo4j Community for larger deployments.
"""

import os
import json
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from loguru import logger

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    logger.warning("networkx not installed. KG disabled.")

# ── Data Classes ─────────────────────────────────────────────
@dataclass
class KGNode:
    """A knowledge graph node."""
    id: str
    type: str          # CROP / DISEASE / SYMPTOM / TREATMENT / WEATHER / PEST
    name: str
    aliases: List[str] = field(default_factory=list)
    properties: Dict = field(default_factory=dict)


@dataclass
class KGEdge:
    """A knowledge graph edge (triple)."""
    source: str
    relation: str
    target: str
    confidence: float = 1.0
    properties: Dict = field(default_factory=dict)


# ── Pre-built Agricultural Knowledge Base ────────────────────
INITIAL_KNOWLEDGE = {
    "nodes": [
        # CROPS
        KGNode("rice", "CROP", "Rice", ["paddy", "Oryza sativa"]),
        KGNode("wheat", "CROP", "Wheat", ["gehun", "Triticum"]),
        KGNode("cotton", "CROP", "Cotton", ["kapas", "Gossypium"]),
        KGNode("tomato", "CROP", "Tomato", ["tamatar"]),
        KGNode("potato", "CROP", "Potato", ["aloo"]),
        KGNode("maize", "CROP", "Maize", ["corn", "makka"]),
        KGNode("sugarcane", "CROP", "Sugarcane", ["ganna", "ikku"]),

        # DISEASES
        KGNode("blast", "DISEASE", "Rice Blast", ["neck blast", "leaf blast"],
               {"pathogen": "Magnaporthe oryzae", "type": "fungal"}),
        KGNode("blight_bact", "DISEASE", "Bacterial Leaf Blight", ["BLB", "kresek"],
               {"pathogen": "Xanthomonas oryzae", "type": "bacterial"}),
        KGNode("brown_spot", "DISEASE", "Brown Spot",
               properties={"pathogen": "Cochliobolus miyabeanus", "type": "fungal"}),
        KGNode("sheath_blight", "DISEASE", "Sheath Blight",
               properties={"pathogen": "Rhizoctonia solani", "type": "fungal"}),
        KGNode("late_blight", "DISEASE", "Late Blight",
               properties={"pathogen": "Phytophthora infestans", "type": "oomycete"}),
        KGNode("early_blight", "DISEASE", "Early Blight",
               properties={"pathogen": "Alternaria solani", "type": "fungal"}),
        KGNode("wheat_rust", "DISEASE", "Wheat Rust", ["yellow rust", "stripe rust"],
               {"pathogen": "Puccinia striiformis", "type": "fungal"}),
        KGNode("cotton_wilt", "DISEASE", "Fusarium Wilt",
               properties={"pathogen": "Fusarium oxysporum", "type": "fungal"}),
        KGNode("cotton_leaf_curl", "DISEASE", "Cotton Leaf Curl Virus", ["CLCuV"],
               properties={"pathogen": "Begomovirus", "type": "viral"}),

        # SYMPTOMS
        KGNode("yellow_spots", "SYMPTOM", "Yellow spots on leaves"),
        KGNode("brown_spots", "SYMPTOM", "Brown spots on leaves"),
        KGNode("wilting", "SYMPTOM", "Wilting of plant"),
        KGNode("leaf_curl", "SYMPTOM", "Leaf curling"),
        KGNode("white_lesions", "SYMPTOM", "White water-soaked lesions"),
        KGNode("necrosis", "SYMPTOM", "Leaf necrosis / burning"),
        KGNode("stunted_growth", "SYMPTOM", "Stunted growth"),
        KGNode("streaks", "SYMPTOM", "Yellow/brown streaks on leaf"),

        # TREATMENTS
        KGNode("carbendazim", "TREATMENT", "Carbendazim 50% WP",
               properties={"type": "fungicide", "dose": "1g per liter of water"}),
        KGNode("tricyclazole", "TREATMENT", "Tricyclazole 75% WP",
               properties={"type": "fungicide", "dose": "0.6g per liter", "for": "blast"}),
        KGNode("copper_oxyc", "TREATMENT", "Copper Oxychloride 50% WP",
               properties={"type": "fungicide", "dose": "2.5g per liter"}),
        KGNode("streptomycin", "TREATMENT", "Streptomycin + Tetracycline",
               properties={"type": "antibiotic", "for": "bacterial diseases"}),
        KGNode("mancozeb", "TREATMENT", "Mancozeb 75% WP",
               properties={"type": "fungicide", "dose": "2g per liter"}),
        KGNode("neem_oil", "TREATMENT", "Neem Oil 1%",
               properties={"type": "organic", "dose": "3ml per liter"}),
        KGNode("urea", "TREATMENT", "Urea", properties={"type": "fertilizer", "N": "46%"}),
        KGNode("dap", "TREATMENT", "Diammonium Phosphate (DAP)",
               properties={"type": "fertilizer", "N": "18%", "P": "46%"}),

        # WEATHER CONDITIONS
        KGNode("high_humidity", "WEATHER", "High humidity (>80%)"),
        KGNode("warm_wet", "WEATHER", "Warm and wet conditions (20-30°C, high rain)"),
        KGNode("cool_moist", "WEATHER", "Cool and moist conditions"),
        KGNode("hot_dry", "WEATHER", "Hot and dry conditions"),

        # PESTS
        KGNode("stem_borer", "PEST", "Rice Stem Borer", ["Scirpophaga incertulas"]),
        KGNode("aphid", "PEST", "Aphid", ["plant lice"]),
        KGNode("whitefly", "PEST", "Whitefly"),
        KGNode("brown_planthopper", "PEST", "Brown Planthopper", ["BPH"]),
    ],
    "edges": [
        # Blast
        KGEdge("blast", "AFFECTS", "rice"),
        KGEdge("blast", "HAS_SYMPTOM", "yellow_spots"),
        KGEdge("blast", "HAS_SYMPTOM", "white_lesions"),
        KGEdge("blast", "HAS_SYMPTOM", "brown_spots"),
        KGEdge("blast", "TREATED_BY", "tricyclazole"),
        KGEdge("blast", "TREATED_BY", "carbendazim"),
        KGEdge("high_humidity", "CAUSES", "blast"),
        KGEdge("warm_wet", "CAUSES", "blast"),

        # Bacterial Leaf Blight
        KGEdge("blight_bact", "AFFECTS", "rice"),
        KGEdge("blight_bact", "HAS_SYMPTOM", "white_lesions"),
        KGEdge("blight_bact", "HAS_SYMPTOM", "wilting"),
        KGEdge("blight_bact", "HAS_SYMPTOM", "streaks"),
        KGEdge("blight_bact", "TREATED_BY", "streptomycin"),
        KGEdge("blight_bact", "TREATED_BY", "copper_oxyc"),

        # Brown Spot
        KGEdge("brown_spot", "AFFECTS", "rice"),
        KGEdge("brown_spot", "HAS_SYMPTOM", "brown_spots"),
        KGEdge("brown_spot", "TREATED_BY", "mancozeb"),
        KGEdge("brown_spot", "TREATED_BY", "carbendazim"),

        # Sheath Blight
        KGEdge("sheath_blight", "AFFECTS", "rice"),
        KGEdge("sheath_blight", "HAS_SYMPTOM", "brown_spots"),
        KGEdge("sheath_blight", "HAS_SYMPTOM", "wilting"),
        KGEdge("sheath_blight", "TREATED_BY", "carbendazim"),
        KGEdge("high_humidity", "CAUSES", "sheath_blight"),

        # Late Blight (Potato/Tomato)
        KGEdge("late_blight", "AFFECTS", "potato"),
        KGEdge("late_blight", "AFFECTS", "tomato"),
        KGEdge("late_blight", "HAS_SYMPTOM", "brown_spots"),
        KGEdge("late_blight", "HAS_SYMPTOM", "necrosis"),
        KGEdge("late_blight", "TREATED_BY", "mancozeb"),
        KGEdge("late_blight", "TREATED_BY", "copper_oxyc"),
        KGEdge("cool_moist", "CAUSES", "late_blight"),

        # Early Blight
        KGEdge("early_blight", "AFFECTS", "tomato"),
        KGEdge("early_blight", "AFFECTS", "potato"),
        KGEdge("early_blight", "HAS_SYMPTOM", "brown_spots"),
        KGEdge("early_blight", "TREATED_BY", "mancozeb"),

        # Wheat Rust
        KGEdge("wheat_rust", "AFFECTS", "wheat"),
        KGEdge("wheat_rust", "HAS_SYMPTOM", "yellow_spots"),
        KGEdge("wheat_rust", "HAS_SYMPTOM", "streaks"),
        KGEdge("wheat_rust", "TREATED_BY", "mancozeb"),
        KGEdge("wheat_rust", "TREATED_BY", "carbendazim"),

        # Cotton Wilt
        KGEdge("cotton_wilt", "AFFECTS", "cotton"),
        KGEdge("cotton_wilt", "HAS_SYMPTOM", "wilting"),
        KGEdge("cotton_wilt", "HAS_SYMPTOM", "stunted_growth"),

        # Cotton Leaf Curl
        KGEdge("cotton_leaf_curl", "AFFECTS", "cotton"),
        KGEdge("cotton_leaf_curl", "HAS_SYMPTOM", "leaf_curl"),
        KGEdge("cotton_leaf_curl", "HAS_SYMPTOM", "stunted_growth"),
        KGEdge("cotton_leaf_curl", "TREATED_BY", "neem_oil"),  # To control vector

        # Pests
        KGEdge("stem_borer", "AFFECTS", "rice"),
        KGEdge("aphid", "AFFECTS", "cotton"),
        KGEdge("brown_planthopper", "AFFECTS", "rice"),
        KGEdge("whitefly", "AFFECTS", "tomato"),

        # Rice crop relationships
        KGEdge("rice", "HAS_SYMPTOM", "yellow_spots"),
        KGEdge("rice", "HAS_SYMPTOM", "brown_spots"),
    ]
}


class AgriculturalKG:
    """
    Lightweight Agricultural Knowledge Graph using NetworkX.
    """

    def __init__(self, kg_path: Optional[str] = None):
        if not _NX_AVAILABLE:
            self.G = None
            logger.error("NetworkX not available. KG disabled.")
            return

        self.G = nx.DiGraph()  # Directed graph
        self._node_index: Dict[str, KGNode] = {}  # id → KGNode
        self._alias_index: Dict[str, str] = {}    # alias → node id

        # Load or build
        if kg_path and os.path.exists(kg_path):
            self.load(kg_path)
        else:
            self._build_initial_kg()
            logger.info(f"KG built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def _build_initial_kg(self):
        """Build KG from initial knowledge base."""
        # Add nodes
        for node in INITIAL_KNOWLEDGE["nodes"]:
            self.add_node(node)
        # Add edges
        for edge in INITIAL_KNOWLEDGE["edges"]:
            self.add_edge(edge)

    def add_node(self, node: KGNode):
        """Add a node to the knowledge graph."""
        # Build attributes dict carefully to avoid duplicate key errors
        attrs = {
            "node_type": node.type,
            "name": node.name,
            "aliases": node.aliases,
            "pathogen": node.properties.get("pathogen"),
            "chemical_type": node.properties.get("type"),
            "dose": node.properties.get("dose"),
            "for_disease": node.properties.get("for"),
            "N_pct": node.properties.get("N"),
            "P_pct": node.properties.get("P"),
        }
        self.G.add_node(node.id, **attrs)
        self._node_index[node.id] = node
        # Index aliases
        self._alias_index[node.name.lower()] = node.id
        for alias in node.aliases:
            self._alias_index[alias.lower()] = node.id

    def add_edge(self, edge: KGEdge):
        """Add an edge (triple) to the knowledge graph."""
        if edge.source not in self.G or edge.target not in self.G:
            logger.warning(f"Edge {edge.source}→{edge.target} skipped (node not found)")
            return
        self.G.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation,
            confidence=edge.confidence,
            **edge.properties
        )

    def resolve_entity(self, name: str, expected_type: Optional[str] = None) -> Optional[str]:
        """
        Resolve entity name/alias to node ID.
        If expected_type is provided, only returns a node of that type.
        """
        name_lower = name.lower().strip()
        
        def _check(nid):
            if not expected_type: return True
            return self.G.nodes[nid].get("node_type") == expected_type

        # Direct match
        if name_lower in self.G and _check(name_lower):
            return name_lower
        # Alias exact match
        if name_lower in self._alias_index:
            nid = self._alias_index[name_lower]
            if _check(nid): return nid
            
        # Partial match
        for alias, node_id in self._alias_index.items():
            if (name_lower in alias or alias in name_lower) and _check(node_id):
                return node_id
        return None

    def get_diseases_for_crop(self, crop: str) -> List[dict]:
        """Get all diseases that affect a specific crop."""
        crop_id = self.resolve_entity(crop)
        if not crop_id:
            return []

        diseases = []
        for source, target, data in self.G.in_edges(crop_id, data=True):
            if data.get("relation") == "AFFECTS":
                node = self._node_index.get(source)
                if node and node.type == "DISEASE":
                    diseases.append({
                        "id": source,
                        "name": self.G.nodes[source].get("name", source),
                        "pathogen": self.G.nodes[source].get("pathogen"),
                        "type": self.G.nodes[source].get("node_type"),
                    })
        return diseases

    def get_treatments_for_disease(self, disease: str) -> List[dict]:
        """Get treatments for a disease."""
        disease_id = self.resolve_entity(disease)
        if not disease_id:
            return []

        treatments = []
        for source, target, data in self.G.out_edges(disease_id, data=True):
            if data.get("relation") == "TREATED_BY":
                node = self._node_index.get(target)
                if node:
                    treatments.append({
                        "id": target,
                        "name": self.G.nodes[target].get("name", target),
                        "dose": self.G.nodes[target].get("dose"),
                        "type": self.G.nodes[target].get("node_type"),
                    })
        return treatments

    def get_symptoms_for_disease(self, disease: str) -> List[str]:
        """Get symptoms associated with a disease."""
        disease_id = self.resolve_entity(disease)
        if not disease_id:
            return []

        symptoms = []
        for source, target, data in self.G.out_edges(disease_id, data=True):
            if data.get("relation") == "HAS_SYMPTOM":
                symptoms.append(self.G.nodes[target].get("name", target))
        return symptoms

    def get_diseases_by_symptom(self, symptom: str, crop: str = None) -> List[dict]:
        """
        Given a symptom, find likely diseases.
        If crop is also provided, filter by crop.
        """
        symptom_id = self.resolve_entity(symptom, expected_type="SYMPTOM")

        diseases = []
        if symptom_id:
            for source, _, data in self.G.in_edges(symptom_id, data=True):
                if data.get("relation") == "HAS_SYMPTOM":
                    node_data = self.G.nodes.get(source, {})
                    if node_data.get("node_type") == "DISEASE":
                        disease_info = {
                            "id": source,
                            "name": node_data.get("name", source),
                            "pathogen": node_data.get("pathogen"),
                        }
                        if crop:
                            # Check if this disease affects the given crop
                            crop_id = self.resolve_entity(crop)
                            if crop_id and self.G.has_edge(source, crop_id):
                                diseases.append(disease_info)
                        else:
                            diseases.append(disease_info)
        return diseases

    def query(self, crop: str = None, symptom: str = None, disease: str = None) -> dict:
        """
        Main query interface for the knowledge graph.

        Returns relevant triples for the query.
        """
        if not _NX_AVAILABLE or self.G is None:
            return {"error": "KG not available"}

        result = {
            "crop": crop,
            "symptom": symptom,
            "disease": disease,
            "diseases_found": [],
            "treatments": [],
            "symptoms_of_disease": [],
            "triples": [],
        }

        # Infer disease from symptoms + crop
        if symptom and not disease:
            result["diseases_found"] = self.get_diseases_by_symptom(symptom, crop)

        # If disease is known, get treatments
        if disease:
            result["treatments"] = self.get_treatments_for_disease(disease)
            result["symptoms_of_disease"] = self.get_symptoms_for_disease(disease)

        # If only crop is given, list all diseases
        if crop and not disease and not symptom:
            result["diseases_found"] = self.get_diseases_for_crop(crop)

        # Build triple summary
        for d in result["diseases_found"]:
            result["triples"].append(f"{d['name']} AFFECTS {crop or 'unknown crop'}")
            for t in self.get_treatments_for_disease(d["id"]):
                result["triples"].append(f"{d['name']} TREATED_BY {t['name']}")

        return result

    def save(self, path: str):
        """Save KG to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = nx.node_link_data(self.G)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"KG saved to {path}")

    def load(self, path: str):
        """Load KG from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.G = nx.node_link_graph(data, directed=True)
        logger.info(f"KG loaded from {path}: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")


# Singleton
_kg_instance: Optional[AgriculturalKG] = None

def get_kg(kg_path: Optional[str] = None) -> AgriculturalKG:
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = AgriculturalKG(kg_path=kg_path)
    return _kg_instance


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Agricultural Knowledge Graph Test")
    print("=" * 60)

    kg = AgriculturalKG()
    print(f"Nodes: {kg.G.number_of_nodes()}, Edges: {kg.G.number_of_edges()}")
    print()

    # Test 1: Diseases for rice
    print("Diseases affecting Rice:")
    for d in kg.get_diseases_for_crop("rice"):
        print(f"  - {d['name']} ({d.get('type', 'unknown')})")

    # Test 2: Treatment for blast
    print("\nTreatments for Rice Blast:")
    for t in kg.get_treatments_for_disease("blast"):
        print(f"  - {t['name']} (dose: {t.get('dose', 'N/A')})")

    # Test 3: Disease from symptom
    print("\nDiseases with 'yellow spots' on rice:")
    for d in kg.get_diseases_by_symptom("yellow spots", "rice"):
        print(f"  - {d['name']}")

    # Test 4: Full query
    print("\nFull Query (crop=rice, symptom=yellow spots):")
    result = kg.query(crop="rice", symptom="yellow spots")
    print(f"  Diseases: {[d['name'] for d in result['diseases_found']]}")
    print(f"  Triples : {result['triples'][:3]}")
