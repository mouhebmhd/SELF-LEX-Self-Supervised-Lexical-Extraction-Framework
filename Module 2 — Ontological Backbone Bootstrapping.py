"""
modules/module2_ontology.py
=============================
Module 2: Ontological Backbone Bootstrapping (Seed Graph)

Purpose
-------
Parse ontological sources to build a directed graph of medical concepts
and their relationships.

Inputs
------
- Ontology files from Module 1.

Outputs
-------
- Graph `G_seed` as a `networkx.DiGraph` with nodes (concept IDs/terms)
  and edges (SYNONYM, BROADER_THAN, NARROWER_THAN).
- A list `V_med` of all unique medical concepts.
"""

import csv
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

import networkx as nx

from utils.file_utils import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)


def load_ontology(file_path: str) -> List[dict]:
    """
    Parse an ontology file (RRF/CSV/XML) into a list of raw records.

    The parser auto-detects format from the file extension. Each record
    is a dict with at minimum a 'concept' key, and optionally
    'parent' / 'relation' keys describing hierarchical relationships.

    Args:
        file_path: Path to the ontology file.

    Returns:
        List of record dicts. Returns an empty list if the file is
        missing, empty, or unparsable (logged as a warning, not fatal,
        since Module 7 fallback relies on best-effort graph coverage).
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        logger.warning(f"Ontology file missing or empty: {file_path}")
        return []

    ext = os.path.splitext(file_path)[1].lower()
    records: List[dict] = []

    try:
        if ext == ".csv":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)

        elif ext == ".xml":
            tree = ET.parse(file_path)
            root = tree.getroot()
            for elem in root.iter():
                # Generic XML walk: treat any element with a 'concept'-like
                # child/attribute as a record. Real UMLS/MeSH XML schemas
                # vary, so this should be adapted per source in practice.
                concept = elem.attrib.get("concept") or (elem.text.strip() if elem.text else None)
                if concept:
                    records.append(
                        {
                            "concept": concept,
                            "parent": elem.attrib.get("parent"),
                            "relation": elem.attrib.get("relation", "BROADER_THAN"),
                        }
                    )

        elif ext in (".rrf", ".dat", ".txt"):
            # UMLS RRF is pipe-delimited; treat generically.
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 2 and parts[0]:
                        records.append({"concept": parts[0], "parent": parts[1] if len(parts) > 1 else None,
                                         "relation": "BROADER_THAN"})
        else:
            logger.warning(f"Unrecognized ontology format for {file_path}, skipping.")

    except Exception as e:
        logger.error(f"Failed to parse ontology file {file_path}: {e}")

    logger.info(f"Parsed {len(records)} raw records from {file_path}")
    return records


def extract_concepts(ontology: List[dict]) -> List[str]:
    """
    Extract the unique list of concept strings from parsed ontology records.

    Args:
        ontology: List of record dicts from `load_ontology`.

    Returns:
        Sorted list of unique concept strings.
    """
    concepts = {rec["concept"].strip().lower() for rec in ontology if rec.get("concept")}
    return sorted(concepts)


def extract_relationships(ontology: List[dict]) -> List[Tuple[str, str, str]]:
    """
    Extract (parent, child, relation_type) triples from parsed records.

    Args:
        ontology: List of record dicts from `load_ontology`.

    Returns:
        List of (parent, child, relation_type) tuples. Records without
        a valid parent are skipped (they become isolated nodes instead).
    """
    relations = []
    for rec in ontology:
        child = rec.get("concept")
        parent = rec.get("parent")
        relation = rec.get("relation") or "BROADER_THAN"
        if child and parent:
            relations.append((parent.strip().lower(), child.strip().lower(), relation))
    return relations


def build_seed_graph(concepts: List[str], relations: List[Tuple[str, str, str]]) -> nx.DiGraph:
    """
    Construct a directed graph from concepts and relationship triples.

    Edge direction convention: parent -> child for BROADER_THAN /
    NARROWER_THAN edges; SYNONYM edges are added bidirectionally.

    Args:
        concepts: List of unique concept strings (graph nodes).
        relations: List of (parent, child, relation_type) triples.

    Returns:
        A populated `networkx.DiGraph`.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(concepts)

    for parent, child, relation in relations:
        graph.add_node(parent)
        graph.add_node(child)
        if relation == "SYNONYM":
            graph.add_edge(parent, child, relation=relation)
            graph.add_edge(child, parent, relation=relation)
        elif relation == "NARROWER_THAN":
            # child is broader than parent in this record's semantics -> reverse
            graph.add_edge(child, parent, relation="BROADER_THAN")
        else:  # default / BROADER_THAN: parent -> child means parent is broader
            graph.add_edge(parent, child, relation="BROADER_THAN")

    logger.info(f"Built seed graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


def save_graph(graph: nx.DiGraph, output_path: str) -> None:
    """
    Persist the seed graph to disk via pickle (gpickle-equivalent).

    Args:
        graph: The `networkx.DiGraph` to save.
        output_path: Destination path (e.g., 'seed_graph.gpickle').
    """
    ensure_dir(output_path)
    import pickle

    with open(output_path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved seed graph to {output_path}")


def load_graph(input_path: str) -> nx.DiGraph:
    """Load a previously saved seed graph from disk."""
    import pickle

    with open(input_path, "rb") as f:
        graph = pickle.load(f)
    logger.debug(f"Loaded seed graph from {input_path}")
    return graph


def run_module2(config: dict) -> Tuple[nx.DiGraph, List[str]]:
    """
    High-level orchestration entry point for Module 2.

    Args:
        config: Full pipeline configuration (parsed config.yaml).

    Returns:
        Tuple of (seed_graph, V_med) where V_med is the list of unique
        medical concept strings.
    """
    paths = config["paths"]
    ont_cfg = config["ontology"]
    ontology_dir = paths["ontology_dir"]

    all_records = []
    if os.path.isdir(ontology_dir):
        for fname in os.listdir(ontology_dir):
            fpath = os.path.join(ontology_dir, fname)
            all_records.extend(load_ontology(fpath))

    concepts = extract_concepts(all_records)
    relations = extract_relationships(all_records)

    if not concepts:
        logger.warning(
            "No concepts extracted from ontology sources. "
            "Downstream modules will operate on an empty V_med list "
            "unless synthetic/test data is supplied."
        )

    seed_graph = build_seed_graph(concepts, relations)

    output_path = os.path.join(paths["intermediate_dir"], ont_cfg["seed_graph_file"])
    save_graph(seed_graph, output_path)

    logger.info(f"Module 2 complete: {len(concepts)} concepts, graph saved to {output_path}")
    return seed_graph, concepts


if __name__ == "__main__":
    import yaml

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_module2(cfg)