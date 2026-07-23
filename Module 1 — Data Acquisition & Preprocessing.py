"""
modules/module1_acquisition.py
================================
Module 1: Data Acquisition & Preprocessing

Purpose
-------
Download and clean required ontologies and patient-facing corpora.

Inputs
------
- URLs or local paths for UMLS/MeSH/ICD-10 (open-access subsets).
- List of corpus sources (MedlinePlus, Wikipedia dumps, Reddit posts).

Outputs
-------
- Raw ontology files stored locally.
- Cleaned text corpus (plain text, sentence-split).
"""

import os
from typing import Dict, List

import requests

from utils.file_utils import ensure_dir, save_text_lines
from utils.logger import get_logger
from utils.text_utils import clean_text, split_sentences

logger = get_logger(__name__)


def download_ontology(source: Dict[str, str], target_dir: str) -> str:
    """
    Download an ontology file from `source['url']` into `target_dir`.

    Args:
        source: Dict with keys 'name', 'url', 'format'.
        target_dir: Local directory to store the file.

    Returns:
        Path to the downloaded file. If the download fails (e.g. no
        network access, placeholder URL), a warning is logged and an
        empty placeholder file is created so downstream modules can
        still run against synthetic/test data.
    """
    ensure_dir(target_dir)
    filename = f"{source['name']}.{source.get('format', 'dat')}"
    target_path = os.path.join(target_dir, filename)

    try:
        logger.info(f"Downloading ontology '{source['name']}' from {source['url']}")
        response = requests.get(source["url"], timeout=30)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Saved ontology to {target_path}")
    except Exception as e:
        logger.warning(
            f"Could not download '{source['name']}' ({e}). "
            f"Creating empty placeholder at {target_path}."
        )
        open(target_path, "a").close()

    return target_path


def download_corpus(source: Dict[str, str], target_dir: str) -> str:
    """
    Download a raw corpus text source into `target_dir`.

    Args:
        source: Dict with keys 'name', 'url'.
        target_dir: Local directory to store the file.

    Returns:
        Path to the downloaded raw text file.
    """
    ensure_dir(target_dir)
    target_path = os.path.join(target_dir, f"{source['name']}.txt")

    try:
        logger.info(f"Downloading corpus '{source['name']}' from {source['url']}")
        response = requests.get(source["url"], timeout=30)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Saved corpus to {target_path}")
    except Exception as e:
        logger.warning(
            f"Could not download corpus '{source['name']}' ({e}). "
            f"Creating empty placeholder at {target_path}."
        )
        open(target_path, "a").close()

    return target_path


def preprocess_text(raw_text: str, min_len: int = 4, max_len: int = 80) -> List[str]:
    """
    Clean raw text and split it into sentences within a length range.

    Args:
        raw_text: Unprocessed text blob.
        min_len: Minimum sentence length (in tokens) to keep.
        max_len: Maximum sentence length (in tokens) to keep.

    Returns:
        List of cleaned sentence strings.
    """
    sentences = split_sentences(raw_text)
    cleaned = []
    for sent in sentences:
        c = clean_text(sent, lowercase=False, strip_punct=False)
        n_tokens = len(c.split())
        if min_len <= n_tokens <= max_len:
            cleaned.append(c)
    logger.debug(f"Preprocessed {len(sentences)} raw sentences -> {len(cleaned)} kept")
    return cleaned


def save_corpus(corpus_list: List[str], output_file: str) -> None:
    """
    Persist the cleaned sentence corpus to disk, one sentence per line.

    Args:
        corpus_list: List of cleaned sentences.
        output_file: Destination path.
    """
    save_text_lines(corpus_list, output_file)


def run_module1(config: dict) -> str:
    """
    High-level orchestration entry point for Module 1.

    Args:
        config: Full pipeline configuration (parsed config.yaml).

    Returns:
        Path to the final cleaned, concatenated corpus file.
    """
    paths = config["paths"]
    acq_cfg = config["acquisition"]

    ontology_dir = paths["ontology_dir"]
    corpus_raw_dir = os.path.join(paths["corpus_dir"], "raw")
    corpus_clean_path = os.path.join(paths["intermediate_dir"], "corpus_clean.txt")

    # 1. Download ontologies
    for source in acq_cfg["ontology_sources"]:
        download_ontology(source, ontology_dir)

    # 2. Download corpora
    raw_corpus_paths = []
    for source in acq_cfg["corpus_sources"]:
        raw_corpus_paths.append(download_corpus(source, corpus_raw_dir))

    # 3. Preprocess & merge
    all_sentences: List[str] = []
    for path in raw_corpus_paths:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            logger.warning(f"Skipping empty/missing corpus file: {path}")
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        all_sentences.extend(
            preprocess_text(
                raw_text,
                min_len=acq_cfg.get("min_sentence_length", 4),
                max_len=acq_cfg.get("max_sentence_length", 80),
            )
        )

    # 4. Save merged corpus
    save_corpus(all_sentences, corpus_clean_path)
    logger.info(f"Module 1 complete: {len(all_sentences)} sentences -> {corpus_clean_path}")
    return corpus_clean_path


if __name__ == "__main__":
    import yaml

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_module1(cfg)