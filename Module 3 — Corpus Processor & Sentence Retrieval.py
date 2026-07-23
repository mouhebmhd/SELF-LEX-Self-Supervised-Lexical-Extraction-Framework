"""
modules/module3_corpus_retrieval.py
=====================================
Module 3: Corpus Processor & Sentence Retrieval

Purpose
-------
Index the corpus and retrieve, for each medical term `t`, all sentences
containing `t`.

Inputs
------
- Cleaned text corpus (Module 1).
- `V_med` list from Module 2.

Outputs
-------
- A dictionary `sentences_dict` mapping each term `t` to a list of
  sentences (strings) that contain `t`.
"""

import os
from collections import defaultdict
from typing import Dict, List

from utils.file_utils import load_text_lines, save_json
from utils.logger import get_logger
from utils.text_utils import contains_whole_token

logger = get_logger(__name__)

# Whoosh is optional; the module works with a pure-Python inverted index
# if Whoosh is not installed, at some cost to retrieval speed on very
# large corpora.
try:
    from whoosh import index as whoosh_index
    from whoosh.fields import ID, TEXT, Schema
    from whoosh.qparser import QueryParser

    _WHOOSH_AVAILABLE = True
except ImportError:
    _WHOOSH_AVAILABLE = False


def build_sentence_index(corpus_file: str, index_dir: str = None):
    """
    Build an inverted index (term -> sentence IDs) over the corpus.

    Uses Whoosh for a proper full-text index if available and
    `index_dir` is provided; otherwise builds a simple in-memory
    Python dict-based index keyed by lowercase token.

    Args:
        corpus_file: Path to the cleaned corpus (one sentence per line).
        index_dir: Directory to persist the Whoosh index (optional).

    Returns:
        Either a Whoosh `Index` object, or a tuple
        `(sentences: List[str], token_index: Dict[str, List[int]])`
        for the pure-Python fallback.
    """
    sentences = load_text_lines(corpus_file)
    logger.info(f"Loaded {len(sentences)} sentences from {corpus_file}")

    if _WHOOSH_AVAILABLE and index_dir:
        os.makedirs(index_dir, exist_ok=True)
        schema = Schema(id=ID(stored=True, unique=True), content=TEXT(stored=True))
        ix = whoosh_index.create_in(index_dir, schema)
        writer = ix.writer()
        for i, sent in enumerate(sentences):
            writer.add_document(id=str(i), content=sent)
        writer.commit()
        logger.info(f"Built Whoosh index with {len(sentences)} documents at {index_dir}")
        return ix

    # Pure-Python fallback: simple inverted index over lowercase tokens
    token_index: Dict[str, List[int]] = defaultdict(list)
    for i, sent in enumerate(sentences):
        for tok in set(sent.lower().split()):
            token_index[tok].append(i)
    logger.info("Built pure-Python fallback inverted index (Whoosh unavailable or index_dir not set)")
    return sentences, dict(token_index)


def retrieve_sentences(term: str, index, max_results: int = 50) -> List[str]:
    """
    Retrieve sentences containing `term` from the built index.

    Args:
        term: Target medical term (may be multi-word).
        index: Either a Whoosh `Index` object or the
               `(sentences, token_index)` fallback tuple returned by
               `build_sentence_index`.
        max_results: Maximum number of sentences to return.

    Returns:
        List of matching sentence strings.
    """
    if _WHOOSH_AVAILABLE and hasattr(index, "searcher"):
        results_out = []
        with index.searcher() as searcher:
            parser = QueryParser("content", index.schema)
            query = parser.parse(f'"{term}"')
            results = searcher.search(query, limit=max_results)
            results_out = [hit["content"] for hit in results]
        return results_out

    # Fallback path
    sentences, token_index = index
    first_word = term.lower().split()[0]
    candidate_ids = token_index.get(first_word, [])
    matches = [sentences[i] for i in candidate_ids]
    return matches[:max_results]


def filter_sentences(sentences: List[str], term: str, case_sensitive: bool = False) -> List[str]:
    """
    Ensure the term appears as a whole token in each sentence
    (avoids partial/substring matches, e.g. 'cardiac' inside 'pericardial').

    Args:
        sentences: Candidate sentences (e.g., from `retrieve_sentences`).
        term: Target term.
        case_sensitive: Whether matching should respect case.

    Returns:
        Filtered list of sentences where `term` is a genuine whole-token match.
    """
    return [s for s in sentences if contains_whole_token(s, term, case_sensitive=case_sensitive)]


def store_sentences(sentences_dict: Dict[str, List[str]], output_file: str) -> None:
    """Persist the term -> sentences mapping to disk as JSON."""
    save_json(sentences_dict, output_file)


def run_module3(config: dict, corpus_file: str, v_med: List[str]) -> Dict[str, List[str]]:
    """
    High-level orchestration entry point for Module 3.

    Args:
        config: Full pipeline configuration.
        corpus_file: Path to the cleaned corpus produced by Module 1.
        v_med: List of unique medical concepts from Module 2.

    Returns:
        `sentences_dict` mapping each term to its list of matching sentences.
    """
    cr_cfg = config["corpus_retrieval"]
    paths = config["paths"]

    index = build_sentence_index(corpus_file, index_dir=cr_cfg.get("index_dir"))

    sentences_dict: Dict[str, List[str]] = {}
    for term in v_med:
        raw_matches = retrieve_sentences(term, index, max_results=cr_cfg.get("max_sentences_per_term", 50))
        filtered = filter_sentences(raw_matches, term, case_sensitive=cr_cfg.get("case_sensitive", False))
        if filtered:
            sentences_dict[term] = filtered
        else:
            logger.debug(f"No sentences found for term '{term}' (will rely on Module 7 fallback)")

    output_path = os.path.join(paths["intermediate_dir"], "sentences_dict.json")
    store_sentences(sentences_dict, output_path)

    logger.info(f"Module 3 complete: sentences found for {len(sentences_dict)}/{len(v_med)} terms")
    return sentences_dict


if __name__ == "__main__":
    import yaml

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    demo_corpus = os.path.join(cfg["paths"]["intermediate_dir"], "corpus_clean.txt")
    demo_vmed = ["hypertension", "myocardial infarction"]
    run_module3(cfg, demo_corpus, demo_vmed)