"""
modules/module4_mlm_harvest.py
=================================
Module 4: MLM-based Candidate Harvesting

Purpose
-------
For each term `t`, use a Masked Language Model (BioBERT/PubMedBERT) to
generate candidate layman substitutions from the retrieved sentences.

Inputs
------
- `sentences_dict` from Module 3.
- Pre-trained MLM (e.g., microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext).

Outputs
-------
- Dictionary `candidates_raw` mapping `t` -> list of (candidate, score) pairs.
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple

from utils.file_utils import save_json
from utils.logger import get_logger
from utils.text_utils import mask_term_in_sentence

logger = get_logger(__name__)


def load_mlm(model_name: str, device: str = "cuda"):
    """
    Load a HuggingFace masked-language-model tokenizer + model pair.

    Args:
        model_name: HuggingFace model identifier.
        device: Preferred device ('cuda' or 'cpu'). Falls back to CPU
                automatically if CUDA is unavailable.

    Returns:
        Tuple of (tokenizer, model, resolved_device_string).
    """
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    resolved_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    if device == "cuda" and resolved_device == "cpu":
        logger.warning("CUDA requested but not available; falling back to CPU.")

    logger.info(f"Loading MLM '{model_name}' on {resolved_device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()

    return tokenizer, model, resolved_device


def mask_term(sentence: str, term: str, mask_token: str = "[MASK]") -> str:
    """
    Replace the target term in `sentence` with the model's mask token.

    Args:
        sentence: Original sentence containing `term`.
        term: Medical term to mask.
        mask_token: Model-specific mask token string.

    Returns:
        The masked sentence.
    """
    return mask_term_in_sentence(sentence, term, mask_token=mask_token)


def get_predictions(masked_sentence: str, model, tokenizer, k: int = 10, device: str = "cpu") -> List[Tuple[str, float]]:
    """
    Run the MLM forward pass and extract the top-k predicted tokens for
    the masked position, along with their softmax probabilities.

    Args:
        masked_sentence: Sentence containing the tokenizer's mask token.
        model: Loaded HuggingFace MLM model.
        tokenizer: Corresponding tokenizer.
        k: Number of top predictions to return.
        device: Device the model is on.

    Returns:
        List of (token_string, probability) tuples, sorted descending
        by probability. Returns an empty list if the mask token is not
        present in the input.
    """
    import torch

    if tokenizer.mask_token not in masked_sentence:
        # Sentence was masked with a placeholder that doesn't match
        # this specific tokenizer's mask token; try to fix it.
        masked_sentence = masked_sentence.replace("[MASK]", tokenizer.mask_token)

    inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    if len(mask_positions) == 0:
        logger.debug(f"No mask token found in: {masked_sentence}")
        return []

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, mask_positions[0]]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k)

    predictions = []
    for prob, tok_id in zip(top_probs.tolist(), top_ids.tolist()):
        token_str = tokenizer.decode([tok_id]).strip()
        if token_str and token_str.isalpha():
            predictions.append((token_str, prob))

    return predictions


def process_term(term: str, sentences: List[str], model, tokenizer, k: int, device: str) -> List[Tuple[str, float]]:
    """
    Run masking + prediction for every sentence containing `term`,
    then aggregate the resulting candidates.

    Args:
        term: Medical term being processed.
        sentences: Sentences containing `term` (from Module 3).
        model: Loaded MLM model.
        tokenizer: Corresponding tokenizer.
        k: Top-k predictions per sentence.
        device: Device string.

    Returns:
        Aggregated, normalized list of (candidate, score) tuples.
    """
    all_predictions: List[Tuple[str, float]] = []
    for sentence in sentences:
        masked = mask_term(sentence, term, mask_token=tokenizer.mask_token)
        if tokenizer.mask_token not in masked:
            continue
        preds = get_predictions(masked, model, tokenizer, k=k, device=device)
        all_predictions.extend(preds)

    return aggregate_candidates(all_predictions)


def aggregate_candidates(candidates_list: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Sum scores across duplicate candidates, then normalize so scores
    sum to 1.0 across the final candidate set.

    Args:
        candidates_list: Raw (candidate, score) tuples, possibly with
                          duplicate candidate strings.

    Returns:
        Deduplicated, normalized, descending-sorted list of
        (candidate, score) tuples.
    """
    score_sums: Dict[str, float] = defaultdict(float)
    for candidate, score in candidates_list:
        score_sums[candidate.lower()] += score

    total = sum(score_sums.values())
    if total == 0:
        return []

    normalized = [(cand, score / total) for cand, score in score_sums.items()]
    normalized.sort(key=lambda x: x[1], reverse=True)
    return normalized


def run_module4(config: dict, sentences_dict: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, float]]]:
    """
    High-level orchestration entry point for Module 4.

    Args:
        config: Full pipeline configuration.
        sentences_dict: term -> sentences mapping from Module 3.

    Returns:
        `candidates_raw` mapping each term to its aggregated candidate list.
    """
    mlm_cfg = config["mlm_harvest"]
    paths = config["paths"]

    tokenizer, model, device = load_mlm(mlm_cfg["model_name"], device=mlm_cfg.get("device", "cpu"))

    candidates_raw: Dict[str, List[Tuple[str, float]]] = {}
    for term, sentences in sentences_dict.items():
        candidates = process_term(
            term, sentences, model, tokenizer, k=mlm_cfg.get("top_k", 10), device=device
        )
        if candidates:
            candidates_raw[term] = candidates
        else:
            logger.debug(f"No MLM candidates generated for term '{term}'")

    output_path = os.path.join(paths["intermediate_dir"], "candidates_raw.json")
    # Convert tuples to lists for JSON serialization
    serializable = {t: [[c, s] for c, s in cands] for t, cands in candidates_raw.items()}
    save_json(serializable, output_path)

    logger.info(f"Module 4 complete: candidates generated for {len(candidates_raw)}/{len(sentences_dict)} terms")
    return candidates_raw


if __name__ == "__main__":
    import yaml

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    demo_sentences = {"hypertension": ["The patient was diagnosed with hypertension last year."]}
    run_module4(cfg, demo_sentences)