"""
modules/module5_llm_consensus.py
===================================
Module 5: Multi-LLM Consensus Filtering

Purpose
-------
Simulate an expert panel by prompting three distinct LLMs and retaining
only candidates that achieve semantic consensus.

Inputs
------
- `candidates_raw` from Module 4.
- Three pre-trained LLMs (e.g., Mistral-7B, Llama-3-8B, Phi-3-mini),
  4-bit quantized.
- Prompt template, similarity threshold `theta`, Sentence-BERT model.

Outputs
-------
- Dictionary `candidates_consensus` mapping `t` -> list of high-consensus
  candidate phrases.
"""

import os
from typing import Dict, List, Tuple

import numpy as np

from utils.file_utils import save_json
from utils.logger import get_logger

logger = get_logger(__name__)


def load_llm(model_name: str, quantize: bool = True):
    """
    Load a causal LLM (optionally 4-bit quantized) and its tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        quantize: Whether to load in 4-bit precision via bitsandbytes.

    Returns:
        Tuple of (tokenizer, model).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading LLM '{model_name}' (4-bit={quantize})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_kwargs = {"device_map": "auto"}
    if quantize:
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb_config
        except ImportError:
            logger.warning("bitsandbytes not available; loading in full precision instead.")

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return tokenizer, model


def generate_response(llm, tokenizer, prompt: str, max_new_tokens: int = 20) -> str:
    """
    Generate a single best layman-language phrase from an LLM given a prompt.

    Args:
        llm: Loaded causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Fully-formatted prompt string.
        max_new_tokens: Generation length cap (kept short: we only want
                        a word or short phrase back).

    Returns:
        The generated replacement phrase, stripped of the prompt prefix
        and surrounding whitespace/quotes.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    completion = full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    cleaned = completion.strip().strip('"').strip("'").split("\n")[0].strip()
    return cleaned


def compute_embeddings(phrases: List[str], sbert_model) -> np.ndarray:
    """
    Compute Sentence-BERT embeddings for a list of phrases.

    Args:
        phrases: List of candidate phrase strings.
        sbert_model: Loaded `SentenceTransformer` instance.

    Returns:
        Numpy array of shape (len(phrases), embedding_dim).
    """
    if not phrases:
        return np.zeros((0, 384))
    return sbert_model.encode(phrases, convert_to_numpy=True, normalize_embeddings=True)


def cluster_by_consensus(candidates: List[str], embeddings: np.ndarray, theta: float, min_cluster_size: int = 2) -> List[str]:
    """
    Retain candidates that form a cluster of size >= `min_cluster_size`
    under cosine-similarity threshold `theta` (simple greedy clustering,
    appropriate for the small candidate sets produced per term, e.g. 3 LLM outputs).

    Args:
        candidates: List of candidate phrases (one per LLM, typically).
        embeddings: Corresponding normalized embeddings.
        theta: Cosine similarity threshold for two phrases to be considered
               the "same" concept.
        min_cluster_size: Minimum number of near-duplicate outputs required
                           for a candidate to be accepted.

    Returns:
        List of unique candidate phrases (one representative per qualifying
        cluster) that reached consensus.
    """
    n = len(candidates)
    if n == 0:
        return []

    sim_matrix = embeddings @ embeddings.T  # cosine similarity since normalized
    visited = [False] * n
    accepted = []

    for i in range(n):
        if visited[i]:
            continue
        cluster_members = [i]
        for j in range(i + 1, n):
            if not visited[j] and sim_matrix[i, j] >= theta:
                cluster_members.append(j)
                visited[j] = True
        visited[i] = True
        if len(cluster_members) >= min_cluster_size:
            # Use the most frequent / first phrase as the cluster representative
            accepted.append(candidates[cluster_members[0]])

    return accepted


def process_term(
    term: str,
    candidate_list: List[Tuple[str, float]],
    llms: List[Tuple[object, object]],
    sbert_model,
    prompt_template: str,
    sentences_dict: Dict[str, List[str]],
    theta: float,
    min_cluster_size: int,
) -> List[str]:
    """
    Run the full consensus pipeline for a single term: prompt each LLM,
    collect their proposed plain-language replacements, embed them, and
    cluster for consensus.

    Args:
        term: Medical term.
        candidate_list: MLM candidates from Module 4 (used to seed context,
                        not directly required by the LLM prompt).
        llms: List of (tokenizer, model) tuples, one per LLM in the panel.
        sbert_model: Loaded SentenceTransformer for embedding.
        prompt_template: Prompt string with `{term}` and `{sentence}` placeholders.
        sentences_dict: term -> example sentences (for prompt context).
        theta: Similarity threshold for consensus clustering.
        min_cluster_size: Minimum agreeing LLMs required.

    Returns:
        List of consensus-approved plain-language candidate phrases.
    """
    example_sentence = sentences_dict.get(term, [""])[0]
    prompt = prompt_template.format(term=term, sentence=example_sentence)

    llm_outputs = []
    for tokenizer, model in llms:
        try:
            response = generate_response(model, tokenizer, prompt)
            if response:
                llm_outputs.append(response)
        except Exception as e:
            logger.error(f"LLM generation failed for term '{term}': {e}")

    return aggregate_consensus(term, llm_outputs, sbert_model, theta, min_cluster_size)


def aggregate_consensus(
    term: str,
    llm_outputs: List[str],
    sbert_model,
    theta: float,
    min_cluster_size: int,
) -> List[str]:
    """
    Embed raw LLM outputs and cluster them to find consensus candidates.

    Args:
        term: Medical term (used for logging only).
        llm_outputs: Raw generated phrases, one per LLM.
        sbert_model: Loaded SentenceTransformer.
        theta: Cosine similarity threshold.
        min_cluster_size: Minimum agreeing outputs required.

    Returns:
        List of consensus phrases for this term.
    """
    if len(llm_outputs) < min_cluster_size:
        logger.debug(f"Term '{term}': insufficient LLM outputs ({len(llm_outputs)}) for consensus.")
        return []

    embeddings = compute_embeddings(llm_outputs, sbert_model)
    consensus = cluster_by_consensus(llm_outputs, embeddings, theta, min_cluster_size)
    return consensus


def run_module5(
    config: dict,
    candidates_raw: Dict[str, List[Tuple[str, float]]],
    sentences_dict: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    High-level orchestration entry point for Module 5.

    Args:
        config: Full pipeline configuration.
        candidates_raw: term -> MLM candidates from Module 4.
        sentences_dict: term -> sentences from Module 3.

    Returns:
        `candidates_consensus` mapping each term to its consensus phrases.
    """
    from sentence_transformers import SentenceTransformer

    llm_cfg = config["llm_consensus"]
    paths = config["paths"]

    sbert_model = SentenceTransformer(llm_cfg["sbert_model"])

    llms = []
    for model_name in llm_cfg["models"]:
        try:
            llms.append(load_llm(model_name, quantize=llm_cfg.get("quantize_4bit", True)))
        except Exception as e:
            logger.error(f"Failed to load LLM '{model_name}': {e}. Skipping it in the panel.")

    if len(llms) < llm_cfg.get("min_cluster_size", 2):
        logger.warning(
            f"Only {len(llms)} LLM(s) loaded successfully; consensus filtering "
            f"requires at least {llm_cfg.get('min_cluster_size', 2)} agreeing outputs "
            f"and may reject all candidates."
        )

    candidates_consensus: Dict[str, List[str]] = {}
    for term, candidate_list in candidates_raw.items():
        consensus = process_term(
            term,
            candidate_list,
            llms,
            sbert_model,
            llm_cfg["prompt_template"],
            sentences_dict,
            theta=llm_cfg.get("similarity_threshold", 0.85),
            min_cluster_size=llm_cfg.get("min_cluster_size", 2),
        )
        if consensus:
            candidates_consensus[term] = consensus

    output_path = os.path.join(paths["intermediate_dir"], "candidates_consensus.json")
    save_json(candidates_consensus, output_path)

    logger.info(f"Module 5 complete: consensus reached for {len(candidates_consensus)}/{len(candidates_raw)} terms")
    return candidates_consensus


if __name__ == "__main__":
    import yaml

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    demo_candidates = {"hypertension": [["high blood pressure", 0.9]]}
    demo_sentences = {"hypertension": ["The patient has hypertension."]}
    run_module5(cfg, demo_candidates, demo_sentences)