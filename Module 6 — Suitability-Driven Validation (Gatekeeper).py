"""
modules/module6_suitability.py
=================================
Module 6: Suitability-Driven Validation (Gatekeeper)

Purpose
-------
Test each candidate substitution in real sentence contexts using a
Suitability (readability) Scorer, and keep only those that improve the
readability score by > a configured threshold on average.

Inputs
------
- `candidates_consensus` from Module 5.
- `sentences_dict` from Module 3.
- A Suitability Scorer returning a literacy percentage `X(%)` for a sentence.

Outputs
-------
- Final KB dictionary `kb_validated` mapping `t` -> list of accepted `g`.
"""

import os
import re
from typing import Dict, List

import requests

from utils.file_utils import save_json
from utils.logger import get_logger
from utils.text_utils import mask_term_in_sentence

logger = get_logger(__name__)


class SuitabilityScorer:
    """
    Wraps a readability/suitability scoring backend. Supports:
      - 'flesch_kincaid': local Flesch-Kincaid Reading Ease computation
        (no external dependency, deterministic, good default/offline mode).
      - 'external_api': delegates to a remote scoring API.
      - 'custom_model': placeholder hook for a project-specific model.

    The `predict` method is the single public entry point downstream
    modules should call.
    """

    def __init__(self, scorer_type: str = "flesch_kincaid", api_url: str = None):
        self.scorer_type = scorer_type
        self.api_url = api_url

    def predict(self, sentence: str) -> float:
        """
        Score a sentence's suitability/readability as a percentage
        (higher = easier to read / more suitable for low-literacy readers).

        Args:
            sentence: The sentence to score.

        Returns:
            A literacy/suitability percentage in [0, 100].
        """
        if self.scorer_type == "flesch_kincaid":
            return self._flesch_reading_ease(sentence)
        elif self.scorer_type == "external_api":
            return self._call_external_api(sentence)
        elif self.scorer_type == "custom_model":
            return self._custom_model_predict(sentence)
        else:
            raise ValueError(f"Unknown scorer_type: {self.scorer_type}")

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower().strip(".,!?;:")
        if not word:
            return 0
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    def _flesch_reading_ease(self, sentence: str) -> float:
        """
        Compute the classic Flesch Reading Ease score:
            206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        Clamped to [0, 100]. Treats the input as a single sentence.
        """
        words = re.findall(r"[A-Za-z']+", sentence)
        n_words = len(words) or 1
        n_sentences = 1
        n_syllables = sum(self._count_syllables(w) for w in words) or 1

        score = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)
        return max(0.0, min(100.0, score))

    def _call_external_api(self, sentence: str) -> float:
        if not self.api_url:
            raise ValueError("scorer_api_url must be set in config for 'external_api' scorer_type")
        try:
            response = requests.post(self.api_url, json={"text": sentence}, timeout=10)
            response.raise_for_status()
            return float(response.json()["score"])
        except Exception as e:
            logger.error(f"External suitability API call failed: {e}. Falling back to Flesch-Kincaid.")
            return self._flesch_reading_ease(sentence)

    def _custom_model_predict(self, sentence: str) -> float:
        """
        Placeholder for a project-specific suitability model
        (e.g., a fine-tuned readability classifier). Replace this
        method body with an actual model call.
        """
        logger.warning("custom_model scorer not implemented; falling back to Flesch-Kincaid.")
        return self._flesch_reading_ease(sentence)


def load_suitability_scorer(scorer_type: str = "flesch_kincaid", api_url: str = None) -> SuitabilityScorer:
    """Instantiate the configured Suitability Scorer."""
    return SuitabilityScorer(scorer_type=scorer_type, api_url=api_url)


def score_sentence(sentence: str, scorer: SuitabilityScorer) -> float:
    """Convenience wrapper around `scorer.predict`."""
    return scorer.predict(sentence)


def evaluate_substitution(
    term: str, candidate: str, sentences: List[str], scorer: SuitabilityScorer
) -> List[float]:
    """
    For each sentence containing `term`, compute the readability delta
    (ΔX) achieved by substituting `candidate` for `term`.

    Args:
        term: Original medical term.
        candidate: Proposed plain-language replacement.
        sentences: Sentences containing `term` (from Module 3).
        scorer: Loaded SuitabilityScorer.

    Returns:
        List of per-sentence percentage-point deltas
        (positive = improvement in readability).
    """
    deltas = []
    for sentence in sentences:
        original_score = score_sentence(sentence, scorer)
        substituted = mask_term_in_sentence(sentence, term, mask_token=candidate)
        # mask_term_in_sentence replaces with `candidate` directly here,
        # reusing the same whole-token-replacement logic as masking.
        new_score = score_sentence(substituted, scorer)
        deltas.append(new_score - original_score)
    return deltas


def accept_candidate(term: str, candidate: str, deltas: List[float], threshold_pct: float = 10.0) -> bool:
    """
    Decide whether a candidate substitution should be accepted based on
    its average readability improvement relative to the original score.

    Args:
        term: Original term (for logging).
        candidate: Candidate replacement (for logging).
        deltas: Per-sentence absolute score deltas from `evaluate_substitution`.
        threshold_pct: Minimum required average percentage-point improvement.

    Returns:
        True if the average delta exceeds `threshold_pct`.
    """
    if not deltas:
        return False
    avg_delta = sum(deltas) / len(deltas)
    accepted = avg_delta > threshold_pct
    logger.debug(
        f"Term='{term}' candidate='{candidate}' avg_delta={avg_delta:.2f} "
        f"threshold={threshold_pct} accepted={accepted}"
    )
    return accepted


def process_term(
    term: str,
    candidate_list: List[str],
    sentences: List[str],
    scorer: SuitabilityScorer,
    min_test_sentences: int,
    threshold_pct: float,
) -> List[str]:
    """
    Evaluate every consensus candidate for a term and return the accepted subset.

    Args:
        term: Medical term.
        candidate_list: Consensus candidates from Module 5.
        sentences: Available sentences for the term.
        scorer: Loaded SuitabilityScorer.
        min_test_sentences: Minimum number of sentences required to run
                             a meaningful evaluation.
        threshold_pct: Required average improvement percentage.

    Returns:
        List of accepted candidate phrases for this term.
    """
    if len(sentences) < min_test_sentences:
        logger.debug(
            f"Term '{term}' has only {len(sentences)} sentences "
            f"(< {min_test_sentences} required); evaluating with what's available."
        )

    accepted = []
    for candidate in candidate_list:
        deltas = evaluate_substitution(term, candidate, sentences, scorer)
        if accept_candidate(term, candidate, deltas, threshold_pct):
            accepted.append(candidate)
    return accepted


def run_module6(
    config: dict,
    candidates_consensus: Dict[str, List[str]],
    sentences_dict: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    High-level orchestration entry point for Module 6.

    Args:
        config: Full pipeline configuration.
        candidates_consensus: term -> consensus candidates from Module 5.
        sentences_dict: term -> sentences from Module 3.

    Returns:
        `kb_validated` mapping each term to its list of accepted candidates.
    """
    suit_cfg = config["suitability"]
    paths = config["paths"]

    scorer = load_suitability_scorer(
        scorer_type=suit_cfg.get("scorer_type", "flesch_kincaid"),
        api_url=suit_cfg.get("scorer_api_url"),
    )

    kb_validated: Dict[str, List[str]] = {}
    for term, candidate_list in candidates_consensus.items():
        sentences = sentences_dict.get(term, [])
        accepted = process_term(
            term,
            candidate_list,
            sentences,
            scorer,
            min_test_sentences=suit_cfg.get("min_test_sentences", 10),
            threshold_pct=suit_cfg.get("improvement_threshold_pct", 10.0),
        )
        if accepted:
            kb_validated[term] = accepted

    output_path = os.path.join(paths["intermediate_dir"], "kb_validated.json")
    save_json(kb_validated, output_path)

    logger.info(f"Module 6 complete: {len(kb_validated)}/{len(candidates_consensus)} terms passed the gatekeeper")
    return kb_validated


if __name__ == "__main__":
    import yaml

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    demo_consensus = {"hypertension": ["high blood pressure"]}
    demo_sentences = {"hypertension": ["The patient was diagnosed with hypertension by the cardiologist."] * 10}
    run_module6(cfg, demo_consensus, demo_sentences)