"""
BLEU Score
===========
Bilingual Evaluation Understudy — measures n-gram overlap between
generated text and reference text.

Note on BLEU for language modeling:
    BLEU is primarily designed for translation, where there's a known
    correct output to compare against. For open-ended generation (like
    our running logs), BLEU is a rough proxy for fluency and vocabulary
    fidelity — not a gold standard. Perplexity is generally more
    informative for LM evaluation.

    Still useful for: fine-tuning evaluation, comparing generation
    strategies, checking if the model has learned domain vocabulary.

Usage:
    from 08_evaluation.bleu_score import sentence_bleu, corpus_bleu
"""

import math
from collections import Counter
from typing import Union


def _ngrams(tokens: list[str], n: int) -> Counter:
    """Count all n-grams in a token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _clipped_precision(hypothesis: list[str], references: list[list[str]], n: int) -> tuple[int, int]:
    """
    Compute clipped n-gram precision: how many hypothesis n-grams appear in references,
    clipped to the max count in any single reference.
    """
    hyp_ngrams = _ngrams(hypothesis, n)
    if not hyp_ngrams:
        return 0, 0

    max_ref_counts: Counter = Counter()
    for ref in references:
        ref_ngrams = _ngrams(ref, n)
        for ngram in hyp_ngrams:
            max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])

    clipped_count = sum(min(count, max_ref_counts[ngram]) for ngram, count in hyp_ngrams.items())
    total_count = sum(hyp_ngrams.values())
    return clipped_count, total_count


def _brevity_penalty(hypothesis_len: int, reference_len: int) -> float:
    """Penalize hypotheses shorter than references."""
    if hypothesis_len >= reference_len:
        return 1.0
    if hypothesis_len == 0:
        return 0.0
    return math.exp(1 - reference_len / hypothesis_len)


def sentence_bleu(
    hypothesis: Union[str, list[str]],
    references: Union[str, list[str], list[list[str]]],
    max_n: int = 4,
    weights: tuple = (0.25, 0.25, 0.25, 0.25),
) -> float:
    """
    Compute BLEU score for a single hypothesis against one or more references.

    Args:
        hypothesis:  generated text (string or token list)
        references:  reference text(s) (string, list of strings, or list of token lists)
        max_n:       maximum n-gram order (BLEU-4 is standard)
        weights:     per-order weights (uniform by default)

    Returns:
        BLEU score in [0, 1]
    """
    # Normalize to token lists
    if isinstance(hypothesis, str):
        hyp_tokens = hypothesis.lower().split()
    else:
        hyp_tokens = hypothesis

    if isinstance(references, str):
        ref_list = [references.lower().split()]
    elif isinstance(references[0], str):
        ref_list = [r.lower().split() for r in references]
    else:
        ref_list = references

    # Brevity penalty uses closest reference length
    ref_lens = [len(r) for r in ref_list]
    closest_ref_len = min(ref_lens, key=lambda l: abs(l - len(hyp_tokens)))
    bp = _brevity_penalty(len(hyp_tokens), closest_ref_len)

    if bp == 0.0:
        return 0.0

    # Compute weighted geometric mean of clipped precisions
    log_score = 0.0
    for n in range(1, max_n + 1):
        clipped, total = _clipped_precision(hyp_tokens, ref_list, n)
        if total == 0 or clipped == 0:
            return 0.0
        log_score += weights[n-1] * math.log(clipped / total)

    return bp * math.exp(log_score)


def corpus_bleu(
    hypotheses: list[str],
    references: list[list[str]],
    max_n: int = 4,
) -> float:
    """
    Compute corpus-level BLEU (aggregates counts before computing precision).

    Args:
        hypotheses:  list of generated strings
        references:  list of reference strings (one per hypothesis)
        max_n:       maximum n-gram order

    Returns:
        Corpus BLEU score in [0, 1]
    """
    assert len(hypotheses) == len(references), "Must have one reference per hypothesis"

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.lower().split()
        ref_tokens = [ref.lower().split()]
        hyp_len += len(hyp_tokens)
        ref_len += len(ref_tokens[0])
        for n in range(1, max_n + 1):
            c, t = _clipped_precision(hyp_tokens, ref_tokens, n)
            clipped_counts[n-1] += c
            total_counts[n-1] += t

    bp = _brevity_penalty(hyp_len, ref_len)
    if bp == 0.0:
        return 0.0

    weights = [1.0 / max_n] * max_n
    log_score = 0.0
    for n in range(max_n):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            return 0.0
        log_score += weights[n] * math.log(clipped_counts[n] / total_counts[n])

    return bp * math.exp(log_score)
