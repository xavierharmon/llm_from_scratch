"""
Extended BPE Roundtrip Tests
=============================
Focused tests for edge cases in encoding and decoding.

Run with: pytest 02_tokenization/tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from step_02_tokenization.bpe_tokenizer import BPETokenizer

SMALL_CORPUS = [
    "easy run 5 miles at 9:30 pace felt good",
    "marathon training long run 18 miles strong",
    "tempo 8 miles 7:45 heart rate 162",
    "recovery jog 3 miles easy effort sunny",
    "5K race personal record 24:15 fast",
]


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer(vocab_size=150)
    t.train(SMALL_CORPUS, verbose=False)
    return t


def test_numbers_encode_cleanly(tok):
    """Pace and distance numbers should encode without raising errors."""
    texts = ["3:45", "26.2", "8:30/mi", "180 bpm", "650 ft"]
    for text in texts:
        ids = tok.encode(text)
        assert isinstance(ids, list) and len(ids) > 0, f"Failed on '{text}'"


def test_roundtrip_preserves_word_order(tok):
    """Word order must be preserved after encode → decode."""
    text = "marathon pace was 8:30 per mile and heart rate 155 bpm"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded.split() == text.split()


def test_repeated_encode_is_deterministic(tok):
    """Encoding the same text twice should give the same result."""
    text = "easy run 6 miles"
    assert tok.encode(text) == tok.encode(text)


def test_all_special_tokens_have_unique_ids(tok):
    """Each special token must have a unique ID."""
    ids = [tok.token_to_id[t] for t in BPETokenizer.SPECIAL_TOKENS]
    assert len(ids) == len(set(ids)), "Duplicate special token IDs detected"


def test_vocab_ids_are_contiguous(tok):
    """Token IDs should form a contiguous range from 0 to len(vocab)-1."""
    expected = set(range(len(tok.vocab)))
    actual = set(tok.vocab.keys())
    assert expected == actual, "Vocab IDs are not contiguous"
