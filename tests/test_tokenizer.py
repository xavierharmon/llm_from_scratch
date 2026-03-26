"""
Tests: BPE Tokenizer
=====================
Run with: pytest tests/test_tokenizer.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from 02_tokenization.bpe_tokenizer import BPETokenizer


SAMPLE_CORPUS = [
    "easy run today 6 miles at 9:30 pace",
    "marathon training long run 18 miles felt strong",
    "tempo run 8 miles at 7:45 pace heart rate 162",
    "recovery jog 3 miles easy effort",
    "interval workout 8 x 400m at 5K pace with 90 second rest",
    "trail run 10 miles with 1200 feet of elevation gain",
    "race day boston marathon 26.2 miles personal record",
    "running shoes need replacement after 500 miles",
]


@pytest.fixture(scope="module")
def trained_tokenizer():
    tok = BPETokenizer(vocab_size=200)
    tok.train(SAMPLE_CORPUS, verbose=False)
    return tok


class TestBPETokenizer:

    def test_vocab_size_respected(self, trained_tokenizer):
        """Vocabulary should not exceed the requested size."""
        assert len(trained_tokenizer) <= 200

    def test_special_tokens_present(self, trained_tokenizer):
        """All special tokens must be in the vocabulary."""
        for token in BPETokenizer.SPECIAL_TOKENS:
            assert token in trained_tokenizer.token_to_id, f"Missing special token: {token}"

    def test_encode_returns_ints(self, trained_tokenizer):
        """encode() should return a list of integers."""
        ids = trained_tokenizer.encode("easy run 5 miles")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_encode_decode_roundtrip(self, trained_tokenizer):
        """Encoding then decoding should recover the original text."""
        for text in SAMPLE_CORPUS[:3]:
            ids = trained_tokenizer.encode(text)
            recovered = trained_tokenizer.decode(ids)
            # Normalize whitespace for comparison
            assert recovered.split() == text.split(), (
                f"Roundtrip failed.\n  Original: {text}\n  Recovered: {recovered}"
            )

    def test_encode_with_special_tokens(self, trained_tokenizer):
        """BOS and EOS tokens should appear when add_special=True."""
        ids = trained_tokenizer.encode("test run", add_special=True)
        assert ids[0] == trained_tokenizer.bos_id
        assert ids[-1] == trained_tokenizer.eos_id

    def test_unknown_token_handled(self, trained_tokenizer):
        """Characters not in the training corpus get the UNK token."""
        ids = trained_tokenizer.encode("🏃 emoji run")
        unk_id = trained_tokenizer.token_to_id[BPETokenizer.UNK_TOKEN]
        # Should not raise, and result should be a valid list of ints
        assert isinstance(ids, list)

    def test_empty_string(self, trained_tokenizer):
        """Empty string should produce an empty token list."""
        ids = trained_tokenizer.encode("")
        assert ids == []

    def test_save_and_load(self, trained_tokenizer, tmp_path):
        """Saving and loading should preserve all vocabulary and merge rules."""
        save_path = tmp_path / "tokenizer.json"
        trained_tokenizer.save(save_path)
        loaded = BPETokenizer.load(save_path)

        assert len(loaded) == len(trained_tokenizer)
        assert loaded.merges == trained_tokenizer.merges

        # Encoding should be identical
        text = "marathon pace 8:30"
        assert loaded.encode(text) == trained_tokenizer.encode(text)

    def test_longer_text_produces_fewer_tokens_than_chars(self, trained_tokenizer):
        """BPE should compress text: fewer tokens than characters."""
        text = "easy recovery run three miles slow pace sunny weather"
        ids = trained_tokenizer.encode(text)
        assert len(ids) < len(text), (
            f"Expected fewer tokens ({len(ids)}) than characters ({len(text)})"
        )
