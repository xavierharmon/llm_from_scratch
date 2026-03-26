"""
Byte-Pair Encoding (BPE) Tokenizer — From Scratch
===================================================
BPE is the tokenization algorithm used by GPT-2, GPT-4, LLaMA, and most
modern LLMs. This implementation builds it from first principles.

Algorithm intuition (data engineering analogy):
    1. Start with a "vocabulary" of individual characters (like a character-level schema)
    2. Count all adjacent pair frequencies — like a GROUP BY on bigrams
    3. Merge the most frequent pair into a new token — like adding a derived column
    4. Repeat until vocab_size is reached
    5. The learned merge rules are your "compression codec"

Example:
    Corpus: "run runner running"
    Start:  r u n | r u n n e r | r u n n i n g
    Step 1: 'un' is most frequent → merge: r [un] | r [un] n e r | r [un] n i n g
    Step 2: 'run' appears → merge: [run] | [run] n e r | [run] n i n g
    ...and so on until vocab_size is reached.
"""

import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer trained from scratch.

    Attributes:
        vocab_size:  target vocabulary size (typically 4k–50k for small LMs)
        merges:      ordered list of (pair → merged) rules learned during training
        vocab:       id → token string mapping
        token_to_id: token string → id mapping
    """

    # Special tokens
    PAD_TOKEN = "<|pad|>"
    UNK_TOKEN = "<|unk|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self._merge_lookup: dict[tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _get_word_freqs(self, corpus: list[str]) -> Counter:
        """Count word frequencies in the corpus (with space prefix convention)."""
        freq: Counter = Counter()
        for line in corpus:
            for word in line.strip().split():
                # Ġ prefix (GPT-2 convention) marks word-initial tokens
                freq["Ġ" + word] += 1
        return freq

    def _word_to_chars(self, word: str) -> tuple[str, ...]:
        """Split a word string into its individual characters."""
        return tuple(word)

    def _get_pair_freqs(self, word_freqs: dict[tuple, int]) -> Counter:
        """Count how often each adjacent pair of tokens appears across all words."""
        pairs: Counter = Counter()
        for word_tokens, freq in word_freqs.items():
            for i in range(len(word_tokens) - 1):
                pairs[(word_tokens[i], word_tokens[i + 1])] += freq
        return pairs

    def _merge_pair(self,
                    pair: tuple[str, str],
                    word_freqs: dict[tuple, int]) -> dict[tuple, int]:
        """Apply one merge rule: replace all occurrences of (a, b) with 'ab'."""
        a, b = pair
        merged = a + b
        new_word_freqs: dict[tuple, int] = {}

        for word_tokens, freq in word_freqs.items():
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and word_tokens[i] == a and word_tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            new_word_freqs[tuple(new_tokens)] = freq

        return new_word_freqs

    def train(self, corpus: list[str], verbose: bool = True) -> "BPETokenizer":
        """
        Train BPE on a list of strings.

        Args:
            corpus:  list of text strings (one per document/sentence)
            verbose: print progress every 500 merges

        Returns:
            self (for chaining)
        """
        # Initialize vocabulary with all unique characters + special tokens
        char_vocab: set[str] = set()
        for line in corpus:
            char_vocab.update(line)

        # Build initial vocab: special tokens first, then chars
        all_tokens = self.SPECIAL_TOKENS + sorted(char_vocab)
        self.vocab = {i: t for i, t in enumerate(all_tokens)}
        self.token_to_id = {t: i for i, t in self.vocab.items()}

        # Initialize word frequencies as character sequences
        word_freqs_raw = self._get_word_freqs(corpus)
        word_freqs: dict[tuple, int] = {
            self._word_to_chars(w): f for w, f in word_freqs_raw.items()
        }

        n_merges = self.vocab_size - len(self.vocab)
        if verbose:
            print(f"Initial vocab: {len(self.vocab)} tokens")
            print(f"Performing {n_merges} BPE merges...")

        for step in range(n_merges):
            pair_freqs = self._get_pair_freqs(word_freqs)
            if not pair_freqs:
                print("  No more pairs to merge. Stopping early.")
                break

            best_pair = pair_freqs.most_common(1)[0][0]
            merged = best_pair[0] + best_pair[1]

            self.merges.append(best_pair)
            self._merge_lookup[best_pair] = merged

            new_id = len(self.vocab)
            self.vocab[new_id] = merged
            self.token_to_id[merged] = new_id

            word_freqs = self._merge_pair(best_pair, word_freqs)

            if verbose and (step + 1) % 500 == 0:
                print(f"  Step {step+1}/{n_merges}: merged '{merged}' "
                      f"(freq={pair_freqs[best_pair]:,})")

        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")
        return self

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def _tokenize_word(self, word: str) -> list[str]:
        """Apply all learned merge rules to a single word."""
        tokens = list(word)
        for (a, b) in self.merges:
            merged = a + b
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """
        Encode a string to a list of token IDs.

        Args:
            text:         input string
            add_special:  prepend BOS and append EOS tokens

        Returns:
            list of integer token IDs
        """
        ids = []
        if add_special:
            ids.append(self.token_to_id[self.BOS_TOKEN])

        for word in text.split():
            prefixed = "Ġ" + word
            tokens = self._tokenize_word(prefixed)
            for token in tokens:
                ids.append(self.token_to_id.get(token, self.token_to_id[self.UNK_TOKEN]))

        if add_special:
            ids.append(self.token_to_id[self.EOS_TOKEN])
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back to a string.

        Args:
            ids: list of integer token IDs

        Returns:
            decoded string
        """
        tokens = []
        for i in ids:
            token = self.vocab.get(i, self.UNK_TOKEN)
            if token in self.SPECIAL_TOKENS:
                continue
            tokens.append(token)

        text = "".join(tokens)
        # Remove Ġ prefix and restore spaces
        text = text.replace("Ġ", " ").strip()
        return text

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save tokenizer vocab and merges to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "merges": [[a, b] for a, b in self.merges],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Load a previously saved tokenizer."""
        with open(path) as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.vocab = {int(k): v for k, v in data["vocab"].items()}
        tokenizer.token_to_id = {v: int(k) for k, v in tokenizer.vocab.items()}
        tokenizer.merges = [(a, b) for a, b in data["merges"]]
        tokenizer._merge_lookup = {(a, b): a + b for a, b in tokenizer.merges}
        return tokenizer

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]
