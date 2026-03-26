"""
PyTorch Dataset for Running Language Model Training
====================================================
Wraps the tokenized corpus into a sliding-window Dataset
compatible with torch.utils.data.DataLoader.

Data engineering analogy:
    RunningDataset is your streaming cursor over a partitioned table.
    Each __getitem__ call is one micro-batch SELECT.
    context_len is the partition/page size.

Key concept — the language modeling objective:
    Given tokens [t0, t1, t2, ... t_{n-1}], predict [t1, t2, ... t_n].
    Input x and target y are the same sequence, offset by 1.
    This is called "next-token prediction" or "causal language modeling."
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class RunningDataset(Dataset):
    """
    Sliding-window dataset over a tokenized running corpus.

    Args:
        token_ids:   flat list or 1-D tensor of all token IDs in the corpus
        context_len: how many tokens the model sees at once (its "memory")
        stride:      step between windows (stride < context_len = overlapping windows)

    Example:
        corpus tokens: [5, 23, 17, 9, 4, 11, 8, 2, ...]
        context_len=4, stride=2:
            window 0 → x=[5,23,17,9],  y=[23,17,9,4]
            window 1 → x=[17,9,4,11],  y=[9,4,11,8]
            ...
    """

    def __init__(self,
                 token_ids: list[int],
                 context_len: int = 256,
                 stride: Optional[int] = None):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.context_len = context_len
        self.stride = stride if stride is not None else context_len

        # Pre-compute valid start indices
        max_start = len(self.token_ids) - self.context_len - 1
        self.starts = list(range(0, max(1, max_start), self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        x = self.token_ids[start:start + self.context_len]
        y = self.token_ids[start + 1:start + self.context_len + 1]  # shifted by 1
        return x, y

    @classmethod
    def from_corpus_file(cls,
                         corpus_path: Path,
                         tokenizer,
                         context_len: int = 256,
                         stride: Optional[int] = None) -> "RunningDataset":
        """Load a corpus text file, tokenize it, and return a dataset."""
        with open(corpus_path) as f:
            text = f.read()
        token_ids = tokenizer.encode(text)
        print(f"Corpus: {len(text):,} chars → {len(token_ids):,} tokens")
        return cls(token_ids, context_len, stride)


def build_dataloaders(
    dataset: RunningDataset,
    batch_size: int = 32,
    val_fraction: float = 0.1,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Split a RunningDataset into train/val DataLoaders.

    Args:
        dataset:       full dataset to split
        batch_size:    number of windows per batch
        val_fraction:  fraction of data to hold out for validation
        num_workers:   parallel data loading workers

    Returns:
        (train_loader, val_loader)
    """
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")
    return train_loader, val_loader
