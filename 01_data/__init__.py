"""
Phase 1: Data
=============
Load, clean, and prepare running activity data for language model training.

Key classes:
    RunningDataset  — PyTorch Dataset wrapping tokenized run logs
    RunPreprocessor — cleans raw CSV exports into normalized text

Key concepts:
    - The "corpus" for our LM is a collection of free-text running log entries
    - Structured fields (pace, HR, distance) are serialized to natural language
    - This mirrors how real LLMs are trained on heterogeneous text sources
"""

from .data_loader import RunningDataset
from .preprocessing import RunPreprocessor

__all__ = ["RunningDataset", "RunPreprocessor"]
