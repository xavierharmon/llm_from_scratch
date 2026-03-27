"""
Preprocessing: Raw running data → clean training corpus
========================================================
Converts structured CSV fields into natural language sentences
suitable for language model training.

Data engineering analogy:
    This is your ETL layer. Raw CSVs are the source tables.
    The output corpus is the denormalized, text-serialized fact table
    that gets fed into the tokenizer pipeline.

Usage:
    python 01_data/preprocessing.py
"""

import re
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


class RunPreprocessor:
    """
    Cleans and serializes raw running activity data into
    a text corpus for language model training.
    """

    def __init__(self, raw_dir: Path = Path("01_data/raw"),
                 output_dir: Path = Path("01_data/processed")):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """Normalize whitespace, fix common encoding issues."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
        return text

    def _format_pace(self, pace_float: float) -> str:
        """Convert decimal pace (9.5) to display format (9:30)."""
        minutes = int(pace_float)
        seconds = int((pace_float - minutes) * 60)
        return f"{minutes}:{seconds:02d}"

    # ------------------------------------------------------------------
    # Serialize structured rows to natural language
    # ------------------------------------------------------------------

    def serialize_activity(self, row: pd.Series) -> str:
        """
        Turn one row of the activities CSV into a training sentence.

        Example output:
            "Easy run on 2021-03-14: 6.2 miles at 9:30/mi, avg HR 142 bpm,
             elevation gain 180 ft. Weather was sunny. Perceived effort 5/10."
        """
        parts = [
            f"{row['run_type'].capitalize()} on {row['date']}:",
            f"{row['distance_miles']} miles at {self._format_pace(row['pace_min_per_mile'])}/mi,",
            f"avg HR {row['avg_heart_rate']} bpm,",
            f"elevation gain {row['elevation_gain_ft']} ft.",
            f"Location: {row['location']}.",
            f"Weather was {row['weather']}.",
            f"Perceived effort {row['perceived_effort']}/10.",
        ]
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def build_corpus(self) -> list[str]:
        """
        Read all raw data sources and produce a unified list of
        training sentences. Each sentence ends with a newline
        and represents one 'document' in the corpus.
        """
        corpus = []

        # 1. Structured activities → serialized sentences
        activities_path = self.raw_dir / "running_activities.csv"
        if activities_path.exists():
            df = pd.read_csv(activities_path)
            for _, row in df.iterrows():
                corpus.append(self.serialize_activity(row))
            print(f"  Serialized {len(df):,} activity rows")
        else:
            print(f"  Warning: {activities_path} not found. Run download_data.py first.")

        # 2. Free-text run notes (already natural language — just clean)
        notes_path = self.raw_dir / "run_notes.txt"
        if notes_path.exists():
            with open(notes_path) as f:
                raw_notes = f.read().split("\n\n")
            notes = [self.clean_text(n) for n in raw_notes if len(n.strip()) > 20]
            corpus.extend(notes)
            print(f"  Loaded {len(notes):,} run log notes")
        else:
            print(f"  Warning: {notes_path} not found.")

        return corpus

    def save_corpus(self, corpus: list[str]) -> Path:
        """Write the corpus to a single text file, one entry per line."""
        output_path = self.output_dir / "corpus.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(corpus))
        total_chars = sum(len(s) for s in corpus)
        print(f"\nCorpus saved to {output_path}")
        print(f"  {len(corpus):,} documents")
        print(f"  {total_chars:,} characters (~{total_chars / 1e6:.1f}M chars)")
        return output_path

    def run(self) -> Path:
        print("Building training corpus...")
        corpus = self.build_corpus()
        return self.save_corpus(corpus)


if __name__ == "__main__":
    preprocessor = RunPreprocessor()
    preprocessor.run()
