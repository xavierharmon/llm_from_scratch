"""
Synthetic Running Data Generator
=================================
Generates realistic running activity data for LLM training without
requiring any external API keys or data downloads.

The generated data covers:
    - Activity CSVs (distance, pace, HR, elevation, type)
    - Free-text run log notes (the primary LM training corpus)
    - Per-mile split tables (for sequence modeling tasks)

Usage:
    from generate_synthetic import SyntheticRunGenerator
    gen = SyntheticRunGenerator(seed=42)
    df = gen.generate_activities(5000)
    notes = gen.generate_run_notes(5000)
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


class SyntheticRunGenerator:
    """Generates synthetic but realistic running data for LLM training."""

    RUN_TYPES = ["easy", "long run", "tempo", "interval", "race", "recovery", "trail", "fartlek"]
    LOCATIONS = [
        "Central Park", "Riverside Park", "the track", "the trails", "the treadmill",
        "downtown", "the lake path", "the waterfront", "the greenway", "the hills",
        "the neighborhood", "the forest preserve", "the beach path", "the canal trail",
    ]
    WEATHER = [
        "sunny", "overcast", "humid", "windy", "cold", "hot", "perfect weather",
        "rainy", "foggy", "crisp", "muggy", "breezy",
    ]
    FEELINGS = [
        "felt strong", "struggled early but finished well", "legs felt heavy",
        "felt effortless", "had to push through miles 8-10", "in the zone",
        "felt tired from yesterday", "best run in weeks", "heart rate was elevated",
        "breathing felt easy", "calves were tight", "felt surprisingly good",
        "mentally tough", "had to walk a bit", "finished feeling fresh",
    ]
    GOALS = [
        "Boston qualifier", "sub-3 marathon", "sub-4 marathon", "first 5K",
        "ultramarathon", "half PR", "staying consistent", "base building",
    ]
    RACES = [
        "Boston Marathon", "Chicago Marathon", "New York Marathon", "Berlin Marathon",
        "local 5K", "half marathon", "trail 50K", "track 10K",
    ]

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self._start_date = datetime(2020, 1, 1)

    # ------------------------------------------------------------------
    # Activity CSV
    # ------------------------------------------------------------------

    def _random_pace(self, run_type: str) -> float:
        """Return pace in minutes per mile."""
        base = {
            "easy": 9.5, "long run": 10.0, "tempo": 7.5, "interval": 6.5,
            "race": 7.0, "recovery": 11.0, "trail": 11.5, "fartlek": 8.5,
        }
        return round(base.get(run_type, 9.0) + np.random.normal(0, 0.5), 2)

    def generate_activities(self, n: int = 10_000) -> pd.DataFrame:
        """Generate n running activities as a DataFrame."""
        records = []
        current_date = self._start_date

        for i in range(n):
            run_type = random.choice(self.RUN_TYPES)
            pace = self._random_pace(run_type)

            if run_type == "long run":
                distance = round(np.random.uniform(10, 22), 2)
            elif run_type == "interval":
                distance = round(np.random.uniform(4, 8), 2)
            elif run_type == "recovery":
                distance = round(np.random.uniform(2, 5), 2)
            elif run_type == "race":
                distance = random.choice([3.1, 6.2, 13.1, 26.2])
            else:
                distance = round(np.random.uniform(4, 12), 2)

            duration_min = round(distance * pace, 1)
            hr = int(np.random.normal(
                {"easy": 140, "long run": 145, "tempo": 162, "interval": 172,
                 "race": 168, "recovery": 130, "trail": 148, "fartlek": 155
                 }.get(run_type, 150), 8
            ))
            elevation = int(np.random.exponential(80)) if run_type != "treadmill" else 0

            records.append({
                "activity_id": i + 1,
                "date": current_date.strftime("%Y-%m-%d"),
                "run_type": run_type,
                "distance_miles": distance,
                "duration_minutes": duration_min,
                "pace_min_per_mile": pace,
                "avg_heart_rate": max(100, min(200, hr)),
                "elevation_gain_ft": elevation,
                "location": random.choice(self.LOCATIONS),
                "weather": random.choice(self.WEATHER),
                "perceived_effort": random.randint(1, 10),
            })
            current_date += timedelta(days=random.choice([1, 1, 1, 2, 2, 3]))

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Free-text run log notes (primary LM corpus)
    # ------------------------------------------------------------------

    def _format_pace(self, pace: float) -> str:
        minutes = int(pace)
        seconds = int((pace - minutes) * 60)
        return f"{minutes}:{seconds:02d}"

    def generate_run_notes(self, n: int = 10_000) -> list[str]:
        """Generate n free-text running log entries for language model training."""
        notes = []
        for _ in range(n):
            run_type = random.choice(self.RUN_TYPES)
            distance = round(np.random.uniform(3, 20), 1)
            pace = self._random_pace(run_type)
            location = random.choice(self.LOCATIONS)
            weather = random.choice(self.WEATHER)
            feeling = random.choice(self.FEELINGS)

            templates = [
                f"{distance} miles at {self._format_pace(pace)} average pace through {location}. "
                f"{weather.capitalize()} conditions. {feeling.capitalize()}. "
                f"Heart rate averaged around {random.randint(135, 175)} bpm.",

                f"Got out for a {run_type} today — {distance} miles around {location}. "
                f"Pace was {self._format_pace(pace)}/mi. {feeling.capitalize()}. "
                f"Weather was {weather}.",

                f"Week {random.randint(1, 52)}, day {random.randint(1, 7)}: {run_type.capitalize()} run. "
                f"{distance} miles, {self._format_pace(pace)}/mi, {random.randint(120, 185)} bpm avg HR. "
                f"Ran at {location}. {feeling.capitalize()}.",

                f"Training log: {run_type} — {distance}mi @ {self._format_pace(pace)}. "
                f"Conditions: {weather}. Notes: {feeling}. "
                f"Elevation gain: {random.randint(0, 800)} ft.",

                f"Morning {run_type} in {weather} conditions. "
                f"Covered {distance} miles at {location} in "
                f"{int(distance * pace)} minutes. {feeling.capitalize()}. "
                f"Effort: {random.randint(5, 9)}/10.",
            ]

            if random.random() < 0.15:
                race = random.choice(self.RACES)
                goal = random.choice(self.GOALS)
                templates.append(
                    f"Race prep for {race}. {distance} miles at {self._format_pace(pace)}/mi. "
                    f"Working toward {goal}. {feeling.capitalize()}."
                )

            notes.append(random.choice(templates))

        return notes

    # ------------------------------------------------------------------
    # Per-mile splits (sequence modeling)
    # ------------------------------------------------------------------

    def generate_splits(self, n_runs: int = 2000) -> pd.DataFrame:
        """Generate per-mile split data for sequence-to-sequence tasks."""
        records = []
        for run_id in range(n_runs):
            run_type = random.choice(self.RUN_TYPES)
            n_miles = random.randint(3, 16)
            base_pace = self._random_pace(run_type)

            # Simulate realistic pacing patterns
            paces = []
            for mile in range(n_miles):
                if run_type == "interval":
                    # Alternating fast/recovery
                    split_pace = base_pace * (0.8 if mile % 2 == 0 else 1.3)
                elif run_type == "negative_split":
                    # Get faster each mile
                    split_pace = base_pace - (mile * 0.05)
                else:
                    # Random drift around base pace
                    split_pace = base_pace + np.random.normal(0, 0.2)
                paces.append(round(max(5.0, split_pace), 2))

            for mile, pace in enumerate(paces):
                records.append({
                    "run_id": run_id,
                    "run_type": run_type,
                    "mile": mile + 1,
                    "pace_min_per_mile": pace,
                    "heart_rate": int(np.random.normal(145 + mile * 0.8, 5)),
                })

        return pd.DataFrame(records)


if __name__ == "__main__":
    gen = SyntheticRunGenerator(seed=42)
    print("Generating sample data...")
    df = gen.generate_activities(100)
    print(df.head())
    print(f"\nRun types: {df['run_type'].value_counts().to_dict()}")
    notes = gen.generate_run_notes(5)
    print("\nSample run log notes:")
    for i, note in enumerate(notes, 1):
        print(f"  {i}. {note}")
