#!/usr/bin/env python3
"""
Find similar soccer players with scikit-learn.

The model uses per-90 production, success-rate metrics, rating, and age, while
filtering by minutes so low-sample players do not dominate the nearest-neighbor
space. Identity columns and raw totals are excluded from the feature vector.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


IDENTITY_COLUMNS = {
    "league_folder",
    "team_id",
    "player_id",
    "player_name",
    "player_image",
    "position",
    "team_name",
    "team_image",
}


@dataclass
class SimilarPlayerModel:
    data: pd.DataFrame
    features: list[str]
    preprocessor: Pipeline
    neighbors: NearestNeighbors
    feature_matrix: np.ndarray

    @classmethod
    def fit(
        cls,
        csv_path: str | Path,
        min_minutes: int = 600,
        extra_features: Iterable[str] = ("age", "ratings"),
    ) -> "SimilarPlayerModel":
        raw = pd.read_csv(csv_path)
        data = raw.loc[raw["played_minutes"] >= min_minutes].copy()
        if data.empty:
            raise ValueError(f"No players remain after min_minutes={min_minutes}.")

        features = select_features(data, extra_features=extra_features)
        preprocessor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )
        matrix = preprocessor.fit_transform(data[features])

        neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
        neighbors.fit(matrix)

        return cls(
            data=data.reset_index(drop=True),
            features=features,
            preprocessor=preprocessor,
            neighbors=neighbors,
            feature_matrix=matrix,
        )

    def save(self, output_path: str | Path) -> None:
        payload = {
            "data": self.data,
            "features": self.features,
            "preprocessor": self.preprocessor,
            "neighbors": self.neighbors,
            "feature_matrix": self.feature_matrix,
        }
        joblib.dump(payload, output_path)

    @classmethod
    def load(cls, model_path: str | Path) -> "SimilarPlayerModel":
        payload = joblib.load(model_path)
        return cls(**payload)

    def find_player_index(
        self,
        player_name: str | None = None,
        player_id: int | None = None,
        team_name: str | None = None,
    ) -> int:
        if player_id is not None:
            matches = self.data.index[self.data["player_id"] == player_id].tolist()
            if matches:
                return matches[0]
            raise ValueError(f"Could not find player_id={player_id}.")

        if not player_name:
            raise ValueError("Provide either player_name or player_id.")

        normalized = normalize_text(player_name)
        candidates = self.data[
            self.data["player_name"].map(normalize_text).str.contains(normalized, na=False)
        ].copy()

        if team_name and not candidates.empty:
            team_norm = normalize_text(team_name)
            team_matches = candidates[
                candidates["team_name"].map(normalize_text).str.contains(team_norm, na=False)
            ]
            if not team_matches.empty:
                candidates = team_matches

        if len(candidates) == 1:
            return int(candidates.index[0])
        if len(candidates) > 1:
            choices = candidates[["player_id", "player_name", "team_name", "league_folder"]]
            raise ValueError(
                "Multiple players matched. Re-run with --player-id or --team.\n"
                + choices.to_string(index=False)
            )

        names = self.data["player_name"].dropna().unique().tolist()
        close = get_close_matches(player_name, names, n=5, cutoff=0.55)
        hint = f" Closest names: {', '.join(close)}" if close else ""
        raise ValueError(f"Could not find player_name={player_name!r}.{hint}")

    def similar_players(
        self,
        player_name: str | None = None,
        player_id: int | None = None,
        team_name: str | None = None,
        top_n: int = 10,
        same_league_only: bool = False,
        same_position_only: bool = False,
    ) -> pd.DataFrame:
        idx = self.find_player_index(player_name=player_name, player_id=player_id, team_name=team_name)
        query = self.data.iloc[idx]

        candidate_count = min(len(self.data), max(top_n + 50, top_n * 20))
        distances, indices = self.neighbors.kneighbors(
            self.feature_matrix[idx : idx + 1],
            n_neighbors=candidate_count,
        )

        rows = []
        for distance, match_idx in zip(distances[0], indices[0]):
            if match_idx == idx:
                continue
            candidate = self.data.iloc[match_idx]
            if same_league_only and candidate["league_folder"] != query["league_folder"]:
                continue
            if (
                same_position_only
                and "position" in self.data.columns
                and candidate["position"] != query["position"]
            ):
                continue
            rows.append(
                {
                    "similarity": round(1 - float(distance), 4),
                    "player_id": int(candidate["player_id"]),
                    "player_name": candidate["player_name"],
                    "position": candidate["position"] if "position" in self.data.columns else "",
                    "team_name": candidate["team_name"],
                    "league_folder": candidate["league_folder"],
                    "age": int(candidate["age"]),
                    "played_minutes": int(candidate["played_minutes"]),
                    "ratings": round(float(candidate["ratings"]), 3),
                }
            )
            if len(rows) >= top_n:
                break

        return pd.DataFrame(rows)


def normalize_text(value: object) -> str:
    return str(value).casefold().strip()


def select_features(data: pd.DataFrame, extra_features: Iterable[str]) -> list[str]:
    per90 = [column for column in data.columns if column.endswith("_p90")]
    success_rates = [column for column in data.columns if column.endswith("Success")]
    extras = [column for column in extra_features if column in data.columns]

    features = per90 + success_rates + extras
    numeric_features = [
        column
        for column in features
        if column not in IDENTITY_COLUMNS and pd.api.types.is_numeric_dtype(data[column])
    ]
    if not numeric_features:
        raise ValueError("No numeric modeling features were found.")
    return numeric_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find similar soccer players with scikit-learn.")
    parser.add_argument("--csv", required=True, help="Path to the source player CSV.")
    parser.add_argument("--player-name", help="Player name to search for.")
    parser.add_argument("--player-id", type=int, help="Player ID to search for.")
    parser.add_argument("--team", help="Optional team name disambiguator.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of similar players to return.")
    parser.add_argument("--min-minutes", type=int, default=600, help="Minimum minutes for model inclusion.")
    parser.add_argument("--same-league-only", action="store_true", help="Only return players from the same league.")
    parser.add_argument("--same-position-only", action="store_true", help="Only return players with the same position.")
    parser.add_argument("--load-model", help="Optional path to an already fitted joblib model.")
    parser.add_argument("--save-model", help="Optional path to save the fitted model with joblib.")
    parser.add_argument("--output-csv", help="Optional path to save the similarity results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = (
        SimilarPlayerModel.load(args.load_model)
        if args.load_model
        else SimilarPlayerModel.fit(args.csv, min_minutes=args.min_minutes)
    )
    results = model.similar_players(
        player_name=args.player_name,
        player_id=args.player_id,
        team_name=args.team,
        top_n=args.top_n,
        same_league_only=args.same_league_only,
        same_position_only=args.same_position_only,
    )

    if args.save_model:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.save_model)

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output_csv, index=False)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
