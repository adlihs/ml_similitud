# Similar Player Model

This project builds a scikit-learn nearest-neighbor model to find soccer players
with similar statistical profiles.

## Why this approach

- Uses per-90 metrics and success rates, so similarity is based on playing style
  rather than raw opportunity.
- Filters out small samples with `--min-minutes` to reduce noise.
- Scales features with `RobustScaler`, which is helpful for soccer data because
  many event columns have outliers.
- Uses `NearestNeighbors(metric="cosine")` from scikit-learn to compare player
  profile direction after scaling.

## Run

```bash
python3 similar_players.py \
  --csv /Users/edmundneil/Downloads/New_Query_2026_05_12_08_53_30-7.csv \
  --player-name "Nathan Saliba" \
  --top-n 10
```

Save the fitted model and output:

```bash
python3 similar_players.py \
  --csv /Users/edmundneil/Downloads/New_Query_2026_05_12_08_53_30-7.csv \
  --player-name "Nathan Saliba" \
  --top-n 10 \
  --save-model similar_players.joblib \
  --output-csv nathan_saliba_similar_players.csv
```

If a name matches multiple players, rerun with `--player-id` or add `--team`.

## Modeling features

The script automatically selects:

- All columns ending in `_p90`
- All columns ending in `Success`
- `age`
- `ratings`

It excludes IDs, names, teams, images, leagues, and raw count totals.

## Streamlit app

The app is designed for publishing with the saved model artifact. Make sure
`similar_players.joblib` is in the same folder as `app.py`, then launch:

```bash
streamlit run app.py
```

The user interface does not require a local CSV path. Rebuild the model from the
CSV only when you want to update the underlying dataset:

```bash
python3 similar_players.py \
  --csv /Users/edmundneil/Downloads/New_Query_2026_05_12_08_53_30-7.csv \
  --player-name "Nathan Saliba" \
  --save-model similar_players.joblib
```
