from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from similar_players import SimilarPlayerModel


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = APP_DIR / "similar_players.joblib"


st.set_page_config(
    page_title="Similar Soccer Players",
    page_icon="soccer",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_saved_model(model_path: str, model_mtime: float) -> SimilarPlayerModel:
    del model_mtime
    return SimilarPlayerModel.load(model_path)


def player_label(row: pd.Series) -> str:
    return f"{row.player_name} | ID {row.player_id}"


def format_results(results: pd.DataFrame) -> pd.DataFrame:
    formatted = results.copy()
    formatted["similarity"] = (formatted["similarity"] * 100).round(1)
    formatted = formatted.rename(
        columns={
            "similarity": "Similarity %",
            "player_id": "Player ID",
            "player_name": "Player",
            "position": "Position",
            "team_name": "Team",
            "league_folder": "League",
            "age": "Age",
            "played_minutes": "Minutes",
            "ratings": "Rating",
        }
    )
    return formatted


def render_player_card(player: pd.Series) -> None:
    st.subheader(player["player_name"])
    st.caption(
        f"{player['position']} | {player['team_name']} | {player['league_folder']} | "
        f"Player ID {player['player_id']}"
    )
    metric_cols = st.columns(5)
    metric_cols[0].metric("Position", player["position"])
    metric_cols[1].metric("Age", int(player["age"]))
    metric_cols[2].metric("Minutes", f"{int(player['played_minutes']):,}")
    metric_cols[3].metric("Rating", f"{float(player['ratings']):.2f}")
    metric_cols[4].metric("Touches p90", f"{float(player['touches_p90']):.1f}")


st.title("Similar Soccer Players")
st.caption("")

with st.sidebar:
    st.header("Results")
    top_n = st.slider("Similar players", min_value=3, max_value=30, value=10)
    same_position_only = st.checkbox("Same position only", value=True)
    same_league_only = st.checkbox("Same league only", value=False)
    min_similarity = st.slider("Minimum similarity %", min_value=0, max_value=100, value=0)

try:
    model = load_saved_model(str(DEFAULT_MODEL), DEFAULT_MODEL.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the saved model at {DEFAULT_MODEL}: {exc}")
    st.stop()

if "position" not in model.data.columns:
    st.error("The saved model does not include a position column. Rebuild similar_players.joblib from the new CSV.")
    st.stop()

age_min = int(model.data["age"].min())
age_max = int(model.data["age"].max())
with st.sidebar:
    age_range = st.slider(
        "Age range",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
    )

players = model.data.sort_values(["league_folder", "team_name", "position", "player_name"]).reset_index(drop=True)

selector_cols = st.columns([2, 2, 1, 1])
league_options = sorted(players["league_folder"].unique().tolist())
selected_league = selector_cols[0].selectbox("1. Choose league", league_options)

league_players = players[players["league_folder"] == selected_league]
team_options = sorted(league_players["team_name"].unique().tolist())
selected_team = selector_cols[1].selectbox("2. Choose team", team_options)

team_players = league_players[league_players["team_name"] == selected_team]
position_options = ["ALL"] + sorted(team_players["position"].unique().tolist())
selected_position = selector_cols[2].selectbox("3. Choose position", position_options)
name_query = selector_cols[3].text_input("Search player")

position_players = (
    team_players
    if selected_position == "ALL"
    else team_players[team_players["position"] == selected_position]
)
filtered_players = position_players.copy()
if name_query:
    filtered_players = filtered_players[
        filtered_players["player_name"].str.contains(name_query, case=False, na=False)
    ]

if filtered_players.empty:
    st.warning("No players match the selected league, team, position, and search text.")
    st.stop()

labels = [player_label(row) for _, row in filtered_players.iterrows()]
selected_label = st.selectbox("4. Choose player", labels)
selected_player_id = int(selected_label.rsplit("ID ", 1)[1])
selected_player = model.data.loc[model.data["player_id"] == selected_player_id].iloc[0]

render_player_card(selected_player)

try:
    raw_results = model.similar_players(
        player_id=selected_player_id,
        top_n=30,
        same_league_only=same_league_only,
        same_position_only=same_position_only,
    )
except Exception as exc:
    st.error(f"Could not calculate similar players: {exc}")
    st.stop()

results = raw_results.copy()
results = results[(results["age"] >= age_range[0]) & (results["age"] <= age_range[1])]
if min_similarity > 0:
    results = results[results["similarity"] >= min_similarity / 100]
results = results.head(top_n)

st.divider()
st.subheader("Most Similar Players")
if results.empty:
    st.info("No similar players match the current result filters.")
else:
    st.dataframe(
        format_results(results),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Similarity %": st.column_config.ProgressColumn(
                "Similarity %",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            )
        },
    )

    st.download_button(
        "Download results",
        data=results.to_csv(index=False),
        file_name=f"similar_players_{selected_player_id}.csv",
        mime="text/csv",
    )

with st.expander("Model details"):
    st.write(f"Players in model: {len(model.data):,}")
    st.write(f"Features used: {len(model.features)}")
    st.dataframe(pd.DataFrame({"feature": model.features}), use_container_width=True, hide_index=True)
