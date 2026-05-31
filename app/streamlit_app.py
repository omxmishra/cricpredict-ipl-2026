# ==================================================
# PATH SETUP (MUST BE FIRST)
# ==================================================

import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==================================================
# IMPORTS
# ==================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from cricpredict.data_loader import (
    load_processed_data,
    load_model
)

from cricpredict.predictor import (
    predict_match
)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="CricPredict IPL 2026",
    page_icon="🏏",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def get_data():
    return load_processed_data()


@st.cache_resource
def get_model():
    return load_model(
        "logistic_regression.pkl"
    )


df = get_data()
model = get_model()

# ==================================================
# TITLE
# ==================================================

st.title("🏏 CricPredict IPL 2026")

st.markdown(
    """
    IPL Match Prediction using Machine Learning,
    Elo Ratings, Team Form, Head-to-Head Records
    and Venue Statistics.
    """
)

# ==================================================
# SIDEBAR
# ==================================================

teams = sorted(
    df["team1"].dropna().unique()
)

venues = sorted(
    df["venue"].dropna().unique()
)

st.sidebar.header("Match Configuration")

team1 = st.sidebar.selectbox(
    "Team 1",
    teams
)

team2 = st.sidebar.selectbox(
    "Team 2",
    teams,
    index=1
)

venue = st.sidebar.selectbox(
    "Venue",
    venues
)

toss_winner = st.sidebar.selectbox(
    "Toss Winner",
    [team1, team2]
)

predict_button = st.sidebar.button(
    "Predict Winner"
)

# ==================================================
# VALIDATION
# ==================================================

if team1 == team2:
    st.error(
        "Please select two different teams."
    )
    st.stop()

# ==================================================
# PREDICTION
# ==================================================

if predict_button:

    result = predict_match(
        model=model,
        model_df=df,
        team1=team1,
        team2=team2,
        venue=venue,
        toss_winner=toss_winner
    )

    st.success(
        f"🏆 Predicted Winner: "
        f"{result['predicted_winner']}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            team1,
            f"{result['team1_probability']}%"
        )

    with col2:
        st.metric(
            team2,
            f"{result['team2_probability']}%"
        )

    prob_df = pd.DataFrame(
        {
            "Team": [team1, team2],
            "Probability": [
                result["team1_probability"],
                result["team2_probability"]
            ]
        }
    )

    fig = px.bar(
        prob_df,
        x="Team",
        y="Probability",
        text="Probability",
        title="Win Probability"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.info(
        "Configure a match and click Predict Winner."
    )