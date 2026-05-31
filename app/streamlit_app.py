import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd

from cricpredict.data_loader import load_processed_data, load_model
from cricpredict.predictor import predict_match

st.set_page_config(
    page_title="CricPredict · IPL 2026",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #1a1d23; color: #e2e4e9; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 980px; margin: 0 auto; }
section[data-testid="stSidebar"] { display: none; }

/* nav */
.nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 0 24px;
    border-bottom: 1px solid #2a2d35;
    margin-bottom: 36px;
}
.nav-logo { font-size: 1.1rem; font-weight: 600; color: #f0f1f3; letter-spacing: -0.01em; }
.nav-logo span { color: #4f8ef7; }
.nav-tag {
    font-size: 0.75rem;
    color: #4f8ef7;
    background: rgba(79,142,247,0.1);
    border: 1px solid rgba(79,142,247,0.2);
    border-radius: 20px;
    padding: 3px 12px;
}

/* form card */
.form-card {
    background: #21242c;
    border: 1px solid #2e3140;
    border-radius: 12px;
    padding: 28px 30px 24px;
    margin-bottom: 24px;
}
.form-card-title {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #5a6070;
    margin-bottom: 20px;
}

/* selectbox */
div[data-testid="stSelectbox"] label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #8b93a5 !important;
    margin-bottom: 5px !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #2a2d38 !important;
    border: 1px solid #363a4a !important;
    border-radius: 8px !important;
    color: #e2e4e9 !important;
    font-size: 0.9rem !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #4f8ef7 !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.12) !important;
}
div[data-testid="stSelectbox"] svg { fill: #5a6070 !important; }

/* radio */
div[data-testid="stRadio"] > label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #8b93a5 !important;
}
div[data-testid="stRadio"] > div { gap: 10px; }
div[data-testid="stRadio"] > div > label {
    background: #2a2d38 !important;
    border: 1px solid #363a4a !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    color: #9ca3b0 !important;
    cursor: pointer;
    transition: all 0.15s;
}
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: rgba(79,142,247,0.12) !important;
    border-color: #4f8ef7 !important;
    color: #7fb3ff !important;
    font-weight: 500 !important;
}

/* button */
div[data-testid="stButton"] > button {
    background: #4f8ef7 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 10px 28px !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s !important;
}
div[data-testid="stButton"] > button:hover { background: #3d7ef0 !important; }

/* result winner card */
.winner-card {
    background: #21242c;
    border: 1px solid #2e3140;
    border-radius: 12px;
    padding: 26px 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}
.winner-label { font-size: 0.72rem; color: #5a6070; font-weight: 500; margin-bottom: 5px; letter-spacing: 0.04em; text-transform: uppercase; }
.winner-name { font-size: 1.45rem; font-weight: 600; color: #f0f1f3; letter-spacing: -0.02em; }
.winner-sub { font-size: 0.8rem; color: #5a6070; margin-top: 3px; }
.winner-prob { font-size: 2.6rem; font-weight: 600; color: #4f8ef7; letter-spacing: -0.04em; line-height: 1; }
.winner-prob-label { font-size: 0.72rem; color: #5a6070; text-align: right; margin-top: 3px; }

/* section card */
.section-card {
    background: #21242c;
    border: 1px solid #2e3140;
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 16px;
}
.section-card-title {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #5a6070;
    margin-bottom: 18px;
}

/* prob bars */
.pbar-row { display: flex; align-items: center; gap: 14px; margin-bottom: 12px; }
.pbar-name { font-size: 0.85rem; font-weight: 500; color: #c8ccd6; width: 150px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pbar-track { flex: 1; height: 6px; background: #2e3140; border-radius: 100px; overflow: hidden; }
.pbar-fill-t1 { height: 100%; background: #4f8ef7; border-radius: 100px; }
.pbar-fill-t2 { height: 100%; background: #4a5060; border-radius: 100px; }
.pbar-val { font-size: 0.82rem; font-weight: 500; color: #8b93a5; width: 40px; text-align: right; flex-shrink: 0; }

/* two col */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }

/* stat rows */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #272a33;
    font-size: 0.83rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: #6b7485; }
.stat-vals { display: flex; gap: 18px; }
.stat-v1 { font-weight: 500; color: #7fb3ff; min-width: 46px; text-align: right; }
.stat-v2 { font-weight: 500; color: #8b93a5; min-width: 46px; text-align: right; }
.team-legend { display: flex; gap: 16px; font-size: 0.75rem; color: #5a6070; margin-bottom: 14px; }
.ldot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }

/* h2h bars */
.h2h-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; font-size: 0.83rem; }
.h2h-name { color: #c8ccd6; font-weight: 500; width: 88px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.h2h-track { flex: 1; height: 5px; background: #2e3140; border-radius: 100px; overflow: hidden; }
.h2h-fill-t1 { height: 100%; background: #4f8ef7; border-radius: 100px; }
.h2h-fill-t2 { height: 100%; background: #4a5060; border-radius: 100px; }
.h2h-pct { font-size: 0.78rem; color: #6b7485; width: 32px; text-align: right; flex-shrink: 0; }

/* kv rows */
.kv-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #272a33;
    font-size: 0.83rem;
}
.kv-row:last-child { border-bottom: none; }
.kv-key { color: #6b7485; }
.kv-val { font-weight: 500; color: #c8ccd6; }

/* idle */
.idle-box {
    background: #21242c;
    border: 1px solid #2e3140;
    border-radius: 12px;
    padding: 64px 32px;
    text-align: center;
}
.idle-icon { font-size: 2.2rem; margin-bottom: 14px; }
.idle-title { font-size: 0.95rem; font-weight: 500; color: #6b7485; margin-bottom: 6px; }
.idle-sub { font-size: 0.82rem; color: #404555; }

/* alert */
div[data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.84rem !important; background: #2a1f2d !important; border-color: #6b3a6b !important; }
div[data-testid="stSpinner"] p { color: #8b93a5 !important; }
</style>
""", unsafe_allow_html=True)

# ── DATA ───────────────────────────────────────────────────────────────────────

@st.cache_data
def get_data():
    return load_processed_data()

@st.cache_resource
def get_model():
    return load_model("logistic_regression.pkl")

df     = get_data()
model  = get_model()
teams  = sorted(df["team1"].dropna().unique())
venues = sorted(df["venue"].dropna().unique())

# ── NAV ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="nav">
  <div class="nav-logo">Cric<span>Predict</span></div>
  <div class="nav-tag">IPL 2026</div>
</div>
""", unsafe_allow_html=True)

# ── FORM ───────────────────────────────────────────────────────────────────────

st.markdown('<div class="form-card"><div class="form-card-title">Match Setup</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", teams)
with col2:
    team2 = st.selectbox("Team 2", teams, index=1)

venue = st.selectbox("Venue", venues)
toss_winner = st.radio("Who won the toss?", [team1, team2], horizontal=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
predict_clicked = st.button("Predict Match Winner")
st.markdown("</div>", unsafe_allow_html=True)

# ── VALIDATE ───────────────────────────────────────────────────────────────────

if team1 == team2:
    st.warning("Please select two different teams.")
    st.stop()

# ── RESULTS ────────────────────────────────────────────────────────────────────

if not predict_clicked:
    st.markdown("""
    <div class="idle-box">
      <div class="idle-icon">🏏</div>
      <div class="idle-title">Ready to predict</div>
      <div class="idle-sub">Fill in the match details above and click Predict Match Winner.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    with st.spinner("Running prediction…"):
        result = predict_match(
            model=model, model_df=df,
            team1=team1, team2=team2,
            venue=venue, toss_winner=toss_winner,
        )

    t1p  = result["team1_probability"]
    t2p  = result["team2_probability"]
    win  = result["predicted_winner"]
    t1s  = result["team1_stats"]
    t2s  = result["team2_stats"]
    h2h  = result["h2h_stats"]
    vs   = result["venue_stats"]
    wp   = t1p if win == team1 else t2p
    loser = team2 if win == team1 else team1

    # Winner
    st.markdown(f"""
    <div class="winner-card">
      <div>
        <div class="winner-label">Predicted Winner</div>
        <div class="winner-name">{win}</div>
        <div class="winner-sub">vs {loser} &nbsp;·&nbsp; {venue}</div>
      </div>
      <div>
        <div class="winner-prob">{wp:.1f}%</div>
        <div class="winner-prob-label">win probability</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Prob bars
    st.markdown(f"""
    <div class="section-card">
      <div class="section-card-title">Win Probability</div>
      <div class="pbar-row">
        <div class="pbar-name">{team1}</div>
        <div class="pbar-track"><div class="pbar-fill-t1" style="width:{t1p}%"></div></div>
        <div class="pbar-val">{t1p:.1f}%</div>
      </div>
      <div class="pbar-row">
        <div class="pbar-name">{team2}</div>
        <div class="pbar-track"><div class="pbar-fill-t2" style="width:{t2p}%"></div></div>
        <div class="pbar-val">{t2p:.1f}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats + H2H
    t1h = h2h["team1_h2h"] * 100
    t2h = h2h["team2_h2h"] * 100

    st.markdown(f"""
    <div class="two-col">
      <div class="section-card" style="margin-bottom:0">
        <div class="section-card-title">Team Statistics</div>
        <div class="team-legend">
          <span><span class="ldot" style="background:#4f8ef7"></span>{team1}</span>
          <span><span class="ldot" style="background:#4a5060"></span>{team2}</span>
        </div>
        <div class="stat-row">
          <span class="stat-key">Elo Rating</span>
          <div class="stat-vals">
            <span class="stat-v1">{t1s['elo']:.0f}</span>
            <span class="stat-v2">{t2s['elo']:.0f}</span>
          </div>
        </div>
        <div class="stat-row">
          <span class="stat-key">Form Score</span>
          <div class="stat-vals">
            <span class="stat-v1">{t1s['form']:.2f}</span>
            <span class="stat-v2">{t2s['form']:.2f}</span>
          </div>
        </div>
        <div class="stat-row">
          <span class="stat-key">Avg Runs (last 5)</span>
          <div class="stat-vals">
            <span class="stat-v1">{t1s['avg_runs']:.1f}</span>
            <span class="stat-v2">{t2s['avg_runs']:.1f}</span>
          </div>
        </div>
        <div class="stat-row">
          <span class="stat-key">Avg Wickets (last 5)</span>
          <div class="stat-vals">
            <span class="stat-v1">{t1s['avg_wickets']:.1f}</span>
            <span class="stat-v2">{t2s['avg_wickets']:.1f}</span>
          </div>
        </div>
      </div>

      <div class="section-card" style="margin-bottom:0">
        <div class="section-card-title">Head to Head</div>
        <div style="font-size:0.78rem;color:#404555;margin-bottom:16px;">Historical win rate between these two teams</div>
        <div class="h2h-row">
          <div class="h2h-name">{team1}</div>
          <div class="h2h-track"><div class="h2h-fill-t1" style="width:{t1h:.0f}%"></div></div>
          <div class="h2h-pct">{t1h:.0f}%</div>
        </div>
        <div class="h2h-row">
          <div class="h2h-name">{team2}</div>
          <div class="h2h-track"><div class="h2h-fill-t2" style="width:{t2h:.0f}%"></div></div>
          <div class="h2h-pct">{t2h:.0f}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Venue
    st.markdown(f"""
    <div class="section-card">
      <div class="section-card-title">Venue Information</div>
      <div class="kv-row">
        <span class="kv-key">Ground</span>
        <span class="kv-val">{venue}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Average first innings score</span>
        <span class="kv-val">{vs['venue_avg_score']:.0f}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Matches played here</span>
        <span class="kv-val">{vs['venue_match_count']}</span>
      </div>
      <div class="kv-row">
        <span class="kv-key">Toss won by</span>
        <span class="kv-val">{toss_winner}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)