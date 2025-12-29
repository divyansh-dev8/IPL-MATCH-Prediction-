import streamlit as st
import pandas as pd
from ipl_prediction import predict_match

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="IPL Match Winner Prediction",
    page_icon="🏏",
    layout="centered"
)

# =============================
# BACKGROUND IMAGE (EXTERNAL LINK)
# =============================
def set_bg_from_url(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 🔴 YAHI LINK CHANGE KARNA HAI (DIRECT .jpg/.png)
set_bg_from_url(
    "https://4kwallpapers.com/images/wallpapers/ipl-2021-ipl-t20-indian-premier-league-cricket-ipl-2021-2560x1440-4994.png"
)

# =============================
# TITLE
# =============================
st.title("🏏 IPL Match Winner Prediction")
st.caption("Predict IPL match winner using Machine Learning")
st.divider()

# =============================
# LOAD DATA (CSV)
# =============================
df = pd.read_csv("data/matches.csv")

teams = sorted(pd.unique(df[['team1', 'team2']].values.ravel()))
venues = sorted(df['venue'].dropna().unique())
seasons = sorted(df['season'].astype(str).str[:4].astype(int).unique())

# =============================
# INPUTS
# =============================
col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Select Team 1", teams)

with col2:
    team2 = st.selectbox("Select Team 2", teams)

toss = st.radio(
    "Toss Decision",
    ["bat", "field"],
    horizontal=True
)

venue = st.selectbox("Select Venue", venues)
season = st.selectbox("Select Season", seasons)

st.divider()

# =============================
# PREDICTION
# =============================
if st.button("Predict Winner"):
    if team1 == team2:
        st.warning("Team 1 and Team 2 cannot be same")
    else:
        result = predict_match(team1, team2, toss, venue, season)

        # SUPPORT BOTH RETURN TYPES
        if isinstance(result, tuple):
            winner, probability = result
            st.success(f"🏆 Predicted Winner: **{winner}**")
            st.info(f"Winning Probability: **{probability:.2f}%**")
        else:
            winner = team1 if result == 1 else team2
            st.success(f"🏆 Predicted Winner: **{winner}**")