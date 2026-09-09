"""
Shared setup for both pages: data loading, palette, and the value-score colouring.

Kept in one place so the Players and Teams pages can't drift apart visually.
"""
import os
import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
PLAYER_DATA = os.path.join(_HERE, "data", "value_scores.parquet")
TEAM_DATA = os.path.join(_HERE, "data", "team_seasons.parquet")

# Chart palette. Blue (good) <-> red (bad) with a neutral grey midpoint, rather
# than the obvious red-to-green: red vs green is the single worst pair for
# colourblind readers, and blue vs red separates cleanly for every kind of
# colour vision (validated: CVD delta-E 21.6 protan, 34.5 tritan).
POS, NEG, INK, GRID = "#2a78d6", "#e34948", "#52514e", "#c3c2b7"
DIVERGING = "blueorange"          # Altair scheme used for continuous value scales

OFFENSE = ["QB", "RB", "WR", "TE"]
DEFENSE = ["EDGE", "DL", "LB", "CB", "S"]

_STOPS = [(-70, (227, 73, 72)), (0, (240, 239, 236)), (70, (42, 120, 214))]


def value_color(v):
    """
    Background colour for a value score, red through neutral to blue.

    Hand-rolled instead of Styler.background_gradient, which pulls in matplotlib
    (~50MB) purely to interpolate three colours.
    """
    if pd.isna(v):
        return ""
    v = max(-70, min(70, float(v)))
    for (x0, c0), (x1, c1) in zip(_STOPS, _STOPS[1:]):
        if x0 <= v <= x1:
            t = 0 if x1 == x0 else (v - x0) / (x1 - x0)
            r, g, b = (round(a + (bb - a) * t) for a, bb in zip(c0, c1))
            break
    else:
        r, g, b = _STOPS[-1][1]
    luma = 0.299 * r + 0.587 * g + 0.114 * b          # dark fills need light text
    return f"background-color: rgb({r},{g},{b}); color: {'#111' if luma > 150 else '#fff'};"


@st.cache_data
def load_players():
    df = pd.read_parquet(PLAYER_DATA)
    df["cap_pct_display"] = df["cap_percent"] * 100
    return df


@st.cache_data
def load_teams():
    return pd.read_parquet(TEAM_DATA)


def sidebar_footer():
    st.divider()
    st.caption("Data: nflverse (stats, snap counts, schedules) + OverTheCap "
               "contracts, joined on gsis_id.")
