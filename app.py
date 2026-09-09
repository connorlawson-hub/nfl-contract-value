"""
NFL Contract Value — Streamlit entry point.

The site has two separate sections, chosen from the sidebar:
  Players — one player at a time, ranked against his position group
  Teams   — whole front offices, ranked on how well they spent the cap

Run locally:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="NFL Contract Value", page_icon="🏈", layout="wide")

players_page = st.Page("views/players.py", title="Player Value",
                       icon=":material/person:", default=True)
teams_page = st.Page("views/teams.py", title="Team Efficiency",
                     icon=":material/groups:")

nav = st.navigation({"Analysis": [players_page, teams_page]})
nav.run()
