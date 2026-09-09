"""
PLAYER VALUE — one player at a time, ranked against his own position group.

This page is about individual contracts. The Team Efficiency page rolls these
same scores up to the front-office level.
"""
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from shared import (OFFENSE, DEFENSE, POS, NEG, INK, GRID, value_color,
                    load_players, sidebar_footer)

df = load_players()

st.title("Player Contract Value")
st.caption(
    "Each player is ranked against others in his **own position group in the same "
    "season** — once on production, once on what he costs against the cap. The "
    "**value score** is the gap. +40 means he produced 40 percentile points better "
    "than he was paid."
)

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.caption("**Viewing individual players.** Switch to Team Efficiency above "
               "for front-office rankings.")
    st.header("Player filters")
    season = st.selectbox("Season", sorted(df.season.unique(), reverse=True))

    side = st.radio("Side of ball", ["Offense", "Defense", "Both"], index=0)
    pool = OFFENSE if side == "Offense" else DEFENSE if side == "Defense" else OFFENSE + DEFENSE
    groups = st.multiselect("Position groups", pool, default=pool)

    st.divider()
    mode = st.radio(
        "Judge players on", ["Season totals", "Per game"],
        help="Season totals = what the team actually got for the money, so missed "
             "games count against the player. Per game = how good he was when he "
             "played, which separates injury from decline.")
    RATE = mode == "Per game"
    sfx = "_rate" if RATE else ""

    st.divider()
    min_avail = st.slider("Minimum availability (% of team games played)", 0, 100, 0, 5)
    cap_min, cap_max = st.slider("Cap hit ($M)", 0.0, 60.0, (0.0, 60.0), 0.5)
    hide_low_conf = st.checkbox(
        "Hide low-confidence groups (CB, S)", value=False,
        help="Box score stats don't capture coverage. See the note below the table.")
    search = st.text_input("Search player")
    sidebar_footer()

VS, PP, CR, VD = f"value_score{sfx}", f"production_pct{sfx}", f"cap_pct_rank{sfx}", f"verdict{sfx}"
QUAL = "qualified_rate" if RATE else "qualified"

d = df[(df.season == season) & (df.pos_group.isin(groups)) & df[QUAL]]
d = d[d.cap_number.between(cap_min, cap_max)]
d = d[d.availability_pct >= min_avail]
if hide_low_conf:
    d = d[d.confidence != "Low"]
if search:
    d = d[d.player.str.contains(search, case=False, na=False)]

tab1, tab2, tab3, tab4 = st.tabs(
    ["Rankings", "Production vs pay", "Injury impact", "Player detail"])

# ----------------------------------------------------------------- rankings
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players shown", len(d))
    c2.metric("Median cap hit", f"${d.cap_number.median():.1f}M" if len(d) else "—")
    c3.metric("Total cap charged", f"${d.cap_number.sum():.0f}M" if len(d) else "—")
    c4.metric("Median availability", f"{d.availability_pct.median():.0f}%" if len(d) else "—")

    order = st.radio("Sort", ["Best value first", "Worst value first"],
                     horizontal=True, label_visibility="collapsed")
    dd = d.sort_values(VS, ascending=(order == "Worst value first"))

    cols = {"player": "Player", "pos_group": "Pos", "team": "Team", "games": "G",
            "availability_pct": "Avail %", "cap_number": "Cap hit ($M)",
            "cap_pct_display": "% of cap", PP: "Production %ile", CR: "Pay %ile",
            VS: "Value score", "cap_surplus" + sfx: "Surplus ($M)", VD: "Verdict",
            "contract_year": "Yr of deal", "confidence": "Confidence"}
    cols = {k: v for k, v in cols.items() if k in dd.columns}

    st.dataframe(
        dd[list(cols)].rename(columns=cols).style
          .map(value_color, subset=["Value score"])
          .format({"Cap hit ($M)": "{:.1f}", "% of cap": "{:.2f}", "Avail %": "{:.0f}",
                   "Production %ile": "{:.0f}", "Pay %ile": "{:.0f}",
                   "Value score": "{:+.0f}", "Surplus ($M)": "{:+.1f}",
                   "Yr of deal": "{:.0f}", "G": "{:.0f}"}),
        use_container_width=True, height=560, hide_index=True)

    st.caption("**Surplus (\\$M)** is the same gap in dollars: what a player at this "
               "production level typically costs, minus what he actually costs.")

    if len(d[d.confidence == "Low"]):
        st.warning(
            "**Read cornerback and safety rankings carefully.** Box score stats only "
            "record what a defender *does*, not what he prevents. A corner nobody "
            "throws at posts almost nothing, so a low score here often means "
            "'avoided', not 'bad'. Interceptions and passes defended also require "
            "the ball to come your way, which is largely outside the player's control.")

    st.download_button("Download as CSV", dd[list(cols)].to_csv(index=False),
                       f"nfl_player_value_{season}.csv", "text/csv")

# ------------------------------------------------------------------ scatter
with tab2:
    st.caption("Anything **above** the diagonal produces more than it costs. Below it, less.")
    if len(d):
        base = alt.Chart(d).encode(
            x=alt.X(f"{CR}:Q", title="Pay percentile (within position group)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(gridColor=GRID, gridOpacity=0.4)),
            y=alt.Y(f"{PP}:Q", title="Production percentile",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(gridColor=GRID, gridOpacity=0.4)))
        pts = base.mark_circle(size=110, opacity=0.8, stroke="white", strokeWidth=1).encode(
            color=alt.Color(f"{VS}:Q", title="Value",
                            scale=alt.Scale(range=[NEG, "#f0efec", POS], domain=[-70, 0, 70])),
            tooltip=["player", "pos_group", "team",
                     alt.Tooltip("cap_number:Q", title="Cap $M", format=".1f"),
                     alt.Tooltip("availability_pct:Q", title="Avail %", format=".0f"),
                     alt.Tooltip(f"{VS}:Q", title="Value", format="+.0f")])
        diag = alt.Chart(pd.DataFrame({"x": [0, 100]})).mark_line(
            strokeDash=[6, 4], color=INK, size=1).encode(x="x:Q", y="x:Q")
        st.altair_chart((pts + diag).properties(height=520), use_container_width=True)

# ------------------------------------------------------------ injury impact
with tab3:
    st.subheader("How much of a bad contract year was injury?")
    st.caption(
        "**Injury gap** = per-game value score minus season-total value score. "
        "A large positive gap means the player was good when he played but wasn't "
        "available. A gap near zero means the production simply wasn't there.")

    inj = df[(df.season == season) & (df.pos_group.isin(groups))
             & df.qualified & df.qualified_rate].copy()
    if hide_low_conf:
        inj = inj[inj.confidence != "Low"]

    ren = {"player": "Player", "pos_group": "Pos", "games": "G", "games_missed": "Missed",
           "cap_number": "Cap $M", "value_score": "Totals",
           "value_score_rate": "Per game", "injury_gap": "Gap"}
    left, right = st.columns(2)
    with left:
        st.markdown("**Hurt, not declining** — biggest injury gaps")
        st.dataframe(
            inj.nlargest(12, "injury_gap")[
                ["player", "pos_group", "games", "games_missed", "cap_number",
                 "value_score", "value_score_rate", "injury_gap"]].rename(columns=ren).round(0),
            use_container_width=True, hide_index=True, height=440)
    with right:
        st.markdown("**Genuinely underperforming** — played nearly every game, still bad value")
        st.dataframe(
            inj[inj.games_missed <= 1].nsmallest(12, "value_score")[
                ["player", "pos_group", "games", "cap_number", "value_score",
                 "value_score_rate"]].rename(columns=ren).round(0),
            use_container_width=True, hide_index=True, height=440)

# ------------------------------------------------------------ player detail
with tab4:
    names = sorted(df[df.pos_group.isin(groups)].player.dropna().unique())
    idx = names.index(d.iloc[0].player) if (search and len(d) and d.iloc[0].player in names) else 0
    who = st.selectbox("Player", names, index=idx)
    h = df[df.player == who].sort_values("season")

    if len(h):
        cur = h[h.season == season]
        cur = cur.iloc[0] if len(cur) else h.iloc[-1]
        a, b, c, e, f = st.columns(5)
        a.metric("Cap hit", f"${cur.cap_number:.1f}M", f"{cur.cap_pct_display:.2f}% of cap")
        b.metric("Production %ile", f"{cur[PP]:.0f}" if pd.notna(cur[PP]) else "—")
        c.metric("Pay %ile", f"{cur[CR]:.0f}" if pd.notna(cur[CR]) else "—")
        e.metric("Value score", f"{cur[VS]:+.0f}" if pd.notna(cur[VS]) else "—", str(cur[VD]))
        f.metric("Availability", f"{cur.availability_pct:.0f}%",
                 f"{int(cur.games_missed)} games missed" if cur.games_missed else "full season")

        if pd.notna(cur.year_signed):
            st.caption(
                f"**{cur.pos_group}** · signed {int(cur.year_signed)} · "
                f"{int(cur.contract_years)}-year deal worth \\${cur.contract_value:.1f}M "
                f"(\\${cur.apy:.1f}M/yr, \\${cur.guaranteed:.1f}M guaranteed) · "
                f"year {int(cur.contract_year)} of the deal.")
        st.info(f"**Confidence for {cur.pos_group}: {cur.confidence}.** {cur.confidence_note}")

        st.subheader("Value across the contract")
        st.caption("Where you can watch a deal turn from a bargain into a cap problem.")
        long = h.melt(id_vars=["season"], value_vars=["value_score", "value_score_rate"],
                      var_name="basis", value_name="score").dropna(subset=["score"])
        long["basis"] = long["basis"].map({"value_score": "Season totals",
                                           "value_score_rate": "Per game"})
        line = alt.Chart(long).mark_line(point=True, size=2).encode(
            x=alt.X("season:O", title="Season"),
            y=alt.Y("score:Q", title="Value score",
                    axis=alt.Axis(gridColor=GRID, gridOpacity=0.4)),
            color=alt.Color("basis:N", title="",
                            scale=alt.Scale(range=[POS, "#eb6834"])),
            tooltip=["season", "basis", alt.Tooltip("score:Q", format="+.0f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=INK).encode(y="y:Q")
        st.altair_chart((line + zero).properties(height=300), use_container_width=True)

        st.dataframe(
            h[["season", "team", "games", "availability_pct", "cap_number",
               "production_pct", "cap_pct_rank", "value_score", "value_score_rate", "verdict"]]
            .rename(columns={"season": "Season", "team": "Team", "games": "G",
                             "availability_pct": "Avail %", "cap_number": "Cap $M",
                             "production_pct": "Production %ile", "cap_pct_rank": "Pay %ile",
                             "value_score": "Value (totals)",
                             "value_score_rate": "Value (per game)", "verdict": "Verdict"})
            .round(0), use_container_width=True, hide_index=True)
