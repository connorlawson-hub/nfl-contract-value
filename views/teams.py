"""
TEAM EFFICIENCY — whole front offices, ranked on how well they spent the cap.

This page aggregates the same player value scores from the Player Value page,
weighted by each player's share of his team's cap.
"""
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from shared import POS, NEG, INK, GRID, value_color, load_teams, sidebar_footer

teams = load_teams()
hist = teams.dropna(subset=["efficiency", "wins"])

st.title("Team Contract Efficiency")
st.caption(
    "This page is about **front offices, not players.** Every player's value score "
    "is averaged into his team's, weighted by his share of the cap — so a bad "
    "\\$40M contract counts far more than a bad \\$2M one — then centred on the "
    "league average for that season. **0 = an average front office that year.**"
)

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.caption("**Viewing whole teams.** Switch to Player Value above for "
               "individual contracts.")
    st.header("Team filters")
    season = st.selectbox("Season", sorted(teams.season.unique(), reverse=True))
    all_seasons = st.checkbox(
        "Plot all seasons at once", value=False,
        help=f"Charts use every team-season from {int(hist.season.min())} to "
             f"{int(hist.season.max())} instead of just the one selected. The "
             "ranking table always shows the selected season.")
    compare = st.radio("Compare efficiency against", ["Wins", "Point differential"])
    sidebar_footer()

ycol, ytitle = ("wins", "Wins") if compare == "Wins" else ("point_diff", "Point differential")
ts = teams[teams.season == season].copy()
plot = hist if all_seasons else hist[hist.season == season]

tabA, tabB, tabC = st.tabs(
    ["League table", "Efficiency vs record", "Does it predict?"])

# -------------------------------------------------------------- league table
with tabA:
    if len(ts):
        best, worst = ts.nlargest(1, "efficiency").iloc[0], ts.nsmallest(1, "efficiency").iloc[0]
        r_now = np.corrcoef(ts.efficiency, ts.wins)[0, 1]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Most efficient", best.team, f"{best.efficiency:+.1f} · {best.record}")
        k2.metric("Least efficient", worst.team, f"{worst.efficiency:+.1f} · {worst.record}")
        k3.metric(f"Efficiency vs wins, {season}", f"r = {r_now:+.2f}")
        k4.metric("Cap graded", f"{ts.cap_coverage.median():.0f}%",
                  "of each team's cap", delta_color="off")

    st.markdown("##### Every team, ranked")
    order = ts.sort_values("efficiency", ascending=False).team.tolist()
    bars = alt.Chart(ts).mark_bar(cornerRadiusEnd=4, height=14).encode(
        y=alt.Y("team:N", sort=order, title=None,
                axis=alt.Axis(labelFontSize=11, grid=False, labelOverlap=False,
                              labelPadding=6, tickCount=32)),
        x=alt.X("efficiency:Q", title="Efficiency vs league average that season",
                axis=alt.Axis(gridColor=GRID, gridOpacity=0.4)),
        color=alt.condition(alt.datum.efficiency > 0, alt.value(POS), alt.value(NEG)),
        tooltip=[alt.Tooltip("team:N", title="Team"),
                 alt.Tooltip("record:N", title="Record"),
                 alt.Tooltip("efficiency:Q", title="Efficiency", format="+.1f"),
                 alt.Tooltip("point_diff:Q", title="Point diff", format="+.0f"),
                 alt.Tooltip("best_deal:N", title="Best deal"),
                 alt.Tooltip("worst_deal:N", title="Worst deal")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK, size=1).encode(x="x:Q")
    st.altair_chart((bars + zero).properties(height=560), use_container_width=True)

    st.markdown(f"##### {season} in full")
    tcols = {"efficiency_rank": "Rank", "team": "Team", "record": "Record",
             "wins": "W", "point_diff": "Pt diff", "efficiency": "Efficiency",
             "n_bargains": "Bargains", "n_overpaid": "Overpays",
             "best_deal": "Best deal", "worst_deal": "Worst deal",
             "scored_cap": "Cap graded ($M)", "cap_coverage": "% of cap graded"}
    tt = ts.sort_values("efficiency", ascending=False)[list(tcols)].rename(columns=tcols)
    st.dataframe(
        tt.style.map(value_color, subset=["Efficiency"])
          .format({"Efficiency": "{:+.1f}", "Pt diff": "{:+.0f}",
                   "Cap graded ($M)": "{:.0f}", "% of cap graded": "{:.0f}%",
                   "W": "{:.0f}"}),
        use_container_width=True, height=520, hide_index=True)

    st.caption(
        f"**A real limitation:** only about **{ts.cap_coverage.median():.0f}% of each "
        "team's salary cap** is graded here. Offensive linemen, kickers and punters "
        "aren't scored at all, and neither are players below the usage minimums. A "
        "team that spent heavily on a great offensive line gets no credit for it.")
    st.download_button("Download team data as CSV", tt.to_csv(index=False),
                       f"nfl_team_efficiency_{season}.csv", "text/csv")

# --------------------------------------------------------- efficiency vs record
with tabB:
    label = (f"all {len(hist)} team-seasons, {int(hist.season.min())}–"
             f"{int(hist.season.max())}") if all_seasons else f"{season} only"
    st.markdown(f"##### Contract efficiency against {ytitle.lower()} — {label}")
    st.caption(
        ("Each dot is one team in one season. " if all_seasons
         else "Each label is a team. ")
        + "The dashed line is the trend across every season in the dataset.")

    # Team abbreviations are readable for one season (32 points) but collide into
    # mush across seven (223). Switch to dots once there are too many.
    enc = dict(
        x=alt.X("efficiency:Q", title="Contract efficiency (vs league average)",
                axis=alt.Axis(gridColor=GRID, gridOpacity=0.4)),
        y=alt.Y(f"{ycol}:Q", title=ytitle, scale=alt.Scale(zero=False),
                axis=alt.Axis(gridColor=GRID, gridOpacity=0.4, titlePadding=10)),
        color=alt.condition(alt.datum.efficiency > 0, alt.value(POS), alt.value(NEG)),
        tooltip=["team:N", "season:O", alt.Tooltip("record:N", title="Record"),
                 alt.Tooltip("efficiency:Q", format="+.1f"),
                 alt.Tooltip("point_diff:Q", format="+.0f")])
    if all_seasons:
        pts = alt.Chart(plot).mark_circle(
            size=90, opacity=0.7, stroke="white", strokeWidth=1).encode(**enc)
    else:
        pts = alt.Chart(plot).mark_text(fontSize=11, fontWeight=600).encode(
            text="team:N", **enc)
    trend = alt.Chart(hist).transform_regression("efficiency", ycol).mark_line(
        color=INK, strokeDash=[5, 4], size=2).encode(x="efficiency:Q", y=f"{ycol}:Q")
    st.altair_chart((pts + trend).properties(height=500), use_container_width=True)

    rr = np.corrcoef(hist.efficiency, hist[ycol])[0, 1]
    st.info(f"Across all seasons: **r = {rr:+.2f}** — contract efficiency explains "
            f"about **{rr**2*100:.0f}%** of the variation in {ytitle.lower()}. "
            "That's a real relationship, and it's the strongest claim this model "
            "can make.")

# ----------------------------------------------------------- does it predict?
with tabC:
    pred = teams.dropna(subset=["efficiency", "next_wins"])
    r_pred = np.corrcoef(pred.efficiency, pred.next_wins)[0, 1]
    r_base = np.corrcoef(pred.wins, pred.next_wins)[0, 1]

    st.subheader("Can this predict next season?")
    st.warning(
        f"**Mostly no — and that's the interesting part.**\n\n"
        f"This season's efficiency correlates with *next* season's wins at only "
        f"**r = {r_pred:+.2f}** (about {r_pred**2*100:.0f}% of the variation). Simply "
        f"knowing a team's record this year predicts next year better "
        f"(**r = {r_base:+.2f}**), and once you know the record, efficiency adds "
        f"essentially nothing on top of it.")

    comp = pd.DataFrame({
        "Predictor of next season's wins": [
            "This season's contract efficiency",
            "This season's wins", "This season's point differential"],
        "r": [r_pred, r_base, np.corrcoef(pred.point_diff, pred.next_wins)[0, 1]]})
    comp["Variance explained"] = (comp.r ** 2 * 100).round(0).astype(int).astype(str) + "%"
    comp["r"] = comp.r.map("{:+.2f}".format)
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown(
        "**Why it doesn't carry over: the market corrects.** A bargain is temporary "
        "by construction — good players on cheap deals get paid. Across 2019–2025:")
    c1, c2 = st.columns(2)
    c1.info("**Of players scoring +30 or better** (a bargain), only **32%** were still "
            "a bargain the next season. **75%** saw their cap charge rise, and the "
            "median more than doubled.")
    c2.info("**Of players scoring −30 or worse** (an overpay), **59%** saw their cap "
            "charge fall the next season — cut, restructured or traded away.")

    st.markdown("##### Efficiency this season against wins the following season")
    # 191 team-seasons here, so dots rather than labels -- the team is in the tooltip.
    sc = alt.Chart(pred).mark_circle(
        size=90, opacity=0.7, stroke="white", strokeWidth=1).encode(
        x=alt.X("efficiency:Q", title="Contract efficiency this season",
                axis=alt.Axis(gridColor=GRID, gridOpacity=0.4)),
        y=alt.Y("next_wins:Q", title="Wins the next season", scale=alt.Scale(zero=False),
                axis=alt.Axis(gridColor=GRID, gridOpacity=0.4, titlePadding=10)),
        color=alt.condition(alt.datum.efficiency > 0, alt.value(POS), alt.value(NEG)),
        tooltip=["team:N", "season:O", alt.Tooltip("efficiency:Q", format="+.1f"),
                 alt.Tooltip("next_wins:Q", title="Next-season wins")])
    tl = alt.Chart(pred).transform_regression("efficiency", "next_wins").mark_line(
        color=INK, strokeDash=[5, 4], size=2).encode(x="efficiency:Q", y="next_wins:Q")
    st.altair_chart((sc + tl).properties(height=440), use_container_width=True)
    st.caption(
        "Compare how flat this line is with the one on the previous tab. Team "
        "efficiency itself barely persists year to year (r = +0.15), so read this "
        "model as a scoreboard for the season that just happened and a guide to "
        "which individual contracts are working — not as a forecast.")
