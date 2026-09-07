"""
Step 3: roll player value scores up to team level and attach season results.

Team contract efficiency is a CAP-WEIGHTED average of player value scores: a bad
$40M contract has to hurt far more than a bad $2M one, so each player's score is
weighted by his share of the team's scored cap.

Run:  python src/team_model.py
Output: data/team_seasons.parquet
"""
import os
import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RAW = os.path.join(DATA, "raw")
GAMES_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
             "schedules/games.csv")


def load_games():
    p = os.path.join(RAW, "games.csv")
    if not os.path.exists(p):
        os.makedirs(RAW, exist_ok=True)
        print("  downloading games.csv ...")
        open(p, "wb").write(requests.get(GAMES_URL, timeout=180).content)
    return pd.read_csv(p, low_memory=False)


def team_records(seasons):
    """One row per team-season: wins, losses, ties, points for/against."""
    g = load_games()
    g = g[(g.game_type == "REG") & g.season.isin(seasons) & g.result.notna()]

    home = pd.DataFrame({
        "season": g.season, "team": g.home_team,
        "pf": g.home_score, "pa": g.away_score, "margin": g.result})
    away = pd.DataFrame({
        "season": g.season, "team": g.away_team,
        "pf": g.away_score, "pa": g.home_score, "margin": -g.result})
    a = pd.concat([home, away], ignore_index=True)

    a["win"] = (a.margin > 0).astype(int)
    a["loss"] = (a.margin < 0).astype(int)
    a["tie"] = (a.margin == 0).astype(int)

    rec = (a.groupby(["season", "team"])
             .agg(games=("win", "size"), wins=("win", "sum"),
                  losses=("loss", "sum"), ties=("tie", "sum"),
                  points_for=("pf", "sum"), points_against=("pa", "sum"))
             .reset_index())
    rec["point_diff"] = rec.points_for - rec.points_against
    rec["win_pct"] = (rec.wins + 0.5 * rec.ties) / rec.games
    rec["record"] = (rec.wins.astype(str) + "-" + rec.losses.astype(str)
                     + np.where(rec.ties > 0, "-" + rec.ties.astype(str), ""))
    return rec


def team_efficiency(scores):
    """Cap-weighted average value score per team-season."""
    d = scores[scores.qualified & scores.cap_number.notna()].copy()
    d["w"] = d.cap_number.clip(lower=0.01)

    def agg(g):
        w = g.w
        return pd.Series({
            "efficiency": np.average(g.value_score, weights=w),
            "efficiency_rate": np.average(
                g.value_score_rate.fillna(g.value_score), weights=w),
            "scored_cap": g.cap_number.sum(),
            "surplus_total": g.cap_surplus.sum(skipna=True),
            "n_players": len(g),
            "n_bargains": int((g.value_score >= 30).sum()),
            "n_overpaid": int((g.value_score <= -30).sum()),
            "worst_deal": g.loc[g.value_score.idxmin(), "player"],
            "worst_deal_score": g.value_score.min(),
            "best_deal": g.loc[g.value_score.idxmax(), "player"],
            "best_deal_score": g.value_score.max(),
            "avg_availability": g.availability_pct.mean(),
        })

    t = d.groupby(["season", "team"]).apply(agg, include_groups=False).reset_index()

    # How much of the salary cap do the players we score actually account for?
    # We don't grade the offensive line or specialists, so this is well short
    # of 100% and the UI needs to say so.
    cap = scores.groupby("season")["salary_cap"].median().rename("salary_cap")
    t = t.merge(cap, on="season", how="left")
    t["cap_coverage"] = (t.scored_cap / t.salary_cap * 100).clip(upper=100)
    return t


def main():
    scores = pd.read_parquet(os.path.join(DATA, "value_scores.parquet"))
    seasons = sorted(scores.season.unique())

    eff = team_efficiency(scores)
    rec = team_records(seasons)
    t = eff.merge(rec, on=["season", "team"], how="inner")

    # Centre each season on its own league average.
    #
    # This matters. value_score = production percentile - pay percentile, so a
    # player in the 95th percentile of pay can score at most +5, while one in the
    # 5th percentile can reach +95. The metric is asymmetrically bounded, and
    # cap-weighting deliberately upweights the expensive players who structurally
    # cannot score high -- so the raw team average is negative for 95% of
    # team-seasons (league mean about -14). That bias is identical for every team,
    # so the ranking is sound, but the raw zero point means nothing. Centring makes
    # 0 = league average for that season, which is what a reader assumes it means.
    # It is a per-season linear shift, so it leaves within-season correlations
    # untouched.
    t["efficiency_raw"] = t["efficiency"]
    t["league_mean"] = t.groupby("season")["efficiency"].transform("mean")
    t["efficiency"] = t["efficiency_raw"] - t["league_mean"]
    t["efficiency_rate"] = (t["efficiency_rate"]
                            - t.groupby("season")["efficiency_rate"].transform("mean"))

    # Ranks within each season, so 1 = most efficient that year.
    t["efficiency_rank"] = t.groupby("season")["efficiency"].rank(
        ascending=False, method="min").astype(int)
    t["wins_rank"] = t.groupby("season")["wins"].rank(
        ascending=False, method="min").astype(int)

    # Next season's result, for the prediction question.
    nxt = t[["season", "team", "wins", "point_diff"]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={"wins": "next_wins", "point_diff": "next_point_diff"})
    t = t.merge(nxt, on=["season", "team"], how="left")

    dest = os.path.join(DATA, "team_seasons.parquet")
    t.to_parquet(dest, index=False)
    print(f"wrote {dest}  ({len(t)} team-seasons, {t.season.min()}-{t.season.max()})")
    print(f"median cap coverage: {t.cap_coverage.median():.0f}% of the salary cap")


if __name__ == "__main__":
    main()
