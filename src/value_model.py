"""
Step 2: score every player-season.

The idea in one sentence: rank a player's production against players in the
SAME position group in the SAME season, rank his cap charge the same way, and
the value score is the gap between the two ranks.

Everything is scored twice:
  * TOTALS  -- season sums. What the team actually got for the money.
  * PER GAME -- rates. How good he was when he was on the field.
A player who missed half the year looks bad on totals and fine on rates, and
the difference between those two numbers IS the injury story.

Run:  python src/value_model.py
Output: data/value_scores.parquet
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# ---------------------------------------------------------------------------
# Production recipes: (column, weight, kind)
#   kind "vol"  -> a counting stat. Divided by games in per-game mode.
#   kind "rate" -> already an efficiency measure. Used unchanged in both modes.
#
# Two things the weights are trying to balance: doing a lot (volume) and doing
# it well (efficiency). Weighting only efficiency makes a backup with 30 snaps
# look like an All-Pro; weighting only volume rewards whoever's team throws most.
# ---------------------------------------------------------------------------
RECIPE = {
    "QB": [("passing_epa", 0.35, "vol"), ("passing_yards", 0.15, "vol"),
           ("passing_tds", 0.15, "vol"), ("passing_first_downs", 0.10, "vol"),
           ("epa_per_dropback", 0.15, "rate"), ("rushing_epa", 0.05, "vol"),
           ("passing_interceptions", -0.05, "vol")],

    # NOTE: rushing EPA is used PER CARRY, never as a season total. See the
    # comment block below -- this is the single biggest trap in the dataset.
    "RB": [("rushing_yards", 0.22, "vol"), ("rushing_first_downs", 0.13, "vol"),
           ("rushing_tds", 0.12, "vol"), ("epa_per_carry", 0.18, "rate"),
           ("receiving_yards", 0.15, "vol"), ("receiving_epa", 0.15, "vol"),
           ("receiving_tds", 0.05, "vol")],

    "WR": [("receiving_epa", 0.30, "vol"), ("receiving_yards", 0.25, "vol"),
           ("receiving_first_downs", 0.15, "vol"), ("receiving_tds", 0.15, "vol"),
           ("target_share", 0.08, "rate"), ("epa_per_target", 0.07, "rate")],

    "TE": [("receiving_epa", 0.30, "vol"), ("receiving_yards", 0.25, "vol"),
           ("receiving_first_downs", 0.15, "vol"), ("receiving_tds", 0.15, "vol"),
           ("target_share", 0.08, "rate"), ("epa_per_target", 0.07, "rate")],

    # Edge rushers are the one defensive group the box score describes well:
    # pressure production shows up directly as sacks and hits.
    "EDGE": [("def_sacks", 0.30, "vol"), ("def_qb_hits", 0.20, "vol"),
             ("def_tackles_for_loss", 0.15, "vol"), ("def_tackles_solo", 0.10, "vol"),
             ("def_fumbles_forced", 0.10, "vol"), ("pressure_per_game", 0.10, "rate"),
             ("def_pass_defended", 0.03, "vol"), ("def_tds", 0.02, "vol")],

    "DL": [("def_tackles_for_loss", 0.22, "vol"), ("def_sacks", 0.20, "vol"),
           ("def_qb_hits", 0.16, "vol"), ("def_tackles_solo", 0.18, "vol"),
           ("def_tackle_assists", 0.10, "vol"), ("def_fumbles_forced", 0.08, "vol"),
           ("stops_per_game", 0.06, "rate")],

    "LB": [("def_tackles_solo", 0.24, "vol"), ("def_tackles_for_loss", 0.16, "vol"),
           ("def_sacks", 0.14, "vol"), ("def_pass_defended", 0.12, "vol"),
           ("def_interceptions", 0.10, "vol"), ("def_qb_hits", 0.08, "vol"),
           ("def_fumbles_forced", 0.08, "vol"), ("stops_per_game", 0.08, "rate")],

    # CB and S are scored with real caveats -- see CONFIDENCE below.
    "CB": [("def_pass_defended", 0.34, "vol"), ("def_interceptions", 0.24, "vol"),
           ("def_tackles_solo", 0.16, "vol"), ("def_tds", 0.08, "vol"),
           ("def_interception_yards", 0.06, "vol"), ("def_fumbles_forced", 0.06, "vol"),
           ("def_tackles_for_loss", 0.06, "vol")],

    "S": [("def_tackles_solo", 0.26, "vol"), ("def_pass_defended", 0.22, "vol"),
          ("def_interceptions", 0.18, "vol"), ("def_tackles_for_loss", 0.10, "vol"),
          ("def_fumbles_forced", 0.08, "vol"), ("def_sacks", 0.08, "vol"),
          ("def_tds", 0.04, "vol"), ("def_tackle_assists", 0.04, "vol")],
}

# How much to trust the ranking for each group, and why. Shown in the UI so the
# numbers aren't taken at face value where the underlying data is weak.
CONFIDENCE = {
    "QB":   ("High", "Passing EPA captures quarterback play well."),
    "RB":   ("High", "Rushing and receiving production is fully counted."),
    "WR":   ("High", "Targets, yards and receiving EPA capture the job."),
    "TE":   ("High", "Blocking isn't measured, so run-first tight ends rate low."),
    "EDGE": ("High", "Sacks, QB hits and tackles for loss describe pass rush well."),
    "DL":   ("Medium", "Run-stuffing interior linemen absorb blocks without recording stats."),
    "LB":   ("Medium", "Tackle totals depend heavily on scheme and how bad the defense is."),
    "CB":   ("Low", "A corner nobody throws at records almost nothing. "
                    "Low scores here often mean 'avoided', not 'bad'."),
    "S":    ("Low", "Coverage safeties are largely invisible in box score stats. "
                    "Deep safeties will rank below box safeties who make tackles."),
}

# Minimum usage to be ranked, as (season total, per game) pairs.
QUALIFY = {
    "QB":   ("attempts", 150, 12), "RB": ("touches", 50, 4),
    "WR":   ("targets", 25, 1.8),  "TE": ("targets", 20, 1.4),
    "EDGE": ("def_snaps_or_tackles", 200, 12),
    "DL":   ("def_snaps_or_tackles", 200, 12),
    "LB":   ("def_snaps_or_tackles", 200, 12),
    "CB":   ("def_snaps_or_tackles", 200, 12),
    "S":    ("def_snaps_or_tackles", 200, 12),
}
MIN_GAMES_RATE = 4          # below this, per-game rates are noise


def _z(s):
    """Z-score, clipped at +/-3 so one freak outlier can't dominate the blend."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.fillna(s.median() if s.notna().any() else 0.0)
    sd = s.std(ddof=0)
    return pd.Series(0.0, index=s.index) if sd == 0 else ((s - s.mean()) / sd).clip(-3, 3)


def _pct(s):
    return s.rank(pct=True) * 100.0


def add_derived(df):
    """Efficiency stats and the usage measures qualification depends on."""
    df = df.copy()
    z = lambda c: pd.to_numeric(df.get(c), errors="coerce").fillna(0)

    def per(num, den, floor):
        d = z(den)
        return np.where(d >= floor, z(num) / d.replace(0, np.nan), np.nan)

    df["epa_per_dropback"] = per("passing_epa", "attempts", 100)
    df["epa_per_carry"] = per("rushing_epa", "carries", 25)
    df["epa_per_target"] = per("receiving_epa", "targets", 15)
    df["touches"] = z("carries") + z("receptions")
    df["tackles_total"] = z("def_tackles_solo") + z("def_tackle_assists")
    g = df["games"].clip(lower=1)
    df["pressure_per_game"] = (z("def_sacks") + 0.5 * z("def_qb_hits")) / g
    df["stops_per_game"] = (z("def_tackles_for_loss") + z("def_sacks")
                            + 0.5 * z("def_tackles_solo")) / g
    # Snap counts are the best availability signal on defense, but coverage is
    # patchy in older seasons, so fall back to tackle volume.
    df["def_snaps_or_tackles"] = np.where(
        df.get("defense_snaps", pd.Series(np.nan, index=df.index)).notna(),
        df.get("defense_snaps"), df["tackles_total"] * 12)
    return df


def mark_qualified(df):
    df = df.copy()
    df["qualified"] = False
    df["qualified_rate"] = False
    for pos, (col, tot, per_g) in QUALIFY.items():
        m = df.pos_group == pos
        if not m.any():
            continue
        v = pd.to_numeric(df.loc[m, col], errors="coerce").fillna(0)
        df.loc[m, "qualified"] = v >= tot
        df.loc[m, "qualified_rate"] = ((v / df.loc[m, "games"].clip(lower=1)) >= per_g) \
                                      & (df.loc[m, "games"] >= MIN_GAMES_RATE)
    return df


def score_group(g, pos, mode):
    """Score one position group within one season, in 'total' or 'rate' mode."""
    g = g.copy()
    games = g["games"].clip(lower=1)
    parts = []
    for col, wt, kind in RECIPE[pos]:
        if col not in g.columns:
            continue
        v = pd.to_numeric(g[col], errors="coerce")
        if mode == "rate" and kind == "vol":
            v = v / games          # counting stats become per-game
        parts.append(wt * _z(v))
    raw = sum(parts) if parts else pd.Series(0.0, index=g.index)

    qual = g["qualified"] if mode == "total" else g["qualified_rate"]
    prod = pd.Series(np.nan, index=g.index)
    prod[qual] = _pct(raw[qual])
    cap_rank = pd.Series(np.nan, index=g.index)
    cap_rank[qual] = _pct(g.loc[qual, "cap_percent"])

    suffix = "" if mode == "total" else "_rate"
    g[f"production_pct{suffix}"] = prod
    g[f"cap_pct_rank{suffix}"] = cap_rank
    g[f"value_score{suffix}"] = prod - cap_rank

    # A dollar reading of the same gap: what a player at this production level
    # typically costs, minus what he actually costs.
    #
    # This has to be a SMOOTHED curve, not a straight interpolation through
    # every player. Interpolating through the raw points makes each player's
    # own salary the expected salary, so the surplus is always exactly zero --
    # which is what the first version of this did. Binning into deciles and
    # taking the median cap charge in each gives a real market curve.
    g[f"expected_cap_percent{suffix}"] = np.nan
    g[f"cap_surplus{suffix}"] = np.nan
    if qual.sum() >= 12:
        sub = g.loc[qual, [f"production_pct{suffix}", "cap_percent"]].dropna()
        if len(sub) >= 12:
            bins = pd.qcut(sub[f"production_pct{suffix}"], 10,
                           labels=False, duplicates="drop")
            curve = (sub.assign(_b=bins).groupby("_b")
                        .agg(x=(f"production_pct{suffix}", "mean"),
                             y=("cap_percent", "median"))
                        .sort_values("x"))
            if len(curve) >= 3:
                exp = np.interp(prod, curve["x"].values, curve["y"].values,
                                left=curve["y"].iloc[0], right=curve["y"].iloc[-1])
                exp = pd.Series(exp, index=g.index).where(prod.notna())
                g[f"expected_cap_percent{suffix}"] = exp
                g[f"cap_surplus{suffix}"] = (exp - g["cap_percent"]) * g["salary_cap"]
    return g


def add_value_scores(df):
    df = add_derived(df)
    df = mark_qualified(df)
    df = df[df.has_contract & df.cap_percent.notna()].copy()

    # Back out each season's salary cap so surplus can be shown in dollars.
    # cap_percent is rounded to 3 decimals in the source, so small contracts give
    # a wildly imprecise estimate -- only use the big ones, where the rounding
    # error is proportionally tiny.
    big = df.cap_percent >= 0.02
    est = pd.Series(np.nan, index=df.index)
    est[big] = df.loc[big, "cap_number"] / df.loc[big, "cap_percent"]
    df["salary_cap"] = df.assign(_e=est).groupby("season")["_e"].transform("median")

    frames = []
    for (season, pos), g in df.groupby(["season", "pos_group"]):
        if pos not in RECIPE:
            continue
        g = score_group(g, pos, "total")
        g = score_group(g, pos, "rate")
        frames.append(g)
    res = pd.concat(frames, ignore_index=True)

    labels = ["Overpaid", "Slightly overpaid", "Fairly paid", "Good value", "Bargain"]
    bins = [-1000, -30, -10, 10, 30, 1000]
    for suffix in ["", "_rate"]:
        v = pd.cut(res[f"value_score{suffix}"], bins=bins, labels=labels)
        res[f"verdict{suffix}"] = (v.cat.add_categories(["Not enough usage"])
                                    .fillna("Not enough usage"))

    res["confidence"] = res.pos_group.map(lambda p: CONFIDENCE[p][0])
    res["confidence_note"] = res.pos_group.map(lambda p: CONFIDENCE[p][1])

    # The availability story: how much of the gap is explained by missing time.
    res["availability_pct"] = res["availability"] * 100
    res["injury_gap"] = res["value_score_rate"] - res["value_score"]
    res["missed_time"] = res["games_missed"] >= 3
    return res


def main():
    df = pd.read_parquet(os.path.join(DATA, "player_seasons.parquet"))
    res = add_value_scores(df)
    dest = os.path.join(DATA, "value_scores.parquet")
    res.to_parquet(dest, index=False)
    print(f"wrote {dest}  ({len(res):,} rows)")
    print(res.groupby("pos_group")[["qualified", "qualified_rate"]].sum().to_string())


if __name__ == "__main__":
    main()
