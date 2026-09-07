"""
Step 1 of the pipeline: download nflverse data and build ONE clean
player-season table covering offense and defense, with contracts attached.

Run:  python src/build_data.py --seasons 2019 2025
Output: data/player_seasons.parquet
"""
import argparse, os
import numpy as np
import pandas as pd
import requests

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(HERE, "..", "data")

# The nine groups we score. Offensive line is deliberately absent: linemen have
# essentially no counting stats, so anything we produced for them would be noise.
OFFENSE = ["QB", "RB", "WR", "TE"]
DEFENSE = ["EDGE", "DL", "LB", "CB", "S"]
GROUPS = OFFENSE + DEFENSE


def fetch(url, path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  downloading {os.path.basename(path)} ...")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    open(path, "wb").write(r.content)
    return path


# ------------------------------------------------------------------ contracts
def load_contracts():
    """
    Use the .parquet asset, NOT the .csv.gz.
    The CSV in this release has been stale since May 2022; only the parquet is
    rebuilt daily. This is the single easiest way to waste an afternoon here.
    """
    p = fetch(f"{BASE}/contracts/historical_contracts.parquet",
              os.path.join(RAW, "historical_contracts.parquet"))
    return pd.read_parquet(p)


def explode_cap_hits(contracts):
    """Flatten the nested season_history array into one row per player-season."""
    rows = []
    for r in contracts.itertuples():
        sh = r.season_history
        if sh is None or len(sh) == 0:
            continue
        for d in sh:
            yr = str(d.get("year", ""))
            if not yr.isdigit():           # skips the 'Total' summary row
                continue
            rows.append({
                "gsis_id": r.gsis_id, "season": int(yr),
                "cap_number": d.get("cap_number"), "cap_percent": d.get("cap_percent"),
                "base_salary": d.get("base_salary"), "cash_paid": d.get("cash_paid"),
                "year_signed": r.year_signed, "contract_years": r.years,
                "contract_value": r.value, "apy": r.apy, "guaranteed": r.guaranteed,
                "draft_round_c": r.draft_round, "draft_year_c": r.draft_year,
            })
    cap = pd.DataFrame(rows)
    # An extension can supersede an older deal for the same season; keep the newer.
    cap = (cap.sort_values(["gsis_id", "season", "year_signed"])
              .drop_duplicates(["gsis_id", "season"], keep="last"))
    cap["contract_year"] = cap["season"] - cap["year_signed"] + 1
    cap["is_rookie_deal"] = cap["draft_year_c"] == cap["year_signed"]
    return cap.dropna(subset=["gsis_id"])


# ------------------------------------------------------- position normalisation
NGS_MAP = {
    "EDGE": "EDGE", "INTERIOR_LINE": "DL", "MLB": "LB", "OLB": "LB",
    "CB": "CB", "SLOT_CB": "CB", "SAFETY": "S", "HIGH_SAFETY": "S",
}
BASE_MAP = {
    "DE": "EDGE", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "OLB": "LB", "ILB": "LB", "MLB": "LB",
    "CB": "CB", "DB": "CB", "S": "S", "SAF": "S", "FS": "S", "SS": "S",
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
}


def assign_group(df, players):
    """
    nflverse position labels are messy: a 3-4 outside linebacker who rushes the
    passer every down is listed as 'LB', and some 'DE's play inside. NGS position
    is the best signal, so we prefer it and fall back to the base label. For
    linebackers with no NGS data we look at behaviour -- a linebacker generating
    a lot of pass rush is really an edge defender.
    """
    ngs = players.set_index("gsis_id")["ngs_position"].to_dict()
    df = df.copy()
    df["ngs_position"] = df["gsis_id"].map(ngs)
    df["pos_group"] = (df["ngs_position"].map(NGS_MAP)
                       .fillna(df["position"].map(BASE_MAP)))

    # Behavioural fallback for unlabeled front-seven players.
    if "def_sacks" in df.columns:
        rush = (df["def_sacks"].fillna(0) + 0.5 * df["def_qb_hits"].fillna(0))
        per_g = rush / df["games"].clip(lower=1)
        unlabeled = df["ngs_position"].isna() & df["position"].isin(["LB", "OLB", "ILB"])
        df.loc[unlabeled & (per_g >= 0.45), "pos_group"] = "EDGE"
    return df


# ---------------------------------------------------------------------- stats
SUM_COLS = [
    # offense
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "sacks_suffered", "passing_first_downs", "passing_epa", "carries", "rushing_yards",
    "rushing_tds", "rushing_first_downs", "rushing_epa", "receptions", "targets",
    "receiving_yards", "receiving_tds", "receiving_first_downs", "receiving_epa",
    "receiving_air_yards", "fumbles_lost_total", "fantasy_points_ppr",
    # defense
    "def_tackles_solo", "def_tackle_assists", "def_tackles_with_assist",
    "def_tackles_for_loss", "def_tackles_for_loss_yards", "def_sacks", "def_sack_yards",
    "def_qb_hits", "def_interceptions", "def_interception_yards", "def_pass_defended",
    "def_fumbles_forced", "def_tds", "def_safeties", "fumble_recovery_opp",
]
MEAN_COLS = ["target_share", "wopr", "passing_cpoe"]


def load_season_stats(seasons):
    frames = []
    for yr in seasons:
        p = fetch(f"{BASE}/stats_player/stats_player_week_{yr}.csv.gz",
                  os.path.join(RAW, f"stats_player_week_{yr}.csv.gz"))
        frames.append(pd.read_csv(p, low_memory=False))
    w = pd.concat(frames, ignore_index=True)
    w = w[w.season_type == "REG"]

    sums = [c for c in SUM_COLS if c in w.columns]
    means = [c for c in MEAN_COLS if c in w.columns]

    g = w.groupby(["player_id", "season"])
    agg = g[sums].sum(min_count=1)
    agg[means] = g[means].mean()
    agg["games"] = g["week"].nunique()
    meta = g.agg(player=("player_display_name", "last"),
                 position=("position", "last"),
                 position_group=("position_group", "last"),
                 team=("team", "last"))
    out = agg.join(meta).reset_index().rename(columns={"player_id": "gsis_id"})

    # How many games did the player's team actually play? This is the denominator
    # for availability, and it handles the 16-game seasons before 2021.
    tg = (w.groupby(["team", "season"])["week"].nunique()
            .rename("team_games").reset_index())
    primary = (w.groupby(["player_id", "season", "team"]).size()
                 .rename("n").reset_index()
                 .sort_values("n").drop_duplicates(["player_id", "season"], keep="last")
                 .rename(columns={"player_id": "gsis_id"})[["gsis_id", "season", "team"]])
    primary = primary.merge(tg, on=["team", "season"], how="left")
    out = out.merge(primary.drop(columns="team"), on=["gsis_id", "season"], how="left")
    out["team_games"] = out["team_games"].fillna(17)
    out["games_missed"] = (out["team_games"] - out["games"]).clip(lower=0)
    out["availability"] = (out["games"] / out["team_games"]).clip(upper=1.0)
    return out


def load_snaps(seasons):
    frames = []
    for yr in seasons:
        try:
            p = fetch(f"{BASE}/snap_counts/snap_counts_{yr}.csv.gz",
                      os.path.join(RAW, f"snap_counts_{yr}.csv.gz"))
            frames.append(pd.read_csv(p, low_memory=False))
        except Exception as e:
            print(f"  (no snap counts for {yr}: {e})")
    if not frames:
        return pd.DataFrame(columns=["pfr_id", "season"])
    s = pd.concat(frames, ignore_index=True)
    s = s[s.game_type == "REG"]
    return (s.groupby(["pfr_player_id", "season"])
              .agg(offense_snaps=("offense_snaps", "sum"),
                   defense_snaps=("defense_snaps", "sum"),
                   offense_pct=("offense_pct", "mean"),
                   defense_pct=("defense_pct", "mean"))
              .reset_index().rename(columns={"pfr_player_id": "pfr_id"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs=2, type=int, default=[2019, 2025])
    args = ap.parse_args()
    seasons = list(range(args.seasons[0], args.seasons[1] + 1))

    print("1/5 contracts")
    cap = explode_cap_hits(load_contracts())
    cap = cap[cap.season.isin(seasons)]

    print("2/5 weekly stats -> season totals")
    stats = load_season_stats(seasons)

    print("3/5 id crosswalk + position groups")
    players = pd.read_csv(fetch(f"{BASE}/players/players.csv.gz",
                                os.path.join(RAW, "players.csv.gz")), low_memory=False)
    xwalk = players[["gsis_id", "pfr_id", "ngs_position", "headshot",
                     "draft_round", "draft_pick", "rookie_season"]].drop_duplicates("gsis_id")
    stats = assign_group(stats, players)
    stats = stats[stats.pos_group.isin(GROUPS)]

    print("4/5 snap counts")
    snaps = load_snaps(seasons)

    print("5/5 joining")
    df = (stats.merge(xwalk.drop(columns="ngs_position"), on="gsis_id", how="left")
               .merge(snaps, on=["pfr_id", "season"], how="left")
               .merge(cap, on=["gsis_id", "season"], how="left"))
    df["has_contract"] = df["cap_number"].notna()
    df["side"] = np.where(df.pos_group.isin(OFFENSE), "Offense", "Defense")

    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, "player_seasons.parquet")
    df.to_parquet(dest, index=False)
    print(f"\nwrote {dest}  ({len(df):,} player-seasons)")
    print(df.groupby("pos_group")["has_contract"].agg(rows="size", matched="sum").to_string())


if __name__ == "__main__":
    main()
