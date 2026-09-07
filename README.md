# NFL Contract Value

Ranks NFL players by how much production they deliver relative to what they cost
against the salary cap — across nine position groups, on offense and defense.

**Live app:** _(add your Streamlit URL here after deploying — see DEPLOY.md)_

## What it does

Every player is ranked twice inside his own position group and season: once on a
production composite, once on his cap charge. The **value score** is the gap
between those two percentile ranks.

- `+40` → produced 40 percentile points better than he was paid
- `-40` → the reverse

Ranking within position-season is what makes a quarterback and a cornerback
comparable, and it avoids the blow-ups you get from dollars-per-yard metrics
when production is near zero or negative.

Every player is scored **two ways**:

| Basis | Question it answers |
| --- | --- |
| Season totals | What did the team actually get for the money? |
| Per game | How good was he when he was on the field? |

The difference between the two is the **injury gap**. A player with a large
positive gap was good but unavailable; a gap near zero means the production
genuinely wasn't there. The app has a tab that splits those two cases apart.

## Position groups

**Offense:** QB, RB, WR, TE
**Defense:** EDGE, DL, LB, CB, S

Offensive line is deliberately excluded — linemen record essentially no counting
stats, so any number produced for them would be noise.

Defensive groups come from NGS position labels where available, since nflverse's
base labels are unreliable (a 3-4 outside linebacker who rushes the passer every
down is listed as "LB"). Where NGS data is missing, unlabeled linebackers with
high pass-rush production are reclassified as EDGE.

## Confidence ratings

Not every position is measured equally well by box score stats, and the app says
so rather than hiding it:

| Group | Confidence | Why |
| --- | --- | --- |
| QB, RB, WR, TE, EDGE | High | The box score captures the job |
| DL, LB | Medium | Run-stuffers absorb blocks without recording stats; tackle totals depend on scheme |
| CB, S | **Low** | A corner nobody throws at records almost nothing |

**The cornerback caveat matters.** Passes defended and interceptions require the
ball to come your way, which is largely outside the player's control. A low score
for a corner often means "avoided", not "bad". The app shows a warning whenever
those groups are in view.

## Teams

A team's **contract efficiency** is every player's value score averaged together,
weighted by his share of the team's cap — a bad $40M contract has to hurt far more
than a bad $2M one — then centred on the league average for that season, so 0 is an
average front office that year.

The centring matters. `value_score = production percentile − pay percentile`, so a
player in the 95th percentile of pay can score at most +5 while one in the 5th can
reach +95. The metric is asymmetrically bounded, and cap-weighting deliberately
upweights the expensive players who structurally can't score high — the raw team
average comes out negative for 95% of team-seasons. That bias is identical for
every team, so the ranking is sound, but the raw zero point is meaningless.

**It describes the season you just watched, and does not predict the next one.**
Measured over 2019–2025 (223 team-seasons):

| Relationship | r | Variance explained |
| --- | --- | --- |
| Efficiency vs wins, same season | +0.55 | 30% |
| Efficiency vs point differential, same season | +0.51 | 26% |
| Efficiency vs **next** season's wins | +0.19 | 4% |
| Prior wins vs next season's wins (baseline) | +0.38 | 15% |

Adding efficiency to a model that already knows last year's record raises R² from
0.147 to 0.147 — it contributes nothing. Partial correlation controlling for record
is −0.02 (p = 0.84).

**Why it doesn't carry over: the market corrects.** A bargain is temporary by
construction. Of player-seasons scoring +30 or better, only 32% were still a bargain
the next year; 75% saw their cap charge rise and the median more than doubled. Of
those scoring −30 or worse, 59% saw their cap charge fall. Team efficiency itself
barely persists year to year (r = +0.15).

Only about **50% of each team's salary cap** is graded, since offensive linemen,
specialists and players below the usage minimums aren't scored.

## Data

All from [nflverse](https://github.com/nflverse/nflverse-data/releases):

| Release | Asset | Used for |
| --- | --- | --- |
| `contracts` | `historical_contracts.parquet` | OverTheCap deals + per-season cap numbers |
| `stats_player` | `stats_player_week_<year>.csv.gz` | weekly box score + EPA, offense and defense |
| `snap_counts` | `snap_counts_<year>.csv.gz` | playing time |
| `players` | `players.csv.gz` | ID crosswalk and NGS positions |
| `schedules` | `games.csv` | team records and point differentials |

**Two things that will bite you if you don't know them:**

1. In the contracts release, **the `.csv.gz` asset is stale** — not rebuilt since
   May 2022. Only `historical_contracts.parquet` is current. Same release, same
   tag, different extension.
2. The parquet carries a nested `season_history` array with the real per-season
   `cap_number` and `cap_percent`. That's far better than approximating from APY,
   and it's what makes the contract-year trend view possible. It also contains a
   `'Total'` summary row that must be skipped when flattening.

Contracts join to stats on `gsis_id`, which is present in both — no name
matching. That matters: there are two Justin Jeffersons (a Vikings WR and a
Browns LB) and name matching would merge them.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py          # data for 2019-2025 is already built in
```

To rebuild or extend the data:

```bash
python src/build_data.py --seasons 2019 2025   # downloads + joins, ~1 min
python src/value_model.py                      # scores everything
python src/team_model.py                       # rolls up to team level
```

Downloads are cached in `data/raw/`; delete it to force a refresh.

See **[DEPLOY.md](DEPLOY.md)** for step-by-step hosting instructions.

## Layout

```
src/build_data.py    fetch nflverse data, join contracts to stats -> player_seasons.parquet
src/value_model.py   production composites, percentiles, value scores -> value_scores.parquet
src/team_model.py    cap-weighted team efficiency + records -> team_seasons.parquet
app.py               Streamlit UI
```

## The model

Production composites blend volume with efficiency per position (see `RECIPE` in
`src/value_model.py`). Z-scores are clipped at ±3 so one outlier can't dominate.

Minimum usage to be ranked: QB 150 attempts, RB 50 touches, WR 25 targets,
TE 20 targets, defenders 200 snaps. Per-game mode additionally requires 4 games.

### Two traps already handled

**Cumulative rushing EPA is negative for nearly every running back** — a handoff
is a below-average play in the abstract, so season-total rushing EPA is
anti-correlated with workload. Weighted at 0.25 in the first version, it put a
1,300-yard rookie in the 1st percentile. Rushing efficiency now enters as **EPA
per carry**, with yards, first downs and touchdowns carrying the volume. Passing
and receiving EPA don't have this problem and stay as totals.

**Expected salary has to be a smoothed curve.** Interpolating straight through
every player's own point makes his own salary the expected salary, so the surplus
is always exactly zero. Production is binned into deciles and the median cap
charge per decile forms the market curve.

## Limitations

- No offensive line, and no kickers or punters.
- Cornerback and safety scores are weak by construction — see above.
- Cap hit is not cash paid, and dead money on a release isn't modeled.
- Production is credited to the player, not adjusted for scheme, supporting cast,
  or quarterback quality.
- The season salary cap is estimated by dividing cap dollars by cap percentage
  and taking a median, so it's within a few percent rather than exact.
