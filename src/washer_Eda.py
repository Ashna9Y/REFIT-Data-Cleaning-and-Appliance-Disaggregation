"""
Task 2 - Washing-machine exploratory analysis on the official CLEANED REFIT data.

Run from the repo root:
    python src/washer_eda.py                      # auto-select houses that pass the quality rule
    python src/washer_eda.py --houses 1 2 5 6 7   # or choose houses yourself

Inputs : data/clean/CLEAN_House<N>.csv   (unzipped CLEAN_REFIT_081116.7z)
Outputs: data/processed/clean_1min/house<N>_1min.csv.gz   1-minute cache (re-used by Task 3)
         reports/t2_availability.csv, t2_cycles.csv, t2_cycle_summary.csv, t2_threshold_sensitivity.csv
         reports/figures/t2_*.png
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
CACHE_DIR = ROOT / "data" / "processed" / "clean_1min"
REP_DIR = ROOT / "reports"
FIG_DIR = REP_DIR / "figures"

# Washing-machine channel per house, from the cleaned readme "APPLIANCE LIST".
# Left out on purpose: House 4 (two washing machines, ambiguous), House 8 (channel blank),
# House 12 (no washer); Washer-Dryers (different load shape) are not used.
WASHER_CH = {1: 5, 2: 2, 3: 6, 5: 3, 6: 2, 7: 5, 9: 3, 10: 5, 11: 3, 13: 3,
             15: 3, 16: 5, 17: 4, 18: 5, 19: 2, 20: 4, 21: 3}

# Cleaned-file timestamps are UTC (checked: House 1 starts 13:06 in the cleaned file vs 14:06 in the raw file,
# same 523 W reading, because the raw file is UK clock time and it was BST). Hour-of-day / weekday analysis
# must therefore use UK local time.
LOCAL_TZ = "Europe/London"

# --- cycle rule (plain language in RESULTS.md) -----------------------------
ON_W = 20            # a minute is 'active' if the washer draws more than 20 W
MAX_PAUSE_MIN = 10   # active stretches separated by < 10 min of low power = same cycle
MIN_DUR_MIN = 10     # a cycle must last >= 10 min ...
MIN_KWH = 0.05       # ... and use >= 0.05 kWh, otherwise it is a blip, not a wash
FULL_DAY_FRAC = 0.80 # a day counts as 'observed' if >= 80% of its minutes have data
# 'Issues' = 1 when the sum of the sub-meters exceeds the aggregate (sensors are not synchronised). It fires mostly WHILE
# appliances run, so dropping every cycle that touches a flagged minute removes ~90% of real cycles (seen on the real data).
# For washer-only EDA the washer channel itself is fine, so we NOTE the flag (issue_frac) instead of discarding.
EXCLUDE_ISSUE_CYCLES = False
ISSUE_FRAC_MAX = 0.5          # only used if EXCLUDE_ISSUE_CYCLES=True: drop cycles with > 50% flagged minutes
# The cleaned data forward-fills sensor dropouts, so a dead monitor shows as a perfectly flat line. A real washer never
# holds exactly the same non-trivial power for this long, so such runs are masked (set to NaN) rather than trusted.
STUCK_MIN = 30

# --- automatic house selection rule ----------------------------------------
MIN_COVERAGE_PCT, MAX_ISSUES_PCT, MIN_CYCLES, MAX_HOUSES = 70, 15, 100, 8


# ----------------------------------------------------------------------------
# Time helpers
# ----------------------------------------------------------------------------
def local_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC (tz-naive) index -> UK wall-clock time (tz-naive). Used for hour-of-day / weekday."""
    return idx.tz_localize("UTC").tz_convert(LOCAL_TZ).tz_localize(None)


def parse_time(s: pd.Series) -> pd.Series:
    """ISO ('2013-10-09 13:06:17') is the expected format; day-first ('09-10-2013 13:06:17') is handled too,
    because a day-first file parsed month-first would silently swap day and month."""
    first = str(s.dropna().iloc[0]).strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", first):
        return pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


# ----------------------------------------------------------------------------
# Loading -> 1 minute
# ----------------------------------------------------------------------------
def find_file(n: int):
    for p in sorted(CLEAN_DIR.rglob("*.csv")):
        m = re.search(r"house_?0*(\d+)", p.name, re.I)
        if m and int(m.group(1)) == n:
            return p
    return None


def load_1min(n: int):
    """Washer + aggregate + Issues flag for house n at 1-minute resolution (cached)."""
    cache = CACHE_DIR / f"house{n}_1min.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    path = find_file(n)
    if path is None:
        return None
    col = f"Appliance{WASHER_CH[n]}"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    need = ["Time", "Aggregate", col, "Issues"]
    missing = [c for c in need if c not in header]
    if missing:
        raise ValueError(f"{path.name}: expected columns {need}, missing {missing}. Header is: {header}")
    df = pd.read_csv(path, usecols=need, dtype={"Aggregate": "float32", col: "float32", "Issues": "float32"})
    df["Time"] = parse_time(df["Time"])
    bad = int(df["Time"].isna().sum())
    df = df.dropna(subset=["Time"]).set_index("Time").sort_index()
    print(f"  House {n}: {path.name}, {len(df):,} rows, {df.index[0]} -> {df.index[-1]} (UTC), unparseable times dropped: {bad}")
    g = df.resample("1min")
    out = pd.DataFrame({
        "Aggregate": g["Aggregate"].mean(),
        "Washer": g[col].mean(),
        "Issues": g["Issues"].max(),
        "n_obs": g["Aggregate"].count(),     # 0 = no rows at all in that minute (a real gap)
    })
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index_label="Time")
    return out


# ----------------------------------------------------------------------------
# Cycle detection
# ----------------------------------------------------------------------------
def find_runs(mask) -> list:
    """Inclusive (start, end) index pairs of consecutive True values."""
    m = np.asarray(mask, dtype=np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def detect_cycles(df: pd.DataFrame, on_w=ON_W, max_pause=MAX_PAUSE_MIN,
                  min_dur=MIN_DUR_MIN, min_kwh=MIN_KWH) -> pd.DataFrame:
    """Rule: active minute = power > on_w. Active stretches separated by fewer than max_pause
    minutes of (observed) low power are merged. Keep cycles with duration >= min_dur and
    energy >= min_kwh. Missing minutes never count as active and never bridge a pause."""
    v = df["Washer"].to_numpy(float)
    nan = np.isnan(v)
    issues = (df["Issues"].fillna(0).to_numpy() > 0)
    merged = []
    for s, e in find_runs((~nan) & (v > on_w)):
        if merged:
            ps, pe = merged[-1]
            if (s - pe - 1) < max_pause and not nan[pe + 1:s].any():
                merged[-1] = (ps, e)
                continue
        merged.append((s, e))
    rows = []
    for s, e in merged:
        seg = v[s:e + 1]
        dur, kwh = e - s + 1, seg.sum() / 60 / 1000   # 1-min mean watts * 1 min
        if dur < min_dur or kwh < min_kwh:
            continue
        rows.append({
            "start": df.index[s], "end": df.index[e], "duration_min": dur, "energy_kwh": kwh,
            "peak_w": seg.max(), "mean_w": seg.mean(),
            "truncated": bool(s == 0 or e == len(v) - 1 or nan[s - 1] or nan[e + 1]),
            "issue_frac": float(issues[s:e + 1].mean()), "i0": s, "i1": e,
        })
    columns = ["start", "end", "duration_min", "energy_kwh", "peak_w", "mean_w",
               "truncated", "issue_frac", "i0", "i1"]
    out = pd.DataFrame(rows, columns=columns)
    if len(out):
        out["start_local"] = local_naive(pd.DatetimeIndex(out["start"]))
        out["end_local"] = local_naive(pd.DatetimeIndex(out["end"]))
    else:
        out["start_local"] = pd.Series(dtype="datetime64[ns]")
        out["end_local"] = pd.Series(dtype="datetime64[ns]")
    return out


def mask_stuck(df: pd.DataFrame, min_run: int = STUCK_MIN):
    """Mask long, exactly flat non-trivial runs in Aggregate and Washer.

    Aggregate flatlines can falsely inflate coverage. Washer flatlines can create
    false cycles. A run is masked only when the same value repeats for >= min_run
    minutes and is above the low-power threshold.
    Returns (new frame, total masked cells across both channels).
    """
    out = df.copy()
    masked_counts = {"aggregate": 0, "washer": 0}
    run_log = []
    for column, threshold in (("Aggregate", ON_W), ("Washer", ON_W)):
        w = out[column]
        const = ((w.diff() == 0) & (w > threshold)).to_numpy()
        col = out.columns.get_loc(column)
        key = "aggregate" if column == "Aggregate" else "washer"
        for s_, e_ in find_runs(const):
            # e_-s_+1 equal-to-previous flags represent e_-s_+2 identical samples.
            if e_ - s_ + 2 >= min_run:
                start, end = s_ - 1, e_
                run_log.append({"channel": key, "start": out.index[start], "minutes": int(end - start + 1),
                                "value_w": float(w.iloc[start])})
                out.iloc[start:end + 1, col] = np.nan
                masked_counts[key] += int(end - start + 1)
    masked_counts["runs"] = run_log
    return out, masked_counts


# ----------------------------------------------------------------------------
# Availability / data-quality table
# ----------------------------------------------------------------------------
def availability_row(n: int, df: pd.DataFrame, cyc: pd.DataFrame, stuck_masked: dict | None = None) -> dict:
    obs = df["n_obs"].to_numpy() > 0
    first, last = df.index[obs][0], df.index[obs][-1]
    stuck_masked = stuck_masked or {"aggregate": 0, "washer": 0}
    span = df.loc[first:last]
    obs_span = span["n_obs"].to_numpy() > 0
    # Missing-row gaps and flatline-masked Aggregate readings are different problems.
    gaps = [e - s + 1 for s, e in find_runs(~obs_span)]
    aggregate_valid = span["Aggregate"].notna().to_numpy()
    w = span["Washer"]
    wv = w[obs_span & w.notna().to_numpy()]
    usable = cyc[~cyc["truncated"] & ~((cyc["issue_frac"] > ISSUE_FRAC_MAX) & EXCLUDE_ISSUE_CYCLES)] if len(cyc) else cyc
    days = max((last - first).days, 1)
    obs_days = max(days * float(aggregate_valid.mean()), 1)
    return {
        "house": n, "washer_channel": f"Appliance{WASHER_CH[n]}",
        "first": first.date(), "last": last.date(), "span_days": days,
        "coverage_pct": round(100 * aggregate_valid.mean(), 1),
        "gaps_over_5min": int(sum(g > 5 for g in gaps)),
        "longest_gap_hours": round(max(gaps, default=0) / 60, 1),
        "issues_pct": round(100 * float((span["Issues"].fillna(0) > 0)[obs_span].mean()), 1),
        "off_time_pct": round(100 * float((wv <= ON_W).mean()), 1),
        "zero_time_pct": round(100 * float((wv == 0).mean()), 1),
        "cycles_detected": int(len(cyc)), "cycles_usable": int(len(usable)),
        "cycles_per_observed_week": round(len(usable) / (obs_days / 7), 1),
        "cycles_touching_issue_pct": round(100 * float((cyc["issue_frac"] > 0).mean()), 1) if len(cyc) else 0.0,
        "agg_masked_min": int(stuck_masked.get("aggregate", 0)),
        "washer_masked_min": int(stuck_masked.get("washer", 0)),
    }


def start_time_clustering(cycles: pd.DataFrame) -> pd.DataFrame:
    """Timer check: share of a house's cycles starting in its single busiest 5-minute slot of the day,
    and the spread of cycle duration. A high share + tiny spread suggests a timer / fixed programme (hypothesis only)."""
    rows = []
    for h, c in cycles.groupby("house"):
        slot = (c["start_local"].dt.hour * 60 + c["start_local"].dt.minute) // 5
        vc = slot.value_counts()
        rows.append({"house": h, "n_cycles": len(c), "top_slot_start": f"{int(vc.index[0]) * 5 // 60:02d}:{int(vc.index[0]) * 5 % 60:02d}",
                     "top_slot_share_pct": round(100 * vc.iloc[0] / len(c), 1),
                     "duration_cv": round(float(c["duration_min"].std() / c["duration_min"].mean()), 2)})
    return pd.DataFrame(rows)


def observed_days(df: pd.DataFrame) -> pd.Series:
    """Per calendar day: True if >= FULL_DAY_FRAC of its minutes have data."""
    day = local_naive(df.index).normalize()
    return df["Washer"].notna().groupby(day).mean() >= FULL_DAY_FRAC     # washer itself must be readable


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_threshold(data: dict, path: Path):
    pooled = pd.concat([d["Washer"] for d in data.values()]).dropna()
    pooled = pooled[pooled > 0.5]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(pooled.clip(upper=2800), bins=np.arange(0, 2800, 10), color="tab:blue")
    ax.set_yscale("log"); ax.axvline(ON_W, color="red", ls="--", label=f"active threshold = {ON_W} W")
    ax.set_xlabel("washing-machine power, 1-min mean (W)"); ax.set_ylabel("minutes (log scale)")
    ax.set_title("Where do washer readings sit? (all selected houses, readings > 0.5 W)")
    ax.legend(); plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def threshold_sensitivity(data: dict) -> pd.DataFrame:
    rows = []
    for on in (10, 20, 30, 50):
        for pause in (5, 10, 15):
            tot = [len(detect_cycles(d, on_w=on, max_pause=pause)) for d in data.values()]
            rows.append({"on_w": on, "max_pause_min": pause, "total_cycles": int(sum(tot)),
                         **{f"house{h}": t for h, t in zip(data, tot)}})
    return pd.DataFrame(rows)


def plot_usage_patterns(cycles: pd.DataFrame, data: dict, path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    hours, dows = [], []
    for h, d in data.items():
        od = observed_days(d)
        c = cycles[cycles["house"] == h]
        c = c[c["start_local"].dt.normalize().map(od).fillna(False).astype(bool)]    # only count cycles on observed days
        if len(c) == 0:
            continue
        hr = c["start_local"].dt.hour.value_counts().reindex(range(24), fill_value=0)
        hours.append((hr / hr.sum() * 100).rename(h))
        n_days = pd.Series(od[od].index.day_of_week).value_counts().reindex(range(7), fill_value=0)
        dw = c["start_local"].dt.day_of_week.value_counts().reindex(range(7), fill_value=0)
        dows.append((dw / n_days.replace(0, np.nan)).rename(h))
    H, D = pd.concat(hours, axis=1), pd.concat(dows, axis=1)
    for h in H:
        ax[0].plot(H.index, H[h], lw=1, alpha=0.6, label=f"House {h}")
        ax[1].plot(D.index, D[h], lw=1, alpha=0.6)
    ax[0].plot(H.index, H.mean(axis=1), "k", lw=3, label="mean of houses")
    ax[1].plot(D.index, D.mean(axis=1), "k", lw=3)
    ax[0].set_xlabel("hour of day when cycle STARTS (UK local time)"); ax[0].set_ylabel("% of house's cycles")
    ax[0].set_xticks(range(0, 24, 3)); ax[0].legend(fontsize=7, ncol=2)
    ax[1].set_xticks(range(7)); ax[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax[1].set_ylabel("cycles per observed day")
    ax[0].set_title("Detected cycle starts by hour"); ax[1].set_title("Detected cycles per observed weekday")
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return H, D


def pick_representative_day(d: pd.DataFrame, cyc: pd.DataFrame):
    """Complete day (>= 95% minutes observed) with >= 1 usable cycle whose washer energy is closest to the median."""
    grp = d["Washer"].notna().groupby(local_naive(d.index).normalize())
    cov, cnt = grp.mean(), grp.size()
    full = set(cov[(cov >= 0.95) & (cnt == 1440)].index)     # 1440 = skips the 23h / 25h clock-change days
    c = cyc[~cyc["truncated"]].copy()
    c["day"] = c["start_local"].dt.normalize()
    c = c[c["day"].isin(full)]
    if (c["issue_frac"] <= 0.1).any():          # prefer a day whose aggregate trace is not flagged
        c = c[c["issue_frac"] <= 0.1]
    if c.empty:
        return None
    e = c.groupby("day")["energy_kwh"].sum()
    return (e - e.median()).abs().idxmin()


def plot_day(d: pd.DataFrame, cyc: pd.DataFrame, day, house: int, path: Path):
    t0 = day.tz_localize(LOCAL_TZ).tz_convert("UTC").tz_localize(None)       # local midnight as UTC
    w = d.loc[t0: t0 + pd.Timedelta(hours=23, minutes=59)]
    x = local_naive(w.index)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]})
    ax[0].plot(x, w["Aggregate"].values, color="tab:blue", lw=1); ax[0].set_ylabel("Aggregate (W)")
    ax[1].plot(x, w["Washer"].values, color="tab:green", lw=1); ax[1].set_ylabel("Washing machine (W)")
    for _, r in cyc[(cyc["start_local"] >= day) & (cyc["start_local"] < day + pd.Timedelta(days=1))].iterrows():
        for a in ax:
            a.axvspan(r["start_local"], r["end_local"], color="orange", alpha=0.25, lw=0)
    ax[0].set_title(f"House {house}, {day:%a %d %b %Y}: whole-house load (top) and washing machine (bottom); "
                    "orange = detected cycle")
    ax[1].set_xlabel("UK local time")
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_cycle_gallery(data: dict, cycles: pd.DataFrame, path: Path, max_panels=8):
    houses = [h for h in data if len(cycles[(cycles["house"] == h) & ~cycles["truncated"]])][:max_panels]
    ncol = 4; nrow = max(1, int(np.ceil(len(houses) / ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.2 * nrow), squeeze=False)
    for ax, h in zip(axes.ravel(), houses):
        c = cycles[(cycles["house"] == h) & ~cycles["truncated"]]
        r = c.iloc[(c["energy_kwh"] - c["energy_kwh"].median()).abs().argsort().iloc[0]]   # median-energy cycle
        d = data[h]; w = d["Washer"].iloc[max(r["i0"] - 20, 0): r["i1"] + 21]
        ax.plot(((w.index - r["start"]).total_seconds() / 60), w.values, color="tab:green", lw=1)
        ax.axhline(ON_W, color="red", ls="--", lw=0.8)
        ax.axvspan(0, r["duration_min"] - 1, color="orange", alpha=0.2, lw=0)
        ax.set_title(f"House {h}: {r['duration_min']:.0f} min, {r['energy_kwh']:.2f} kWh, peak {r['peak_w']:.0f} W", fontsize=9)
        ax.set_xlabel("minutes from cycle start"); ax.set_ylabel("W")
    for ax in axes.ravel()[len(houses):]:
        ax.axis("off")
    fig.suptitle("Detected washing-machine cycles (one median-energy cycle per house; red dashed = 20 W threshold)", y=1.0)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_cross_house(cycles: pd.DataFrame, path: Path):
    order = cycles.groupby("house")["energy_kwh"].median().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (col, lab) in zip(axes, [("energy_kwh", "energy per cycle (kWh)"),
                                     ("duration_min", "duration (min)"), ("peak_w", "peak power (W)")]):
        ax.boxplot([cycles.loc[cycles["house"] == h, col] for h in order], showfliers=False)
        ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([f"H{h}" for h in order])
        ax.set_ylabel(lab); ax.set_title(lab)
    fig.suptitle("Cross-house comparison of cycle characteristics (boxes = IQR, whiskers 1.5 IQR, outliers hidden)", y=1.0)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", type=int, nargs="*", help="houses to use (default: auto-select)")
    args = ap.parse_args()
    REP_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. load every candidate house we have a file for, detect cycles, build the availability table
    data, cyc_all, rows, masked_runs = {}, {}, [], []
    for n in sorted(WASHER_CH):
        d = load_1min(n)
        if d is None:
            continue
        print(f"House {n}: loaded {len(d):,} one-minute rows")
        d, stuck_counts = mask_stuck(d)
        for r in stuck_counts.pop("runs"):
            masked_runs.append({"house": n, **r})
        data[n] = d; cyc_all[n] = detect_cycles(d); rows.append(availability_row(n, d, cyc_all[n], stuck_counts))
    if not rows:
        raise SystemExit(f"No House files found in {CLEAN_DIR}. Expected names like CLEAN_House1.csv")
    avail = pd.DataFrame(rows)
    avail.to_csv(REP_DIR / "t2_availability.csv", index=False)
    mr = pd.DataFrame(masked_runs, columns=["house", "channel", "start", "minutes", "value_w"])
    mr.to_csv(REP_DIR / "t2_masked_runs.csv", index=False)
    if len(mr):
        a_ = mr[mr["channel"] == "aggregate"]
        print(f"\nFrozen-run audit: {len(a_)} aggregate runs masked; {100 * (a_['value_w'] < 300).mean():.0f}% of them "
              f"sit below 300 W (normal baseline - if this is high the rule may be eating real data). "
              f"Longest run: {mr['minutes'].max()} min. Full list: reports/t2_masked_runs.csv")
    print("\n=== Availability of ALL candidate houses ===")
    print(avail.to_string(index=False, col_space=10))

    # 2. choose houses
    if args.houses:
        chosen = [h for h in args.houses if h in data]
    else:
        ok = avail[(avail.coverage_pct >= MIN_COVERAGE_PCT) & (avail.issues_pct <= MAX_ISSUES_PCT)
                   & (avail.cycles_usable >= MIN_CYCLES)]
        chosen = ok.sort_values("cycles_usable", ascending=False).head(MAX_HOUSES)["house"].tolist()
    if len(chosen) < 4:
        print(f"WARNING: only {len(chosen)} houses selected; the assignment needs >= 4. Check the table above.")
    chosen = sorted(chosen)
    if not chosen:
        raise SystemExit("No house passed the selection rule. Look at the table above and either relax "
                         "MIN_COVERAGE_PCT / MAX_ISSUES_PCT / MIN_CYCLES at the top of the script or pass --houses.")
    print(f"\nHouses used: {chosen}")

    # 3. cycles of chosen houses (apply exclusions)
    parts = []
    for h in chosen:
        c = cyc_all[h].copy(); c.insert(0, "house", h)
        parts.append(c)
    cycles = pd.concat(parts, ignore_index=True)
    n0 = len(cycles)
    keep = ~cycles["truncated"]
    if EXCLUDE_ISSUE_CYCLES:
        keep &= ~(cycles["issue_frac"] > ISSUE_FRAC_MAX)
    print(f"Chosen houses: {n0} cycles; {int(cycles['truncated'].sum())} touch a data gap/edge; "
          f"{100 * (cycles['issue_frac'] > 0).mean():.0f}% touch at least one 'Issues' minute (kept, but noted in t2_cycles.csv)")
    cycles = cycles[keep].reset_index(drop=True)
    print(f"Cycles: {n0} detected, {len(cycles)} kept (dropped {n0 - len(cycles)} truncated by a gap{' or heavily flagged' if EXCLUDE_ISSUE_CYCLES else ''})")
    cycles.drop(columns=["i0", "i1"]).to_csv(REP_DIR / "t2_cycles.csv", index=False)

    summ = cycles.groupby("house").agg(
        n_cycles=("energy_kwh", "size"), duration_median=("duration_min", "median"),
        energy_median_kwh=("energy_kwh", "median"), energy_iqr_kwh=("energy_kwh", lambda x: x.quantile(.75) - x.quantile(.25)),
        peak_median_w=("peak_w", "median")).round(2)
    summ.to_csv(REP_DIR / "t2_cycle_summary.csv")
    print("\n=== Cycle characteristics per house ===")
    print(summ.to_string())

    # 4. figures
    sel = {h: data[h] for h in chosen}
    plot_threshold(sel, FIG_DIR / "t2_threshold_histogram.png")
    threshold_sensitivity(sel).to_csv(REP_DIR / "t2_threshold_sensitivity.csv", index=False)
    H, D = plot_usage_patterns(cycles, sel, FIG_DIR / "t2_usage_hour_dow.png")
    days_ = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"\nPeak hour: mean of houses {H.mean(axis=1).idxmax()}:00, median of houses {H.median(axis=1).idxmax()}:00; "
          f"per-house peaks: {({int(k): int(v) for k, v in H.idxmax().items()})}")
    print(f"Busiest weekday: mean of houses {days_[int(D.mean(axis=1).idxmax())]}, median of houses {days_[int(D.median(axis=1).idxmax())]}")
    stc = start_time_clustering(cycles)
    stc.to_csv(REP_DIR / "t2_start_time_clustering.csv", index=False)
    print("\n=== Start-time clustering (timer hint) ===")
    print(stc.to_string(index=False))
    plot_cycle_gallery(sel, cycles, FIG_DIR / "t2_cycle_gallery.png")
    plot_cross_house(cycles, FIG_DIR / "t2_cross_house.png")

    best = cycles["house"].value_counts().idxmax()
    day = pick_representative_day(sel[best], cycles[cycles["house"] == best])
    if day is not None:
        plot_day(sel[best], cycles[cycles["house"] == best], day, best, FIG_DIR / "t2_representative_day.png")
        print(f"Representative day: House {best}, {day.date()}")
    print("\nDone. See reports/ and reports/figures/.")


if __name__ == "__main__":
    main()