"""
Task 1 - Basic inspection and cleaning of RAW REFIT House 1.

Run from the repo root:
    python src/clean_house1.py

Inputs  (put the two raw CSVs in data/raw/):
    RAW_House1_Part1.csv   (2013-10-09 -> 2014-10-01, unavailable sensors = 0)
    RAW_House1_Part2.csv   (2014-06-27 -> 2015-07-10, unavailable sensors = NaN)

Outputs:
    data/processed/house1_1min_clean.csv.gz   cleaned 1-minute table (UTC index + flags)
    reports/house1_quality_summary.json       numbers quoted in DATA.md
    reports/house1_outages.csv                every gap longer than MAX_FILL_MIN
    reports/figures/*.png                     before/after plots
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration: every threshold used in cleaning lives here, with its reason.
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
REP_DIR = ROOT / "reports"
FIG_DIR = REP_DIR / "figures"

LOCAL_TZ = "Europe/London"   # raw timestamps are UK wall-clock time (see DATA.md)
MAX_FILL_MIN = 5             # fill gaps up to 5 consecutive minutes, flag longer ones
APP_MAX_W = 4000             # a UK plug-in monitor cannot see >~3 kW; matches REFIT's own spike limit
AGG_EXTREME_W = 20000        # flag (not remove) aggregate readings above this
STUCK_MIN = 60               # MINUTES: 1-min aggregate identical for >= 60 min = dead logger (applied after resampling)
STUCK_MIN_W = 20             # only treat a flat aggregate above this power as suspicious
MIN_VALID_DAY_FRAC = 0.80    # overview plot: draw a daily mean only if >= 80% of that day's minutes are valid
RESAMPLE = "1min"

AGG = "Aggregate"
APPS = [f"Appliance{i}" for i in range(1, 10)]
VALUE_COLS = [AGG] + APPS
# Names from the REFIT readme, House 1
APP_NAMES = {
    "Appliance1": "Fridge", "Appliance2": "Freezer(1)", "Appliance3": "Freezer(2)",
    "Appliance4": "Washer Dryer", "Appliance5": "Washing Machine", "Appliance6": "Dishwasher",
    "Appliance7": "Computer", "Appliance8": "Television Site", "Appliance9": "Electric Heater",
}


# ----------------------------------------------------------------------------
# 1. Loading and inspection
# ----------------------------------------------------------------------------
def load_part(path: Path) -> pd.DataFrame:
    """Read one raw CSV. Time stays as naive local wall-clock; row order is preserved."""
    dtypes = {c: "float32" for c in VALUE_COLS}   # halves memory; watts are integers anyway
    df = pd.read_csv(path, dtype=dtypes, usecols=["Time"] + VALUE_COLS)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    return df


def inspect_raw(df: pd.DataFrame, name: str) -> dict:
    """Quality checks on the untouched raw file. Returns a dict of findings."""
    t = df["Time"]
    step = t.diff().dt.total_seconds()
    dup_ts = t.duplicated(keep=False)
    dd = df.loc[dup_ts, ["Time"] + VALUE_COLS]
    grp = dd.groupby("Time")[VALUE_COLS].nunique(dropna=False)
    del dd
    # Count the two UK daylight-saving cases separately. Ambiguous = repeated autumn hour;
    # nonexistent = skipped spring hour.
    ambiguous_local = t.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward")
    nonexistent_local = t.dt.tz_localize(LOCAL_TZ, ambiguous=True, nonexistent="NaT")
    local = t.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT")
    rep = {
        "file": name,
        "rows": int(len(df)),
        "unparseable_timestamps": int(t.isna().sum()),
        "first_timestamp": str(t.min()),
        "last_timestamp": str(t.max()),
        "out_of_order_steps": int((step < 0).sum()),
        "rows_with_duplicated_timestamp": int(dup_ts.sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_timestamp_groups": int(len(grp)),
        "duplicate_groups_with_conflicting_values": int((grp.max(axis=1) > 1).sum()),
         "rows_in_ambiguous_dst_hour": int((ambiguous_local.isna() & t.notna()).sum()),
        "rows_in_nonexistent_dst_hour": int((nonexistent_local.isna() & t.notna()).sum()),
        "rows_unlocalizable_dst_total": int((local.isna() & t.notna()).sum()),
        "median_step_s": float(step[step >= 0].median()),
        "steps_over_60s": int((step > 60).sum()),
        "steps_over_5min": int((step > 300).sum()),
        "steps_over_1h": int((step > 3600).sum()),
        "nan_per_column": {c: int(df[c].isna().sum()) for c in VALUE_COLS},
        "aggregate_zero_or_negative": int((df[AGG] <= 0).sum()),
        "aggregate_above_20kW": int((df[AGG] > AGG_EXTREME_W).sum()),
        "aggregate_max_W": float(df[AGG].max()),
        "appliance_rows_above_4kW": {c: int((df[c] > APP_MAX_W).sum()) for c in APPS},
        "appliance_max_W": {c: float(df[c].max()) for c in APPS},
        "step_seconds_histogram_top8": {
            str(int(k)): int(v) for k, v in step[step >= 0].round().value_counts().head(8).items()
        },
    }
    return rep


# ----------------------------------------------------------------------------
# 2. Cleaning
# ----------------------------------------------------------------------------
def stitch_parts(p1: pd.DataFrame, p2: pd.DataFrame) -> pd.DataFrame:
    """Part1 until Part2 starts, then Part2.
    Reason: Part2 marks unavailable sensors as NaN (Part1 uses 0, which looks like 'off')."""
    cut = p2["Time"].min()
    return pd.concat([p1[p1["Time"] < cut], p2], ignore_index=True)


def to_utc_index(df: pd.DataFrame):
    """Raw Time is UK wall-clock (clock jumps back 1 h each October, skips 1 h each March).
    Convert to UTC so time is continuous. Rows inside the repeated October hour cannot be
    attributed to the right pass and are dropped (counted and reported)."""
    t = df["Time"]
    local = t.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT")
    ambiguous_local = t.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward")
    nonexistent_local = t.dt.tz_localize(LOCAL_TZ, ambiguous=True, nonexistent="NaT")
    valid_time = t.notna()
    keep = local.notna()
    out = df.loc[keep].copy()
    out.index = local[keep].dt.tz_convert("UTC")
    out = out.drop(columns=["Time"])
    dst_log = {
        "rows_dropped_unparseable_time": int((~valid_time).sum()),
        "rows_dropped_ambiguous_dst_hour": int((ambiguous_local.isna() & valid_time).sum()),
        "rows_dropped_nonexistent_dst_hour": int((nonexistent_local.isna() & valid_time).sum()),
    }
    return out, dst_log


def remove_impossible(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level (before resampling, so one bad reading cannot pollute a minute average)."""
    df = df.copy()
    df.loc[df[AGG] <= 0, AGG] = np.nan               # a house with a fridge is never at 0 W
    for c in APPS:
        df.loc[(df[c] < 0) | (df[c] > APP_MAX_W), c] = np.nan   # 16-bit overflow style garbage
    return df


def dedupe_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time; rows sharing a timestamp are averaged (NaN ignored).
    Exact duplicates collapse to themselves; conflicting ones (sensors polled at slightly
    different moments) become their mean, so no information is thrown away."""
    return df.sort_index().groupby(level=0).mean()


def fill_short_gaps(s: pd.Series, max_len: int, method: str) -> pd.Series:
    """Fill only NaN runs of <= max_len consecutive samples; longer runs are left untouched
    (pandas' own `limit` would wrongly fill the first max_len minutes of a long outage)."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    short = isna & (run_len <= max_len)
    if method == "linear":
        filled = s.interpolate(method="time", limit_area="inside")
    else:  # forward-fill: appliance monitors log only on change, so 'last value' is the data's own logic
        filled = s.ffill()
    return s.where(~short, filled)


def mask_flat_runs(s: pd.Series, min_run: int = STUCK_MIN,
                   min_value: float = STUCK_MIN_W) -> pd.Series:
    """Boolean mask of a frozen logger: a 1-MINUTE series that repeats the exact same value for
    >= min_run minutes while above min_value. Call it on the resampled 1-minute series
    (min_run is in minutes, not raw ~8 s samples)."""
    v = s.to_numpy(dtype=float)
    valid = ~np.isnan(v)
    same = np.zeros(len(v), dtype=bool)
    if len(v) > 1:
        same[1:] = valid[1:] & valid[:-1] & (np.abs(v[1:] - v[:-1]) < 1e-6) & (v[1:] > min_value)
    frozen = np.zeros(len(v), dtype=bool)
    for a, b in find_true_runs(same):
        if (b - a + 2) >= min_run:          # b-a+1 equal-to-previous flags = b-a+2 identical samples
            frozen[a - 1:b + 1] = True
    return pd.Series(frozen, index=s.index)


def find_true_runs(mask) -> list[tuple[int, int]]:
    """Inclusive (start, end) runs of True values in a one-dimensional mask."""
    m = np.asarray(mask, dtype=np.int8)
    edges = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1))


def clean_pipeline(raw: pd.DataFrame):
    """raw (stitched, wall-clock) -> cleaned 1-minute frame + log of what was done."""
    log = {}
    df, dst_log = to_utc_index(raw)
    log.update(dst_log)
    n_bad = ((df[APPS] > APP_MAX_W) | (df[APPS] < 0)).sum().sum()    # counts BOTH >4 kW and negatives
    df = remove_impossible(df)
    log["appliance_readings_set_to_nan_(>4kW_or_<0)"] = int(n_bad)
    df = dedupe_and_sort(df)
    log["rows_after_dedupe"] = int(len(df))

    # Aggregate extremes: record before they get averaged into a minute
    # (extremes were kept in df; just remember which minutes contain one)
    extreme_min = df[AGG].gt(AGG_EXTREME_W).resample(RESAMPLE).sum().gt(0)

    one = df.resample(RESAMPLE).mean()          # mean power per minute
    one["extreme_agg"] = extreme_min.reindex(one.index, fill_value=False).astype(bool)

    # Frozen logger: 1-min aggregate identical for >= STUCK_MIN minutes. Whole row -> NaN (all channels),
    # because a dead logger makes every channel untrustworthy; stays a long gap (never interpolated).
    frozen = mask_flat_runs(one[AGG])
    one.loc[frozen, VALUE_COLS] = np.nan
    one["frozen"] = frozen
    log["minutes_masked_frozen_logger"] = int(frozen.sum())
    log["frozen_blocks"] = int(len(find_true_runs(frozen.to_numpy())))

    raw_nan_min = one[VALUE_COLS].isna()        # NaN minutes BEFORE any filling

    for c in VALUE_COLS:
        one[c] = fill_short_gaps(one[c], MAX_FILL_MIN, "linear" if c == AGG else "ffill")

    one["filled_agg"] = raw_nan_min[AGG] & one[AGG].notna()
    one["outage"] = one[AGG].isna()             # aggregate still missing = long gap, left as NaN
    return one, log


def dst_windows(index: pd.DatetimeIndex) -> list:
    """UTC windows lost to the repeated October hour: BOTH passes are dropped, so 00:00-01:59 UTC
    (120 minutes) on the last Sunday of October."""
    wins = []
    for y in range(index.min().year, index.max().year + 1):
        d = pd.Timestamp(f"{y}-10-31", tz="UTC")
        while d.weekday() != 6:
            d -= pd.Timedelta(days=1)
        wins.append((d, d + pd.Timedelta(minutes=119)))
    return wins


def outage_table(flag: pd.Series, frozen: pd.Series, windows: list) -> pd.DataFrame:
    """One row per missing-aggregate run, with a reason: dst_clock_change / frozen_logger / data_or_sensor_outage."""
    run = (flag != flag.shift()).cumsum()
    rows = []
    for _, g in flag.groupby(run):
        if not g.iloc[0]:
            continue
        start, end = g.index[0], g.index[-1]
        fz = int(frozen.reindex(g.index).sum())
        in_dst = any(start >= a and end <= b for a, b in windows)
        if in_dst:
            reason = "dst_clock_change"
        elif fz > 0.5 * len(g):
            reason = "frozen_logger"
        else:
            reason = "data_or_sensor_outage"
        rows.append({"start_utc": start, "end_utc": end, "minutes": len(g),
                     "frozen_minutes": fz, "gap_type": reason})
    return pd.DataFrame(rows, columns=["start_utc", "end_utc", "minutes", "frozen_minutes", "gap_type"])


# ----------------------------------------------------------------------------
# 3. "Before" version (naive) and plotting
# ----------------------------------------------------------------------------
def naive_before(raw: pd.DataFrame) -> pd.DataFrame:
    """What you'd get with no cleaning: sort, take 1-min mean. Wall-clock time, no spike removal,
    no gap handling. Used only to show the effect of cleaning."""
    d = raw.set_index("Time").sort_index()
    return d.resample(RESAMPLE).mean()


def plot_overview(before: pd.DataFrame, after: pd.DataFrame, path: Path):
    """Whole-period daily view of aggregate; shaded = outages > MAX_FILL_MIN minutes."""
    b = before[AGG].resample("1D").mean()
    a_l = after[AGG].tz_convert(LOCAL_TZ).tz_localize(None)
    a = a_l.resample("1D").mean()
    valid_frac = a_l.notna().resample("1D").mean()
    a = a.where(valid_frac >= MIN_VALID_DAY_FRAC)      # a mean over a few valid hours is not a daily mean
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True, sharey=True)
    ax[0].plot(b.index, b.values, lw=0.9, color="tab:red")
    ax[0].set_title("BEFORE cleaning: daily mean aggregate (naive 1-min resample; missing periods are not explicitly flagged)")
    ax[1].plot(a.index, a.values, lw=0.9, color="tab:blue")
    out = after["outage"].tz_convert(LOCAL_TZ).tz_localize(None)
    o = out.resample("1D").mean()
    for d_, frac in o.items():
        if frac > 0.5:
            ax[1].axvspan(d_, d_ + pd.Timedelta(days=1), color="grey", alpha=0.35, lw=0)
    ax[1].set_title("AFTER cleaning: daily mean aggregate (days <80% valid not drawn; grey = day mostly in outage)")
    for x in ax:
        x.set_ylabel("Watts")
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_window(before: pd.DataFrame, after: pd.DataFrame, start, end, channels, title, path: Path):
    """Zoom on one window, channels in rows, before (left) vs after (right)."""
    after_l = after.tz_convert(LOCAL_TZ).tz_localize(None)
    fig, axes = plt.subplots(len(channels), 2, figsize=(14, 2.6 * len(channels)), sharex=True)
    for r, c in enumerate(channels):
        nm = "Aggregate" if c == AGG else f"{c}: {APP_NAMES[c]}"
        b = before.loc[start:end, c]; a = after_l.loc[start:end, c]
        axes[r, 0].plot(b.index, b.values, lw=0.8, color="tab:red")
        axes[r, 1].plot(a.index, a.values, lw=0.8, color="tab:blue")
        if "filled_agg" in after_l.columns:            # orange dots = minutes that were interpolated/forward-filled
            fm = after_l.loc[start:end, "filled_agg"].to_numpy(dtype=bool)
            if fm.any():
                axes[r, 1].scatter(a.index[fm], a.values[fm], s=10, color="tab:orange", zorder=3,
                                   label="filled minute" if r == 0 else None)
                if r == 0:
                    axes[r, 1].legend(fontsize=7, loc="upper right")
        axes[r, 0].set_ylabel(f"{nm}\n(W)", fontsize=8)
        if r == 0:
            axes[r, 0].set_title("BEFORE (raw, naive 1-min mean)")
            axes[r, 1].set_title("AFTER (cleaned)")
        # use the same y-range for fair comparison, driven by the cleaned data
        ymax = max(np.nanmax(a.values) if a.notna().any() else 1, 1) * 1.15
        axes[r, 1].set_ylim(-0.02 * ymax, ymax)
        if b.max() > ymax:
            axes[r, 0].annotate(f"raw max {b.max():,.0f} W", xy=(0.01, 0.82), xycoords="axes fraction",
                                color="tab:red", fontsize=8)
    fig.suptitle(title, y=1.0)
    for x in axes[-1]:
        x.tick_params(axis="x", rotation=20)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def pick_demo_day(clean: pd.DataFrame):
    """A local calendar day with a filled gap, >=95% valid aggregate and a real washer cycle (>1 kW)."""
    l = clean.tz_convert(LOCAL_TZ).tz_localize(None)
    g = l.groupby(l.index.normalize())
    ok = (g["filled_agg"].sum() > 0) & (g[AGG].apply(lambda x: x.notna().mean()) >= 0.95) \
         & (g["Appliance5"].max() > 1000) & (g.size() == 1440)
    days = ok[ok].index
    return days[len(days) // 2] if len(days) else None


# ----------------------------------------------------------------------------
def main():
    for d in (OUT_DIR, REP_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    p1 = load_part(RAW_DIR / "RAW_House1_Part1.csv")
    p2 = load_part(RAW_DIR / "RAW_House1_Part2.csv")

    summary = {"inspection": [inspect_raw(p1, "Part1"), inspect_raw(p2, "Part2")]}

    raw = stitch_parts(p1, p2)
    summary["stitch"] = {
        "part1_rows_used": int((p1["Time"] < p2["Time"].min()).sum()),
        "part2_rows_used": int(len(p2)),
        "switch_time": str(p2["Time"].min()),
    }
    del p1, p2

    clean, log = clean_pipeline(raw)
    summary["cleaning_log"] = log

    outages = outage_table(clean["outage"], clean["frozen"], dst_windows(clean.index))
    long_out = outages[outages["minutes"] > MAX_FILL_MIN]
    long_out.to_csv(REP_DIR / "house1_outages.csv", index=False)

    n = len(clean)
    summary["result"] = {
        "minutes_in_grid": n,
        "first_utc": str(clean.index.min()), "last_utc": str(clean.index.max()),
        "aggregate_minutes_filled_short_gap": int(clean["filled_agg"].sum()),
        "aggregate_minutes_in_outage": int(clean["outage"].sum()),
        "aggregate_coverage_pct": round(100 * clean[AGG].notna().mean(), 2),
        "number_of_outages": int(len(long_out)),
        "dst_clock_change_gaps": int((long_out["gap_type"] == "dst_clock_change").sum()) if len(long_out) else 0,
        "frozen_logger_gaps": int((long_out["gap_type"] == "frozen_logger").sum()) if len(long_out) else 0,
        "data_or_sensor_outages": int((long_out["gap_type"] == "data_or_sensor_outage").sum()) if len(long_out) else 0,
        "outages_over_1h": int((long_out["minutes"] > 60).sum()),
        "longest_outage_hours": round(float(long_out["minutes"].max()) / 60, 1),
        "extreme_aggregate_minutes_flagged": int(clean["extreme_agg"].sum()),
        "nan_minutes_per_channel_after_cleaning": {c: int(clean[c].isna().sum()) for c in VALUE_COLS},
        "washing_machine_(Appliance5)_pct_valid_minutes_above_20W": round(float(clean["Appliance5"].dropna().gt(20).mean() * 100), 2),
    }

    # Save cleaned table (UTC index + flags)
    clean.to_csv(OUT_DIR / "house1_1min_clean.csv.gz", index_label="timestamp_utc")

    # ---- plots ---------------------------------------------------------------
    before = naive_before(raw)
    plot_overview(before, clean, FIG_DIR / "t1_overview_aggregate_before_after.png")

    # Zoom window: 2 days around the biggest raw appliance spike that survived into the naive 1-min series
    spike_ch = "Appliance1"
    spike_t = before[spike_ch].idxmax()
    s, e = spike_t.floor("D") - pd.Timedelta(days=0), spike_t.floor("D") + pd.Timedelta(days=2)
    plot_window(before, clean, s, e, [AGG, "Appliance1", "Appliance5"],
                f"Spike removal example around {spike_t:%Y-%m-%d} (Aggregate, Fridge, Washing machine)",
                FIG_DIR / "t1_window_spikes_before_after.png")

    # Zoom window: a day with a filled gap AND a washer cycle; filled minutes are marked orange
    day = pick_demo_day(clean)
    if day is not None:
        plot_window(before, clean, day, day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1),
                    [AGG, "Appliance2", "Appliance5"],
                    f"Typical day with a short filled gap and a wash ({day:%Y-%m-%d})",
                    FIG_DIR / "t1_window_typical_day_before_after.png")

    with open(REP_DIR / "house1_quality_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({"stitch": summary["stitch"], "cleaning_log": log, "result": summary["result"]}, indent=2, default=str))
    print("\n10 longest outages:")
    print(long_out.sort_values("minutes", ascending=False).head(10).to_string(index=False))
    print("\nOutages by reason:", long_out["gap_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()