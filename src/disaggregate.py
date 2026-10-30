"""
Task 3 - Estimate washing-machine power from the household AGGREGATE (1-minute, official cleaned REFIT).

Run from the repo root, AFTER src/washer_eda.py (it re-uses that script's 1-minute cache and rules):
    python src/disaggregate.py                 # houses chosen by the same rule as Task 2
    python src/disaggregate.py --houses 1 2 5 7 --no-loho

Needs:  pandas, numpy, matplotlib, scikit-learn

Design (details and reasons are in RESULTS.md):
  * Model input  : ONLY the aggregate power (+ local hour / weekday). No other appliance channel is used,
                   because in real life they are not metered.
  * Model        : gradient boosting (sklearn HistGradientBoostingRegressor) on window features around each minute.
  * Split        : by TIME inside every house: first 60% train | 1-day buffer | next ~20% validation |
                   1-day buffer | last ~20% test. The buffers are longer than any feature window (+-60 min),
                   so no test minute shares a feature window with a training minute. The test period is
                   never used for any choice (hyper-parameters / cut-off are picked on validation).
  * Extra check  : leave-one-house-out (LOHO) = train on the other houses, test on a house the model never saw.
  * Baselines    : always-off, global-mean, and a usage-profile (mean washer power by weekday x hour,
                   learned from that house's training period).
  * Metrics      : MAE, RMSE, energy error (SAE), on/off precision-recall-F1 at 20 W, and cycle-level
                   recall / precision / energy error. Reasons are in RESULTS.md.
"""
import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import washer_Eda as w2          # same folder: re-use loading, frozen-run masking and the cycle rule

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
REP_DIR = ROOT / "reports"
FIG_DIR = REP_DIR / "figures"

ON_W = w2.ON_W                    # 20 W, the same "active" threshold as Task 2
TRAIN_FRAC, VAL_END_FRAC = 0.60, 0.80
BUFFER = pd.Timedelta(days=1)
NEG_KEEP = 0.20                   # keep 20% of 'far from any cycle' minutes for training (weight 5 restores balance)
NEAR_MIN = 60                     # minutes around an active minute that are always kept
SEED = 0
GRID = [dict(max_iter=150, max_leaf_nodes=31), dict(max_iter=300, max_leaf_nodes=31),
        dict(max_iter=300, max_leaf_nodes=63)]
CUTOFFS = (0, 5, 10, 20)          # predictions below this many watts are set to 0 (chosen on validation)


# ----------------------------------------------------------------------------
# Features: everything is computed from the aggregate only
# ----------------------------------------------------------------------------
def make_features(agg: pd.Series) -> pd.DataFrame:
    """Window features around every minute. Looks +-60 min around it (offline estimation, like Seq2Point)."""
    a = agg.astype("float32")
    f = {"agg": a}
    for k in (1, 2, 3, 5, 10, 15, 30):
        f[f"agg_m{k}"] = a.shift(k)
        f[f"agg_p{k}"] = a.shift(-k)
    for win in (5, 15, 31, 61, 121):
        r = a.rolling(win, center=True, min_periods=max(1, win // 2))
        f[f"mean{win}"], f[f"std{win}"] = r.mean(), r.std()
        f[f"max{win}"], f[f"min{win}"] = r.max(), r.min()
    f["agg_minus_base"] = a - f["min121"]                 # load above the quiet baseline of the last/next 2 h
    f["max61_minus_base"] = f["max61"] - f["min121"]
    f["step_in"] = a - a.shift(1)
    f["step_out"] = a.shift(-1) - a
    loc = w2.local_naive(a.index)
    hr = loc.hour + loc.minute / 60
    f["hour_sin"] = pd.Series(np.sin(2 * np.pi * hr / 24), index=a.index, dtype="float32")
    f["hour_cos"] = pd.Series(np.cos(2 * np.pi * hr / 24), index=a.index, dtype="float32")
    f["dow"] = pd.Series(loc.dayofweek, index=a.index, dtype="float32")
    return pd.DataFrame(f).astype("float32")


def split_bounds(index: pd.DatetimeIndex):
    t0, t1 = index[0], index[-1]
    span = t1 - t0
    train_end = t0 + TRAIN_FRAC * span
    val_start, val_end = train_end + BUFFER, t0 + VAL_END_FRAC * span
    test_start = val_end + BUFFER
    return t0, train_end, val_start, val_end, test_start, t1


def prepare_house(n: int, rng: np.random.Generator) -> dict:
    d, _ = w2.mask_stuck(w2.load_1min(n))                 # same frozen-run masking as Task 2
    agg, wash = d["Aggregate"], d["Washer"]
    feats = make_features(agg)
    t0, tr_end, va_s, va_e, te_s, t1 = split_bounds(d.index)
    valid = agg.notna() & wash.notna()
    active = (wash > ON_W).astype(float)
    near = active.rolling(2 * NEAR_MIN + 1, center=True, min_periods=1).max() > 0
    keep = near | (rng.random(len(d)) < NEG_KEEP)
    weight = np.where(near, 1.0, 1.0 / NEG_KEEP).astype("float32")

    def sub(mask):                                         # subsampled + weighted rows for fitting
        m = (mask & valid & keep).to_numpy()
        return feats[m], wash[m].to_numpy("float32"), weight[m]

    idx = d.index
    tr_mask = pd.Series((idx >= t0) & (idx < tr_end), index=idx)
    va_mask = pd.Series((idx >= va_s) & (idx < va_e), index=idx)
    te_mask = (idx >= te_s)
    return {
        "house": n, "train": sub(tr_mask), "val": sub(va_mask),
        "test_X": feats[te_mask], "test_wash": wash[te_mask], "test_agg": agg[te_mask],
        "test_issues": d["Issues"].fillna(0)[te_mask] > 0,
        "train_wash_full": wash[tr_mask & wash.notna()],   # for the usage-profile baseline
        "bounds": (t0, tr_end, va_s, va_e, te_s, t1),
    }


# ----------------------------------------------------------------------------
# Models and baselines
# ----------------------------------------------------------------------------
def fit_hgb(X, y, w, params):
    m = HistGradientBoostingRegressor(learning_rate=0.08, min_samples_leaf=50, l2_regularization=1.0,
                                      early_stopping=False, random_state=SEED, **params)
    return m.fit(X, y, sample_weight=w)


def postprocess(pred, agg, cutoff):
    """Physical clean-up: a washer cannot draw < 0 W or more than the whole house; tiny values -> 0."""
    p = np.clip(np.asarray(pred, float), 0, None)
    p[p < cutoff] = 0.0
    ag = np.asarray(agg, float)
    ok = ~np.isnan(ag)
    p[ok] = np.minimum(p[ok], ag[ok])
    return p


def weighted_mae(y, p, w):
    return float(np.average(np.abs(y - p), weights=w))


def usage_profile(train_wash: pd.Series):
    """Baseline: mean washer power for each (weekday, hour) of the house's TRAINING period."""
    loc = w2.local_naive(train_wash.index)
    return train_wash.groupby([loc.dayofweek, loc.hour]).mean()


def profile_predict(prof: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    loc = w2.local_naive(index)
    key = pd.MultiIndex.from_arrays([loc.dayofweek, loc.hour])
    return prof.reindex(key).fillna(0.0).to_numpy()


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def minute_metrics(y, p, thr=ON_W) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    yt, pt = y > thr, p > thr
    tp, fp, fn = int((yt & pt).sum()), int((~yt & pt).sum()), int((yt & ~pt).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if tp and prec + rec else 0.0
    return {"MAE_w": np.abs(y - p).mean(), "RMSE_w": float(np.sqrt(((y - p) ** 2).mean())),
            "energy_true_kwh": y.sum() / 60000, "energy_pred_kwh": p.sum() / 60000,
            "energy_bias_pct": 100 * (p.sum() - y.sum()) / y.sum() if y.sum() else np.nan,
            "SAE_pct": 100 * abs(p.sum() - y.sum()) / y.sum() if y.sum() else np.nan,
            "precision": prec, "recall": rec, "F1": f1}


def cycle_metrics(true_wash: pd.Series, pred: np.ndarray, agg: pd.Series):
    """Cycle-level view: find real cycles with the Task-2 rule, then ask how well the prediction recovers each."""
    t = pd.DataFrame({"Washer": true_wash, "Issues": 0.0})
    tc = w2.detect_cycles(t)
    tc = tc[~tc["truncated"]].reset_index(drop=True)
    pdf = pd.DataFrame({"Washer": np.where(true_wash.notna() & agg.notna(), pred, np.nan), "Issues": 0.0}, index=true_wash.index)
    pc = w2.detect_cycles(pdf)
    p_arr = np.nan_to_num(pdf["Washer"].to_numpy(float))
    rows = []
    for _, r in tc.iterrows():
        i0, i1 = int(r["i0"]), int(r["i1"])
        e_pred = p_arr[i0:i1 + 1].sum() / 60000
        hit = bool(((pc["i0"] <= i1) & (pc["i1"] >= i0)).any()) if len(pc) else False
        other = (agg.to_numpy(float)[i0:i1 + 1] - true_wash.to_numpy(float)[i0:i1 + 1])
        rows.append({"start": r["start"], "duration_min": r["duration_min"], "energy_true_kwh": r["energy_kwh"],
                     "energy_pred_kwh": e_pred, "rel_err": (e_pred - r["energy_kwh"]) / r["energy_kwh"],
                     "detected": hit, "other_load_w": float(np.nanmedian(np.clip(other, 0, None)))})
    cyc = pd.DataFrame(rows)
    p_prec = np.nan
    if len(pc):
        p_prec = float(np.mean([((tc["i0"] <= r["i1"]) & (tc["i1"] >= r["i0"])).any() if len(tc) else False
                                for _, r in pc.iterrows()]))
    return cyc, {"true_cycles": len(tc), "pred_cycles": len(pc),
                 "cycle_recall": float(cyc["detected"].mean()) if len(cyc) else np.nan, "cycle_precision": p_prec,
                 "median_abs_cycle_energy_err_pct": 100 * float(cyc["rel_err"].abs().median()) if len(cyc) else np.nan}


def evaluate(h: dict, preds: dict, label_extra=None):
    """preds: model name -> prediction array for the full test slice. Returns metric rows and cycle tables."""
    rows, cyc_tables = [], {}
    ok = (h["test_wash"].notna() & h["test_agg"].notna()).to_numpy()
    no_issue = ok & ~h["test_issues"].to_numpy()
    for name, p in preds.items():
        for subset, m in (("all_valid_minutes", ok), ("excluding_Issues_minutes", no_issue)):
            if m.sum() == 0:
                continue
            r = {"house": h["house"], "model": name, "subset": subset, "n_minutes": int(m.sum()),
                 **minute_metrics(h["test_wash"].to_numpy(float)[m], p[m])}
            if subset == "all_valid_minutes":
                cyc, cm = cycle_metrics(h["test_wash"], p, h["test_agg"])
                r.update(cm); cyc_tables[name] = cyc
            rows.append(r)
    return rows, cyc_tables


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def local_day_start_utc(ts) -> pd.Timestamp:
    """Start (00:00 UK time) of the local day containing a naive-UTC timestamp, returned as naive UTC."""
    loc = w2.local_naive(pd.DatetimeIndex([pd.Timestamp(ts)]))[0].normalize()
    return loc.tz_localize(w2.LOCAL_TZ).tz_convert("UTC").tz_localize(None)


def plot_day(h, preds, day_start, path, title):
    sl = slice(day_start, day_start + pd.Timedelta(hours=23, minutes=59))
    x = w2.local_naive(h["test_agg"].loc[sl].index)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.6]})
    ax[0].plot(x, h["test_agg"].loc[sl].values, color="tab:blue", lw=1); ax[0].set_ylabel("Aggregate (W)")
    ax[1].plot(x, h["test_wash"].loc[sl].values, color="black", lw=1.4, label="actual washer")
    idx_all = h["test_agg"].index
    pos = idx_all.get_indexer(h["test_agg"].loc[sl].index)
    ax[1].plot(x, preds["model"][pos], color="tab:red", lw=1.1, label="model estimate")
    ax[1].plot(x, preds["usage_profile"][pos], color="tab:green", lw=1, ls="--", label="usage-profile baseline")
    ax[1].set_ylabel("Washing machine (W)"); ax[1].set_xlabel("UK local time"); ax[1].legend(loc="upper right", fontsize=8)
    ax[0].set_title(title)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_metrics(res: pd.DataFrame, path):
    r = res[(res["subset"] == "all_valid_minutes") & res["model"].isin(["always_off", "usage_profile", "model"])]
    houses = sorted(r["house"].unique())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (col, lab) in zip(ax, [("F1", "on/off F1 at 20 W (higher is better)"), ("MAE_w", "MAE (W, lower is better)"),
                                  ("SAE_pct", "total-energy error % (lower is better)")]):
        wd = 0.27
        for j, mdl in enumerate(["always_off", "usage_profile", "model"]):
            v = [r[(r["house"] == hh) & (r["model"] == mdl)][col].iloc[0] for hh in houses]
            a.bar(np.arange(len(houses)) + (j - 1) * wd, v, wd, label=mdl)
        a.set_xticks(range(len(houses))); a.set_xticklabels([f"H{hh}" for hh in houses]); a.set_title(lab)
    ax[0].legend(fontsize=8)
    fig.suptitle("Test period, per house: model vs baselines", y=1.0)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_cycle_scatter(cyc: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for hh, g in cyc.groupby("house"):
        ax.scatter(g["energy_true_kwh"], g["energy_pred_kwh"], s=10, alpha=0.6, label=f"H{hh}")
    m = max(cyc["energy_true_kwh"].max(), cyc["energy_pred_kwh"].max()) * 1.05
    ax.plot([0, m], [0, m], "k--", lw=1); ax.set_xlabel("actual cycle energy (kWh)"); ax.set_ylabel("estimated (kWh)")
    ax.set_title("Per-cycle energy, test period"); ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------
def choose_houses(args) -> list:
    if args.houses:
        return sorted(args.houses)
    av = pd.read_csv(REP_DIR / "t2_availability.csv")
    ok = av[(av.coverage_pct >= w2.MIN_COVERAGE_PCT) & (av.issues_pct <= w2.MAX_ISSUES_PCT) & (av.cycles_usable >= w2.MIN_CYCLES)]
    return sorted(ok.sort_values("cycles_usable", ascending=False).head(w2.MAX_HOUSES)["house"].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", type=int, nargs="*")
    ap.add_argument("--no-loho", action="store_true", help="skip the leave-one-house-out check (saves time)")
    args = ap.parse_args()
    REP_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t_start = time.time()

    houses = choose_houses(args)
    print(f"Houses: {houses}")
    H = {}
    for n in houses:
        H[n] = prepare_house(n, rng)
        b = H[n]["bounds"]
        print(f"  House {n}: train {b[0].date()}..{b[1].date()} | val {b[2].date()}..{b[3].date()} | test {b[4].date()}..{b[5].date()}"
              f" | fit rows train {len(H[n]['train'][1]):,} val {len(H[n]['val'][1]):,}")

    # ---- pick hyper-parameters on VALIDATION only
    Xtr = pd.concat([H[n]["train"][0] for n in houses]); ytr = np.concatenate([H[n]["train"][1] for n in houses])
    wtr = np.concatenate([H[n]["train"][2] for n in houses])
    Xva = pd.concat([H[n]["val"][0] for n in houses]); yva = np.concatenate([H[n]["val"][1] for n in houses])
    wva = np.concatenate([H[n]["val"][2] for n in houses])
    best = None
    for params in GRID:
        m = fit_hgb(Xtr, ytr, wtr, params)
        pv = postprocess(m.predict(Xva), Xva["agg"].to_numpy(), 0)
        mae = weighted_mae(yva, pv, wva)
        print(f"  grid {params}: validation MAE {mae:.2f} W")
        if best is None or mae < best[0]:
            best = (mae, params, m)
    _, best_params, model = best
    pv_raw = model.predict(Xva)
    cut_scores = {c: weighted_mae(yva, postprocess(pv_raw, Xva["agg"].to_numpy(), c), wva) for c in CUTOFFS}
    cutoff = min(cut_scores, key=cut_scores.get)
    print(f"Chosen on validation: {best_params}, low-power cut-off {cutoff} W")
    # feature importance: permutation on a validation sample
    from sklearn.inspection import permutation_importance
    samp = np.random.default_rng(SEED).choice(len(yva), size=min(40000, len(yva)), replace=False)
    pi = permutation_importance(model, Xva.iloc[samp], yva[samp], n_repeats=3, random_state=SEED,
                                scoring="neg_mean_absolute_error", sample_weight=wva[samp])
    pd.Series(pi.importances_mean, index=Xtr.columns).sort_values(ascending=False).round(3).to_csv(
        REP_DIR / "t3_feature_importance.csv", header=["mae_increase_w_when_shuffled"])

    # ---- evaluate on each house's TEST period
    all_rows, cyc_all, test_preds = [], [], {}
    for n in houses:
        h = H[n]
        X = h["test_X"]
        p_model = postprocess(model.predict(X), h["test_agg"].to_numpy(), cutoff)
        prof = usage_profile(h["train_wash_full"])
        preds = {"always_off": np.zeros(len(X)), "global_mean": np.full(len(X), float(ytr.mean())),
                 "usage_profile": profile_predict(prof, X.index), "model": p_model}
        test_preds[n] = preds
        rows, cyc = evaluate(h, preds)
        all_rows += rows
        c = cyc["model"].copy(); c.insert(0, "house", n); cyc_all.append(c)
    res = pd.DataFrame(all_rows)

    # ---- leave-one-house-out: the harder question (a house the model has never seen)
    if not args.no_loho and len(houses) > 1:
        loho = []
        for n in houses:
            others = [k for k in houses if k != n]
            X = pd.concat([H[k]["train"][0] for k in others] + [H[k]["val"][0] for k in others])
            y = np.concatenate([H[k]["train"][1] for k in others] + [H[k]["val"][1] for k in others])
            wt = np.concatenate([H[k]["train"][2] for k in others] + [H[k]["val"][2] for k in others])
            m = fit_hgb(X, y, wt, best_params)
            h = H[n]
            p = postprocess(m.predict(h["test_X"]), h["test_agg"].to_numpy(), cutoff)
            rows, _ = evaluate(h, {"model_leave_one_house_out": p})
            loho += rows
            print(f"  LOHO house {n} done")
        res = pd.concat([res, pd.DataFrame(loho)], ignore_index=True)
    res.to_csv(REP_DIR / "t3_metrics.csv", index=False)

    cyc_df = pd.concat(cyc_all, ignore_index=True)
    cyc_df.to_csv(REP_DIR / "t3_test_cycles.csv", index=False)

    # ---- console summary
    show = ["house", "model", "n_minutes", "MAE_w", "RMSE_w", "SAE_pct", "precision", "recall", "F1",
            "true_cycles", "cycle_recall", "cycle_precision", "median_abs_cycle_energy_err_pct"]
    main_tbl = res[res["subset"] == "all_valid_minutes"][show].round(2)
    print("\n=== Test-period metrics (all valid minutes) ===")
    print(main_tbl.to_string(index=False))
    pooled = []
    for mdl in main_tbl["model"].unique():
        g = res[(res["subset"] == "all_valid_minutes") & (res["model"] == mdl)]
        pooled.append({"model": mdl, "mean_MAE_w": g["MAE_w"].mean(), "mean_F1": g["F1"].mean(),
                       "mean_SAE_pct": g["SAE_pct"].mean(), "mean_cycle_recall": g["cycle_recall"].mean(),
                       "mean_cycle_energy_err_pct": g["median_abs_cycle_energy_err_pct"].mean()})
    pooled = pd.DataFrame(pooled).round(2)
    pooled.to_csv(REP_DIR / "t3_metrics_pooled.csv", index=False)
    print("\n=== Mean over houses ===")
    print(pooled.to_string(index=False))
    ex = res[(res["model"] == "model")].pivot_table(index="house", columns="subset", values=["MAE_w", "F1"]).round(3)
    print("\n=== Model with / without minutes flagged 'Issues' ===")
    print(ex.to_string())

    # ---- failure analysis: does error grow when other loads run during the cycle?
    c = cyc_df.dropna(subset=["other_load_w"]).copy()
    c["abs_err_pct"] = 100 * c["rel_err"].abs()
    c["other_load_band"] = pd.qcut(c["other_load_w"], 3, labels=["low", "mid", "high"], duplicates="drop")
    fa = c.groupby("other_load_band", observed=True).agg(n=("abs_err_pct", "size"), other_load_median_w=("other_load_w", "median"),
                                                          median_abs_energy_err_pct=("abs_err_pct", "median"),
                                                          cycle_recall=("detected", "mean")).round(2)
    fa.to_csv(REP_DIR / "t3_failure_by_other_load.csv")
    print("\n=== Failure analysis: cycle error vs. other load running at the same time ===")
    print(fa.to_string())

    # ---- plots
    plot_metrics(res, FIG_DIR / "t3_metrics_by_house.png")
    plot_cycle_scatter(cyc_df, FIG_DIR / "t3_cycle_energy_scatter.png")
    best_house = cyc_df["house"].value_counts().idxmax()
    cb = cyc_df[cyc_df["house"] == best_house]
    typical = cb.iloc[(cb["rel_err"].abs() - cb["rel_err"].abs().median()).abs().argsort().iloc[0]]
    d0 = local_day_start_utc(typical["start"])
    plot_day(H[best_house], test_preds[best_house], d0, FIG_DIR / "t3_actual_vs_predicted.png",
             f"House {best_house}, test period, {w2.local_naive(pd.DatetimeIndex([d0]))[0]:%a %d %b %Y}: actual vs estimated "
             f"washing-machine power (a typical-error cycle)")
    worst = c.sort_values("abs_err_pct", ascending=False).iloc[0]
    wh = int(worst["house"])
    d1 = local_day_start_utc(worst["start"])
    plot_day(H[wh], test_preds[wh], d1, FIG_DIR / "t3_failure_case.png",
             f"Failure case - House {wh}, {w2.local_naive(pd.DatetimeIndex([d1]))[0]:%a %d %b %Y}: cycle energy error "
             f"{worst['abs_err_pct']:.0f}%, other load during the cycle ~{worst['other_load_w']:.0f} W")
    print(f"\nTotal runtime {(time.time() - t_start) / 60:.1f} min. Outputs: reports/t3_*.csv, reports/figures/t3_*.png")


if __name__ == "__main__":
    main()