"""
Layer D — Price analysis (SECONDARY supporting evidence)
=========================================================
This layer examines whether CBAM exposure correlates with carbon pricing
levels.  It is explicitly secondary to the adoption (Layer A), sectoral
design (Layer B), and entrant-level (Layer C) results that form the
paper's core contribution.

INTERPRETATION NOTE — EXCHANGE RATES & PRICE DATA
--------------------------------------------------
Prices in cpi_panel are nominal USD converted from local currency at
April 1 exchange rates.  Exchange rate movements are embedded in USD
values.  A jurisdiction whose currency appreciates against USD will
mechanically show higher USD prices without any policy change.

Therefore:
  (a) Primary specification: log nominal price in USD with the caveat
      stated explicitly in all output table footers.
  (b) Preferred robustness: log real price using country-own GDP
      deflator (linked series) from controls_panel.  This removes
      within-country price level changes but does not remove
      cross-country exchange rate effects.
  (c) Interpretation of exposure interaction coefficients should
      remain modest given this limitation.

Run from project root:
    python3 src/analyse_layer_d_price.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = os.path.join(ROOT, "data", "output")
OUT_DIR = os.path.join(DATA_OUT, "layer_d")
os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────

def _stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def _fmt_coef(b, se, p):
    return f"{b:+.4f}{_stars(p)} ({se:.4f})"


# ========================================================================
# STEP 0 — Load data
# ========================================================================
t0 = time.time()
print("=" * 72)
print("LAYER D — Price analysis")
print("=" * 72)

cpi = pd.read_csv(os.path.join(DATA_OUT, "cpi_panel.csv"))
controls = pd.read_csv(os.path.join(DATA_OUT, "controls_panel.csv"))
exposure_annual = pd.read_csv(os.path.join(DATA_OUT, "cbam_exposure_annual.csv"))

print(f"  cpi_panel:           {len(cpi):>6,} rows")
print(f"  controls_panel:      {len(controls):>6,} rows")
print(f"  cbam_exposure_annual:{len(exposure_annual):>6,} rows")

# ========================================================================
# STEP 1 — Price sample: implemented, non-null price, 2016-2024,
#           drop EEA-ETS extensions
# ========================================================================
print("\n" + "=" * 72)
print("STEP 1 — Price sample construction")
print("=" * 72)

df = cpi.copy()
n0 = len(df)

df = df[df["price_usd"].notna()].copy()
print(f"  After drop null price_usd:     {len(df):>5,}  (dropped {n0 - len(df):,})")

n1 = len(df)
df = df[df["status_implemented"] == 1].copy()
print(f"  After keep implemented only:   {len(df):>5,}  (dropped {n1 - len(df):,})")

n2 = len(df)
df = df[(df["year"] >= 2016) & (df["year"] <= 2024)].copy()
print(f"  After keep 2016-2024:          {len(df):>5,}  (dropped {n2 - len(df):,})")

n3 = len(df)
df = df[df["is_eea_ets_extension"] != 1].copy()
print(f"  After drop EEA-ETS extensions: {len(df):>5,}  (dropped {n3 - len(df):,})")

# ========================================================================
# STEP 2 — National primary, all as robustness
# ========================================================================
print("\n" + "=" * 72)
print("STEP 2 — National vs all subsamples")
print("=" * 72)

national_price_sample = df[df["scope"] == "National"].copy()
all_price_sample = df.copy()

print(f"  national_price_sample: {len(national_price_sample):>5,} rows")
print(f"  all_price_sample:      {len(all_price_sample):>5,} rows")
print(f"  subnational only:      {len(all_price_sample) - len(national_price_sample):>5,} rows")

# ========================================================================
# STEP 3 — Merge controls (including deflator)
# ========================================================================
print("\n" + "=" * 72)
print("STEP 3 — Merge controls")
print("=" * 72)

ctrl_cols = ["iso3", "year",
             "log_gdp_pc_constant2015_lag1",
             "wgi_composite_estimate_lag1",
             "gdp_deflator_linked",
             "gdp_deflator_growth_linked"]
ctrl = controls[ctrl_cols].copy()

for sample_name, sample_df in [("national_price_sample", national_price_sample),
                                ("all_price_sample", all_price_sample)]:
    n_pre = len(sample_df)
    merged = sample_df.merge(ctrl, on=["iso3", "year"], how="left")
    assert len(merged) == n_pre, f"Merge changed row count for {sample_name}"
    if sample_name == "national_price_sample":
        national_price_sample = merged
    else:
        all_price_sample = merged

print(f"  national after merge: {len(national_price_sample):>5,} rows")
print(f"  all after merge:      {len(all_price_sample):>5,} rows")

# ========================================================================
# STEP 4 — Construct real price
# ========================================================================
print("\n" + "=" * 72)
print("STEP 4 — Real price construction")
print("=" * 72)

for sample_name, sdf in [("national_price_sample", national_price_sample),
                          ("all_price_sample", all_price_sample)]:
    sdf["real_price_usd"] = np.where(
        sdf["gdp_deflator_linked"].notna(),
        sdf["price_usd"] / (sdf["gdp_deflator_linked"] / 100),
        np.nan
    )
    sdf["log_real_price_usd"] = np.where(
        sdf["real_price_usd"].notna(),
        np.log(sdf["real_price_usd"] + 1),
        np.nan
    )
    n_real = sdf["real_price_usd"].notna().sum()
    print(f"  {sample_name}: {n_real:>5,} / {len(sdf):>5,} rows with real price")

# ========================================================================
# STEP 5 — Drop null exposure (EU/EFTA instruments)
# ========================================================================
print("\n" + "=" * 72)
print("STEP 5 — Drop null exposure rows")
print("=" * 72)

for label, sdf in [("national", national_price_sample),
                    ("all", all_price_sample)]:
    n_pre = len(sdf)
    mask = sdf["cbam_exposure_avg"].notna()
    if label == "national":
        national_price_sample = sdf[mask].copy()
        print(f"  national: {len(national_price_sample):>5,}  (dropped {n_pre - len(national_price_sample):,} null exposure)")
    else:
        all_price_sample = sdf[mask].copy()
        print(f"  all:      {len(all_price_sample):>5,}  (dropped {n_pre - len(all_price_sample):,} null exposure)")

# ========================================================================
# STEP 6 — Instrument-type subsamples (national only)
# ========================================================================
print("\n" + "=" * 72)
print("STEP 6 — Instrument-type subsamples")
print("=" * 72)

ets_sample = national_price_sample[
    national_price_sample["instrument_type"] == "ETS"
].copy()
tax_sample = national_price_sample[
    national_price_sample["instrument_type"] == "Carbon tax"
].copy()

print(f"  ets_sample (national): {len(ets_sample):>5,} rows")
print(f"  tax_sample (national): {len(tax_sample):>5,} rows")

# ========================================================================
# Merge time-varying exposure for Spec 3
# ========================================================================
print("\n  Merging time-varying exposure …")

exp_ann = exposure_annual[["iso3", "t", "cbam_exposure_annual"]].rename(
    columns={"t": "year", "cbam_exposure_annual": "cbam_exposure_annual_tv"}
)

for label in ["national_price_sample", "all_price_sample",
              "ets_sample", "tax_sample"]:
    sdf = locals()[label]
    n_pre = len(sdf)
    sdf = sdf.merge(exp_ann, on=["iso3", "year"], how="left")
    # If cpi_panel already had cbam_exposure_annual, prefer the merged version
    # for time-varying spec; the column from cpi_panel may differ from the
    # dedicated file, so we keep both.
    assert len(sdf) == n_pre, f"Exposure annual merge changed rows for {label}"
    locals()[label] = sdf

# Re-assign after locals() trick
national_price_sample = locals()["national_price_sample"]
all_price_sample = locals()["all_price_sample"]
ets_sample = locals()["ets_sample"]
tax_sample = locals()["tax_sample"]

print("  Done.\n")

# ── Post indicators ─────────────────────────────────────────────────────
POST_YEARS = [2019, 2021, 2023]

for sdf in [national_price_sample, all_price_sample, ets_sample, tax_sample]:
    for py in POST_YEARS:
        col = f"post_{py}"
        if col not in sdf.columns:
            sdf[col] = (sdf["year"] >= py).astype(int)

# ── Control variables ────────────────────────────────────────────────────
CONTROLS = ["log_gdp_pc_constant2015_lag1", "wgi_composite_estimate_lag1"]

# ========================================================================
# REGRESSION ENGINE
# ========================================================================

def run_price_reg(sdf, dep_var, exog_vars, entity_col="cpi_id",
                  time_col="year", cluster_col="iso3",
                  drop_absorbed=True):
    """Run PanelOLS with CPI-level FE, year FE, clustered at country level."""
    reg_cols = [dep_var] + exog_vars + [entity_col, time_col, cluster_col]
    reg_df = sdf[reg_cols].dropna(subset=[dep_var] + exog_vars).copy()

    if len(reg_df) < 10:
        return None

    reg_df = reg_df.set_index([entity_col, time_col])
    dep = reg_df[dep_var]
    exog = reg_df[exog_vars]
    cluster_df = pd.DataFrame({"cluster": reg_df[cluster_col].values},
                              index=reg_df.index)

    try:
        mod = PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                       drop_absorbed=drop_absorbed)
        res = mod.fit(cov_type="clustered", clusters=cluster_df)
        return res
    except Exception as e:
        print(f"    [WARN] Regression failed: {e}")
        return None


# ========================================================================
# TABLE 1 — Price descriptive by year
# ========================================================================
print("=" * 72)
print("TABLE 1 — Price descriptive by year (national implemented CPIs)")
print("=" * 72)

desc_df = national_price_sample.copy()
rows_t1 = []
for yr in range(2016, 2025):
    yd = desc_df[desc_df["year"] == yr]
    if len(yd) == 0:
        continue
    rows_t1.append({
        "year": yr,
        "n_cpis": len(yd),
        "mean_price_usd": round(yd["price_usd"].mean(), 2),
        "median_price_usd": round(yd["price_usd"].median(), 2),
        "p90_price_usd": round(yd["price_usd"].quantile(0.9), 2),
        "n_ets": (yd["instrument_type"] == "ETS").sum(),
        "n_tax": (yd["instrument_type"] == "Carbon tax").sum(),
        "mean_price_ets": round(yd.loc[yd["instrument_type"] == "ETS", "price_usd"].mean(), 2)
            if (yd["instrument_type"] == "ETS").any() else np.nan,
        "mean_price_tax": round(yd.loc[yd["instrument_type"] == "Carbon tax", "price_usd"].mean(), 2)
            if (yd["instrument_type"] == "Carbon tax").any() else np.nan,
    })

table1 = pd.DataFrame(rows_t1)
table1.to_csv(os.path.join(OUT_DIR, "table1_price_descriptive.csv"), index=False)
print(table1.to_string(index=False))

# ========================================================================
# REGRESSIONS — Specs 1-4 across subsamples
# ========================================================================
print("\n" + "=" * 72)
print("REGRESSIONS")
print("=" * 72)

SAMPLES = {
    "national": national_price_sample,
    "all": all_price_sample,
    "ets": ets_sample,
    "tax": tax_sample,
}

# ── Spec 1: Exposure main effect (no post interaction) ──────────────────
# NOTE: cbam_exposure_avg is time-invariant per CPI, so entity FEs absorb it.
# Run without entity FEs (year FE only) to identify the cross-sectional level
# effect, or note absorption. We run with entity_effects=False for this spec.
print("\n--- Spec 1: Exposure main effect (year FE only — cross-sectional) ---")
spec1_results = {}
for sname, sdf in SAMPLES.items():
    exog_vars = ["cbam_exposure_avg"] + CONTROLS
    reg_cols = ["log_price_usd"] + exog_vars + ["cpi_id", "year", "iso3"]
    reg_df = sdf[reg_cols].dropna(subset=["log_price_usd"] + exog_vars).copy()
    if len(reg_df) < 10:
        spec1_results[sname] = None
        print(f"  {sname:>10s}: insufficient data")
        continue
    reg_df = reg_df.set_index(["cpi_id", "year"])
    cluster_df = pd.DataFrame({"cluster": reg_df["iso3"].values}, index=reg_df.index)
    try:
        mod = PanelOLS(reg_df["log_price_usd"], reg_df[exog_vars],
                       entity_effects=False, time_effects=True,
                       drop_absorbed=True)
        res = mod.fit(cov_type="clustered", clusters=cluster_df)
        spec1_results[sname] = res
        b = res.params.get("cbam_exposure_avg", np.nan)
        se = res.std_errors.get("cbam_exposure_avg", np.nan)
        p = res.pvalues.get("cbam_exposure_avg", np.nan)
        print(f"  {sname:>10s}: exposure = {_fmt_coef(b, se, p)}  N={int(res.nobs)}")
    except Exception as e:
        spec1_results[sname] = None
        print(f"  {sname:>10s}: regression failed: {e}")
print("  (No entity FE — cbam_exposure_avg is time-invariant, absorbed by CPI FE)")

# ── Spec 2: Exposure x Post interaction (primary) ───────────────────────
print("\n--- Spec 2: Exposure × Post interaction (nominal) ---")
spec2_results = {}  # (sample, post_year) -> result
for sname, sdf in SAMPLES.items():
    for py in POST_YEARS:
        interaction = f"exposure_x_post{py}"
        sdf[interaction] = sdf["cbam_exposure_avg"] * sdf[f"post_{py}"]
        exog_vars = [interaction] + CONTROLS
        res = run_price_reg(sdf, "log_price_usd", exog_vars)
        spec2_results[(sname, py)] = res
        if res is not None:
            b = res.params.get(interaction, np.nan)
            se = res.std_errors.get(interaction, np.nan)
            p = res.pvalues.get(interaction, np.nan)
            print(f"  {sname:>10s} Post-{py}: {_fmt_coef(b, se, p)}  N={int(res.nobs)}")
        else:
            print(f"  {sname:>10s} Post-{py}: insufficient data")

# ── Spec 3: Time-varying exposure × Post ────────────────────────────────
print("\n--- Spec 3: Time-varying exposure × Post ---")
spec3_results = {}
for sname, sdf in SAMPLES.items():
    for py in POST_YEARS:
        interaction = f"exposure_tv_x_post{py}"
        sdf[interaction] = sdf["cbam_exposure_annual_tv"].fillna(0) * sdf[f"post_{py}"]
        exog_vars = [interaction] + CONTROLS
        # Only run if we have time-varying exposure
        if sdf["cbam_exposure_annual_tv"].notna().sum() < 10:
            spec3_results[(sname, py)] = None
            print(f"  {sname:>10s} Post-{py}: insufficient time-varying exposure")
            continue
        res = run_price_reg(sdf, "log_price_usd", exog_vars)
        spec3_results[(sname, py)] = res
        if res is not None:
            b = res.params.get(interaction, np.nan)
            se = res.std_errors.get(interaction, np.nan)
            p = res.pvalues.get(interaction, np.nan)
            print(f"  {sname:>10s} Post-{py}: {_fmt_coef(b, se, p)}  N={int(res.nobs)}")
        else:
            print(f"  {sname:>10s} Post-{py}: insufficient data")

# ── Spec 4: Real price robustness ───────────────────────────────────────
print("\n--- Spec 4: Real price robustness (Post-2021 & Post-2023) ---")
spec4_results = {}  # (sample, post_year, dep_var) -> result
for sname, sdf in SAMPLES.items():
    for py in [2021, 2023]:
        interaction = f"exposure_x_post{py}"
        # Ensure interaction exists
        if interaction not in sdf.columns:
            sdf[interaction] = sdf["cbam_exposure_avg"] * sdf[f"post_{py}"]
        exog_vars = [interaction] + CONTROLS
        # Nominal (re-run for side-by-side comparison)
        res_nom = run_price_reg(sdf, "log_price_usd", exog_vars)
        spec4_results[(sname, py, "nominal")] = res_nom
        # Real
        res_real = run_price_reg(sdf, "log_real_price_usd", exog_vars)
        spec4_results[(sname, py, "real")] = res_real
        if res_nom is not None and res_real is not None:
            b_n = res_nom.params.get(interaction, np.nan)
            b_r = res_real.params.get(interaction, np.nan)
            print(f"  {sname:>10s} Post-{py}: nominal={b_n:+.4f}  real={b_r:+.4f}  "
                  f"N_nom={int(res_nom.nobs)} N_real={int(res_real.nobs)}")
        else:
            print(f"  {sname:>10s} Post-{py}: insufficient data")

# ========================================================================
# TABLE 2 — Nominal price regressions (national primary)
# ========================================================================
print("\n" + "=" * 72)
print("TABLE 2 — Nominal price regressions")
print("=" * 72)


def _extract_row(res, var_name):
    """Extract formatted coefficient info from regression result."""
    if res is None:
        return {"coef": "", "N": ""}
    b = res.params.get(var_name, np.nan)
    se = res.std_errors.get(var_name, np.nan)
    p = res.pvalues.get(var_name, np.nan)
    return {"coef": _fmt_coef(b, se, p), "N": str(int(res.nobs))}


t2_rows = []

# Spec 1 row
for sname, col_label in [("national", "National all"),
                          ("ets", "National ETS"),
                          ("tax", "National tax")]:
    res = spec1_results.get(sname)
    info = _extract_row(res, "cbam_exposure_avg")
    t2_rows.append({
        "spec": "Spec 1: exposure level",
        "sample": col_label,
        "coefficient": info["coef"],
        "N": info["N"],
    })

# Spec 2 rows
for py in POST_YEARS:
    interaction = f"exposure_x_post{py}"
    for sname, col_label in [("national", "National all"),
                              ("ets", "National ETS"),
                              ("tax", "National tax")]:
        res = spec2_results.get((sname, py))
        info = _extract_row(res, interaction)
        t2_rows.append({
            "spec": f"Spec 2: exposure × Post-{py}",
            "sample": col_label,
            "coefficient": info["coef"],
            "N": info["N"],
        })

table2 = pd.DataFrame(t2_rows)
# Add footer row
footer = pd.DataFrame([{
    "spec": "NOTE",
    "sample": "",
    "coefficient": "Nominal USD prices. Exchange rate movements embedded in USD values.",
    "N": "",
}])
table2_out = pd.concat([table2, footer], ignore_index=True)
table2_out.to_csv(os.path.join(OUT_DIR, "table2_price_nominal.csv"), index=False)
print(table2.to_string(index=False))

# ========================================================================
# TABLE 3 — Real price robustness
# ========================================================================
print("\n" + "=" * 72)
print("TABLE 3 — Real price robustness")
print("=" * 72)

t3_rows = []
for py in [2021, 2023]:
    interaction = f"exposure_x_post{py}"
    for sname, col_label in [("national", "National all"),
                              ("ets", "National ETS"),
                              ("tax", "National tax")]:
        res_n = spec4_results.get((sname, py, "nominal"))
        res_r = spec4_results.get((sname, py, "real"))
        info_n = _extract_row(res_n, interaction)
        info_r = _extract_row(res_r, interaction)
        t3_rows.append({
            "post": f"Post-{py}",
            "sample": col_label,
            "nominal_coef": info_n["coef"],
            "nominal_N": info_n["N"],
            "real_coef": info_r["coef"],
            "real_N": info_r["N"],
        })

table3 = pd.DataFrame(t3_rows)
footer3 = pd.DataFrame([{
    "post": "NOTE",
    "sample": "",
    "nominal_coef": ("Real price = nominal USD / (GDP deflator / 100). "
                     "GDP deflator: WDI linked series (NY.GDP.DEFL.ZS.AD). "
                     "Within-country price changes removed but cross-country "
                     "exchange rate effects remain."),
    "nominal_N": "",
    "real_coef": "",
    "real_N": "",
}])
table3_out = pd.concat([table3, footer3], ignore_index=True)
table3_out.to_csv(os.path.join(OUT_DIR, "table3_price_real_robustness.csv"), index=False)
print(table3.to_string(index=False))

# ========================================================================
# TABLE 4 — Time-varying exposure results
# ========================================================================
print("\n" + "=" * 72)
print("TABLE 4 — Time-varying exposure")
print("=" * 72)

t4_rows = []
for py in POST_YEARS:
    interaction = f"exposure_tv_x_post{py}"
    for sname, col_label in [("national", "National all"),
                              ("all", "All CPIs"),
                              ("ets", "National ETS"),
                              ("tax", "National tax")]:
        res = spec3_results.get((sname, py))
        info = _extract_row(res, interaction)
        t4_rows.append({
            "post": f"Post-{py}",
            "sample": col_label,
            "coefficient": info["coef"],
            "N": info["N"],
        })

table4 = pd.DataFrame(t4_rows)
table4.to_csv(os.path.join(OUT_DIR, "table4_price_timevary_exposure.csv"), index=False)
print(table4.to_string(index=False))

# ========================================================================
# TABLE 5 — National vs all CPIs comparison (Spec 2 Post-2021)
# ========================================================================
print("\n" + "=" * 72)
print("TABLE 5 — National vs all CPIs (Spec 2, Post-2021)")
print("=" * 72)

interaction_21 = "exposure_x_post2021"
t5_rows = []
for sname, col_label in [("national", "National CPIs only"),
                          ("all", "All CPIs (national + subnational)")]:
    res = spec2_results.get((sname, 2021))
    info = _extract_row(res, interaction_21)
    t5_rows.append({
        "sample": col_label,
        "coefficient": info["coef"],
        "N": info["N"],
    })

table5 = pd.DataFrame(t5_rows)
table5.to_csv(os.path.join(OUT_DIR, "table5_national_vs_all.csv"), index=False)
print(table5.to_string(index=False))

# ========================================================================
# VALIDATION
# ========================================================================
print("\n" + "=" * 72)
print("VALIDATION")
print("=" * 72)

# 1. Price sample composition
print("\n--- V1: Price sample composition ---")
print(f"  Total CPI-year rows (national): {len(national_price_sample):,}")
print(f"  Total CPI-year rows (all):      {len(all_price_sample):,}")
print(f"  Subnational rows:               "
      f"{len(all_price_sample) - len(national_price_sample):,}")

print("\n  National sample by year:")
for yr in range(2016, 2025):
    n = (national_price_sample["year"] == yr).sum()
    if n > 0:
        print(f"    {yr}: {n:>4,} rows")

n_real_nat = national_price_sample["real_price_usd"].notna().sum()
print(f"\n  Rows with non-null real price (national): {n_real_nat:,} / {len(national_price_sample):,}")
print(f"  Rows with non-null real price (all):      "
      f"{all_price_sample['real_price_usd'].notna().sum():,} / {len(all_price_sample):,}")

# 2. Price range check
print("\n--- V2: Price range check ---")
for vname, col in [("price_usd", "price_usd"),
                    ("real_price_usd", "real_price_usd")]:
    vals = national_price_sample[col].dropna()
    if len(vals) == 0:
        print(f"  {vname}: no data")
        continue
    print(f"  {vname}:  min={vals.min():.2f}  max={vals.max():.2f}  "
          f"mean={vals.mean():.2f}  median={vals.median():.2f}")
    if (vals > 500).any():
        n_high = (vals > 500).sum()
        print(f"    ⚠ {n_high} observations with {vname} > 500 USD")
    if (vals < 0).any():
        n_neg = (vals < 0).sum()
        print(f"    ⚠ {n_neg} observations with {vname} < 0")

# 3. ETS vs tax price comparison
print("\n--- V3: ETS vs tax price comparison ---")
for yr in range(2016, 2025):
    ets_yr = national_price_sample[
        (national_price_sample["year"] == yr) &
        (national_price_sample["instrument_type"] == "ETS")
    ]["price_usd"]
    tax_yr = national_price_sample[
        (national_price_sample["year"] == yr) &
        (national_price_sample["instrument_type"] == "Carbon tax")
    ]["price_usd"]
    if len(ets_yr) > 0 or len(tax_yr) > 0:
        ets_mean = f"{ets_yr.mean():.2f}" if len(ets_yr) > 0 else "n/a"
        tax_mean = f"{tax_yr.mean():.2f}" if len(tax_yr) > 0 else "n/a"
        ets_std = f"{ets_yr.std():.2f}" if len(ets_yr) > 1 else "n/a"
        tax_std = f"{tax_yr.std():.2f}" if len(tax_yr) > 1 else "n/a"
        print(f"  {yr}: ETS mean={ets_mean} (sd={ets_std}, n={len(ets_yr)})  "
              f"Tax mean={tax_mean} (sd={tax_std}, n={len(tax_yr)})")

ets_cv = national_price_sample.loc[
    national_price_sample["instrument_type"] == "ETS"
].groupby("year")["price_usd"].std().mean()
tax_cv = national_price_sample.loc[
    national_price_sample["instrument_type"] == "Carbon tax"
].groupby("year")["price_usd"].std().mean()
if pd.notna(ets_cv) and pd.notna(tax_cv):
    if ets_cv > tax_cv:
        print("  → ETS shows higher year-to-year price dispersion than carbon taxes.")
    else:
        print("  → Carbon taxes show higher year-to-year price dispersion than ETS.")

# 4. Deflator coverage
print("\n--- V4: Deflator coverage ---")
n_defl = national_price_sample["gdp_deflator_linked"].notna().sum()
print(f"  CPI-year rows with deflator (national): {n_defl:,} / {len(national_price_sample):,}")

missing_defl = national_price_sample.loc[
    national_price_sample["gdp_deflator_linked"].isna(),
    ["iso3", "country"]
].drop_duplicates()
if len(missing_defl) > 0:
    print(f"  Countries missing deflator:")
    for _, row in missing_defl.iterrows():
        print(f"    {row['iso3']} — {row['country']}")
else:
    print("  All countries have deflator data.")

# 5. Exchange rate sensitivity diagnostic
print("\n--- V5: Exchange rate sensitivity diagnostic ---")
diag_df = national_price_sample[["cpi_id", "year", "price_usd",
                                  "gdp_deflator_growth_linked"]].copy()
diag_df = diag_df.sort_values(["cpi_id", "year"])
diag_df["price_change"] = diag_df.groupby("cpi_id")["price_usd"].diff()
diag_df = diag_df.dropna(subset=["price_change", "gdp_deflator_growth_linked"])

if len(diag_df) > 5:
    r = diag_df["price_change"].corr(diag_df["gdp_deflator_growth_linked"])
    level = "High" if abs(r) > 0.3 else "Low"
    driver = "is" if abs(r) > 0.3 else "is not"
    print(f"  Correlation between USD price changes and deflator growth: {r:.3f}.")
    print(f"  {level} correlation suggests exchange rate {driver} a major driver "
          f"of USD price variation.")
else:
    print("  Insufficient data for exchange rate sensitivity diagnostic.")

# 6. National vs all CPIs sample comparison
print("\n--- V6: National vs all CPIs sample comparison ---")
print(f"  national_price_sample: {len(national_price_sample):,} rows")
print(f"  all_price_sample:      {len(all_price_sample):,} rows")

national_cpi_ids = set(national_price_sample["cpi_id"].unique())
all_cpi_ids = set(all_price_sample["cpi_id"].unique())
subnational_ids = all_cpi_ids - national_cpi_ids
if subnational_ids:
    print(f"  Subnational CPIs in all but not national ({len(subnational_ids)}):")
    sub_info = all_price_sample[
        all_price_sample["cpi_id"].isin(subnational_ids)
    ][["cpi_id", "country", "scope"]].drop_duplicates()
    for _, row in sub_info.iterrows():
        print(f"    {row['cpi_id']} — {row['country']} ({row['scope']})")
else:
    print("  No subnational CPIs in sample (all have scope == National).")

# ── Done ─────────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print("\n" + "=" * 72)
print(f"Layer D complete. {elapsed:.1f}s elapsed.")
print(f"Output directory: {OUT_DIR}")
print("=" * 72)
