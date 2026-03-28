"""
Layer C — Sectoral Design Analysis (Core Contribution)
=======================================================

World Bank research project: EU CBAM spillovers and carbon pricing.

This is the CORE CONTRIBUTION layer. It tests whether more CBAM-exposed
jurisdictions were more likely to adopt new instruments that cover
industry from inception. This is what Bruegel cannot test.

CRITICAL DATA FACT — EXTENSIVE MARGIN DOMINANCE
-------------------------------------------------
Industry coverage is almost entirely an extensive margin phenomenon.
Of 52 CPIs that first covered industry after 2016:
  - 51 entered the coverage database ALREADY covering industry
  - Only ETS_CN genuinely expanded to cover industry after entry
    (Power only from 2021, added Industry in 2023)

The analysis reflects this. The question is:
  conditional on a new instrument entering, is it more likely to
  be designed to cover industry from inception in jurisdictions
  with greater CBAM trade exposure?
This is not about within-instrument scope expansion.

Design C3 caveat
-----------------
Design C3 is explicitly descriptive and should not be interpreted
causally. The near-absence of within-instrument variation (only ETS_CN
genuinely expanded to cover industry post-entry) means the country-
sector-year variation primarily reflects cohort composition of new
entrants, not within-instrument expansion. Triple-interaction
coefficients reflect cross-sectional patterns across entry cohorts,
not causal within-unit changes.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = os.path.join(ROOT, "data", "output")
DATA_LOOKUP = os.path.join(ROOT, "data", "lookup")
LAYER_C = os.path.join(DATA_OUT, "layer_c")
os.makedirs(LAYER_C, exist_ok=True)

CPI_PATH = os.path.join(DATA_OUT, "cpi_panel.csv")
EXPOSURE_AVG_PATH = os.path.join(DATA_OUT, "cbam_exposure_avg.csv")
EXPOSURE_ANNUAL_PATH = os.path.join(DATA_OUT, "cbam_exposure_annual.csv")
CONTROLS_PATH = os.path.join(DATA_OUT, "controls_panel.csv")
CPI_LIST_PATH = os.path.join(DATA_LOOKUP, "cpi_list.xlsx")

SECTORS = [
    "Power", "Industry", "Road_transport", "Aviation", "Shipping",
    "Buildings", "Waste", "Agriculture", "Forestry",
]
SECTOR_COLS = [f"covers_{s.lower()}" for s in SECTORS]

# Sector column name mapping (cpi_panel uses lowercase)
SECTOR_COL_MAP = {
    "Power": "covers_power",
    "Industry": "covers_industry",
    "Road_transport": "covers_road_transport",
    "Aviation": "covers_aviation",
    "Shipping": "covers_shipping",
    "Buildings": "covers_buildings",
    "Waste": "covers_waste",
    "Agriculture": "covers_agriculture",
    "Forestry": "covers_forestry",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def _fmt_coef(coef, se, pv):
    """Return formatted coefficient string: coef***, (SE)."""
    return f"{coef:.4f}{_stars(pv)}", f"({se:.4f})"


# ============================================================================
# STEP 1 — Load all data
# ============================================================================
t0 = time.time()
print("=" * 72)
print("STEP 1 — Load all data")
print("=" * 72)

cpi = pd.read_csv(CPI_PATH)
print(f"  CPI panel rows loaded:   {len(cpi):,}")

exposure_avg = pd.read_csv(EXPOSURE_AVG_PATH)
print(f"  Countries with avg exposure: {len(exposure_avg):,}")

exposure_annual = pd.read_csv(EXPOSURE_ANNUAL_PATH).rename(columns={"t": "year"})
print(f"  Annual exposure rows:    {len(exposure_annual):,}")

controls = pd.read_csv(CONTROLS_PATH)
print(f"  Controls panel rows:     {len(controls):,}")

master_list = pd.read_excel(CPI_LIST_PATH)
print(f"  Master list CPIs:        {len(master_list):,}")

print(f"  Step 1 done in {time.time() - t0:.1f}s\n")

# ============================================================================
# STEP 2 — Determine entry_year from master list
# ============================================================================
t1 = time.time()
print("=" * 72)
print("STEP 2 — Determine entry_year from master list")
print("=" * 72)

ml = master_list.rename(columns={
    "Unique identifier": "cpi_id",
    "Type": "instrument_type_ml",
    "Scope": "scope_ml",
    "Start date": "start_date",
    "Under development start": "under_dev_start",
    "Country/region": "country_ml",
    "Jurisdiction covered": "jurisdiction_ml",
    "Status": "status_ml",
})

# Entry year: Under development start if available, else Start date
ml["entry_year"] = ml["under_dev_start"].where(ml["under_dev_start"].notna(),
                                                ml["start_date"])
ml["entry_year"] = ml["entry_year"].where(ml["entry_year"].notna(), np.nan)

# Use master-list instrument type (map to match panel conventions)
type_map = {"Carbon tax": "Carbon tax", "ETS": "ETS", "Undefined": "Undefined"}
ml["instrument_type_ml"] = ml["instrument_type_ml"].map(type_map)

print(f"  CPIs with entry_year:    {ml['entry_year'].notna().sum()} / {len(ml)}")
print(f"  entry_year from under_dev_start: {ml['under_dev_start'].notna().sum()}")
print(f"  entry_year from start_date:      "
      f"{(ml['entry_year'].notna() & ml['under_dev_start'].isna()).sum()}")
print(f"  entry_year missing:      {ml['entry_year'].isna().sum()}")

print(f"  Step 2 done in {time.time() - t1:.1f}s\n")

# ============================================================================
# STEP 3 — Build entrant sample
# ============================================================================
t2 = time.time()
print("=" * 72)
print("STEP 3 — Build entrant sample")
print("=" * 72)

# Filter to ETS/Carbon tax with entry_year 2016-2024
entrants = ml[
    (ml["entry_year"] >= 2016) & (ml["entry_year"] <= 2024) &
    (ml["instrument_type_ml"].isin(["ETS", "Carbon tax"]))
].copy()
entrants["entry_year"] = entrants["entry_year"].astype(int)
print(f"  CPIs with entry_year 2016-2024, ETS or Carbon tax: {len(entrants)}")

# Exclude CPIs already active before entry_year
# For each candidate entrant, check if it was status_implemented==1 or
# status_under_development==1 in any year < entry_year in cpi_panel
already_active = []
for _, row in entrants.iterrows():
    cid = row["cpi_id"]
    ey = row["entry_year"]
    prior = cpi[(cpi["cpi_id"] == cid) & (cpi["year"] < ey)]
    if len(prior) > 0:
        was_active = ((prior["status_implemented"] == 1) |
                      (prior["status_under_development"] == 1)).any()
        if was_active:
            already_active.append(cid)

entrants = entrants[~entrants["cpi_id"].isin(already_active)].copy()
print(f"  After excluding already-active before entry_year: {len(entrants)}")
if already_active:
    print(f"  Excluded as already active: {already_active}")

# Get iso3 and scope from cpi_panel (more reliable than master list for merge)
cpi_meta = cpi.drop_duplicates("cpi_id")[["cpi_id", "iso3", "country", "scope",
                                           "instrument_type"]].copy()
entrants = entrants.merge(cpi_meta, on="cpi_id", how="left")

# Include both National and Subnational
entrants = entrants[entrants["scope"].isin(["National", "Subnational"])].copy()
print(f"  After scope filter (National + Subnational): {len(entrants)}")

# ---- Industry at entry ----
# For each entrant, find first year with coverage_active==1 at or after entry_year
industry_records = []
for _, row in entrants.iterrows():
    cid = row["cpi_id"]
    ey = row["entry_year"]

    obs = cpi[(cpi["cpi_id"] == cid) & (cpi["year"] >= ey) &
              (cpi["coverage_active"] == 1)].sort_values("year")

    if len(obs) == 0:
        industry_records.append({
            "cpi_id": cid,
            "industry_at_entry": np.nan,
            "sectors_at_entry": np.nan,
            "n_sectors_at_entry": np.nan,
            "first_coverage_year": np.nan,
            "flag": "no_coverage_data",
        })
    else:
        first = obs.iloc[0]
        industry_records.append({
            "cpi_id": cid,
            "industry_at_entry": int(first["covers_industry"]),
            "sectors_at_entry": first["sector_string"] if pd.notna(first["sector_string"]) else "",
            "n_sectors_at_entry": int(first["n_sectors_covered"]),
            "first_coverage_year": int(first["year"]),
            "flag": "",
        })

ind_df = pd.DataFrame(industry_records)
entrants = entrants.merge(ind_df, on="cpi_id", how="left")

# Gap years
entrants["gap_years"] = np.where(
    entrants["first_coverage_year"].notna(),
    entrants["first_coverage_year"] - entrants["entry_year"],
    np.nan,
)

# ---- Merge exposure ----
entrants = entrants.merge(
    exposure_avg[["iso3", "cbam_exposure_avg"]], on="iso3", how="left"
)
entrants = entrants.merge(
    exposure_annual[["iso3", "year", "cbam_exposure_annual"]].rename(
        columns={"year": "entry_year", "cbam_exposure_annual": "cbam_exposure_entry_year"}
    ),
    on=["iso3", "entry_year"], how="left",
)

# ---- Post indicators ----
entrants["post_2019_entry"] = (entrants["entry_year"] > 2019).astype(int)
entrants["post_2021_entry"] = (entrants["entry_year"] > 2021).astype(int)
entrants["post_2023_entry"] = (entrants["entry_year"] > 2023).astype(int)

# ---- Merge controls at entry_year - 1 ----
entrants["ctrl_year"] = entrants["entry_year"] - 1
ctrl_cols = ["iso3", "year", "log_gdp_pc_constant2015_lag1",
             "wgi_composite_estimate_lag1"]
ctrl_merge = controls[ctrl_cols].rename(columns={"year": "ctrl_year"})
entrants = entrants.merge(ctrl_merge, on=["iso3", "ctrl_year"], how="left")

# ---- Select and order columns ----
ENTRANT_COLS = [
    "cpi_id", "country", "iso3", "entry_year", "scope", "instrument_type",
    "industry_at_entry", "sectors_at_entry", "n_sectors_at_entry",
    "first_coverage_year", "gap_years", "flag",
    "cbam_exposure_avg", "cbam_exposure_entry_year",
    "post_2019_entry", "post_2021_entry", "post_2023_entry",
    "log_gdp_pc_constant2015_lag1", "wgi_composite_estimate_lag1",
]
entrants_final = entrants[[c for c in ENTRANT_COLS if c in entrants.columns]].copy()

print(f"\n  Entrant sample: {len(entrants_final)} instruments")
print(f"  By scope: {entrants_final['scope'].value_counts().to_dict()}")
print(f"  By instrument_type: {entrants_final['instrument_type'].value_counts().to_dict()}")
print(f"  With coverage data: {entrants_final['industry_at_entry'].notna().sum()}")
print(f"  Without coverage data: {entrants_final['industry_at_entry'].isna().sum()}")
print(f"  Step 3 done in {time.time() - t2:.1f}s\n")


# ============================================================================
# VALIDATION 1 — Entrant sample composition
# ============================================================================
print("=" * 72)
print("VALIDATION 1 — Entrant sample composition")
print("=" * 72)

print(f"\n  Total entrants: {len(entrants_final)}")
print(f"  By scope:")
for s, n in entrants_final["scope"].value_counts().items():
    print(f"    {s}: {n}")
print(f"  By instrument_type:")
for t, n in entrants_final["instrument_type"].value_counts().items():
    print(f"    {t}: {n}")
print(f"\n  By entry_year cohort:")
for yr in range(2016, 2025):
    sub = entrants_final[entrants_final["entry_year"] == yr]
    if len(sub) > 0:
        ind_share = sub["industry_at_entry"].mean() if sub["industry_at_entry"].notna().any() else "N/A"
        if isinstance(ind_share, float):
            ind_share = f"{ind_share:.2f}"
        print(f"    {yr}: {len(sub)} entrants, industry share = {ind_share}")
    else:
        print(f"    {yr}: 0 entrants")

print(f"\n  N with coverage data (industry_at_entry not NaN): "
      f"{entrants_final['industry_at_entry'].notna().sum()}")
no_cov = entrants_final[entrants_final["industry_at_entry"].isna()]
print(f"  N without coverage data: {len(no_cov)}")
if len(no_cov) > 0:
    print(f"  CPIs without coverage data:")
    for _, r in no_cov.iterrows():
        print(f"    {r['cpi_id']} ({r['country']}, entry_year={r['entry_year']})")

print(f"\n  Gap years distribution (entry_year to first_coverage_year):")
gaps = entrants_final["gap_years"].dropna()
if len(gaps) > 0:
    print(f"    N: {len(gaps)}")
    print(f"    Mean: {gaps.mean():.1f}")
    print(f"    Median: {gaps.median():.1f}")
    print(f"    Min: {gaps.min():.0f}, Max: {gaps.max():.0f}")
    for g in sorted(gaps.unique()):
        print(f"    gap={int(g)}: {(gaps == g).sum()} CPIs")

# ============================================================================
# VALIDATION 2 — ETS_CN classification check
# ============================================================================
print("\n" + "=" * 72)
print("VALIDATION 2 — ETS_CN classification check")
print("=" * 72)

ets_cn = entrants_final[entrants_final["cpi_id"] == "ETS_CN"]
if len(ets_cn) == 1:
    r = ets_cn.iloc[0]
    print(f"  ETS_CN check: entry_year={r['entry_year']}, "
          f"first_coverage_year={r['first_coverage_year']}, "
          f"industry_at_entry={r['industry_at_entry']}, "
          f"sectors_at_entry={r['sectors_at_entry']}")
    if r["industry_at_entry"] == 1:
        print("  *** ERROR: ETS_CN should have industry_at_entry == 0 "
              "(Power only at entry in 2021). Fix logic! ***")
    elif r["industry_at_entry"] == 0:
        print("  CORRECT: ETS_CN entered with Power only — industry_at_entry == 0.")
    else:
        print(f"  ETS_CN has industry_at_entry = {r['industry_at_entry']} (NaN = no coverage data)")
elif len(ets_cn) == 0:
    print("  ETS_CN not in entrant sample (may not meet entry criteria).")
else:
    print(f"  WARNING: Multiple ETS_CN rows: {len(ets_cn)}")

# ============================================================================
# VALIDATION 3 — Industry at entry spot-check
# ============================================================================
print("\n" + "=" * 72)
print("VALIDATION 3 — Industry at entry spot-check (10 random entrants)")
print("=" * 72)

spot = entrants_final[entrants_final["industry_at_entry"].notna()]
if len(spot) > 10:
    spot = spot.sample(10, random_state=42)
for _, r in spot.iterrows():
    print(f"  {r['cpi_id']:25s} entry_year={r['entry_year']}  "
          f"first_cov_year={r['first_coverage_year']:.0f}  "
          f"industry={r['industry_at_entry']:.0f}  "
          f"sectors={r['sectors_at_entry']}")

# ============================================================================
# VALIDATION 4 — Within-instrument sector variation
# ============================================================================
print("\n" + "=" * 72)
print("VALIDATION 4 — Within-instrument sector coverage variation")
print("=" * 72)

cpi_active = cpi[cpi["coverage_active"] == 1].sort_values(["cpi_id", "year"])
n_changers = 0
total_cpis = cpi_active["cpi_id"].nunique()
for cid, grp in cpi_active.groupby("cpi_id"):
    if grp["n_sectors_covered"].diff().abs().sum() > 0:
        n_changers += 1

print(f"  CPIs with any within-instrument sector coverage change: "
      f"{n_changers} out of {total_cpis}.")
variation_desc = "minimal" if n_changers <= 5 else "substantial"
print(f"  This confirms that within-instrument variation is {variation_desc}.")

# ============================================================================
# STEP 4 — Design C1b: CPI-level Panel DiD (PRIMARY SPECIFICATION)
# ============================================================================
t3 = time.time()
print("\n" + "=" * 72)
print("STEP 4 — Design C1b: CPI-level Panel DiD (PRIMARY SPECIFICATION)")
print("=" * 72)

# Build panel: all CPIs where instrument_type in (ETS, Carbon tax),
# is_eea_ets_extension == 0, cbam_exposure_avg not null.
# Keep ALL years 2016-2026 including pre-entry years (coverage_active == 0).
panel_c1b = cpi[
    (cpi["instrument_type"].isin(["ETS", "Carbon tax"])) &
    (cpi["is_eea_ets_extension"] == 0) &
    (cpi["year"] >= 2016) & (cpi["year"] <= 2026)
].copy()

# Merge exposure (use avg from exposure file; drop any existing column first)
if "cbam_exposure_avg" in panel_c1b.columns:
    panel_c1b = panel_c1b.drop(columns=["cbam_exposure_avg"])
panel_c1b = panel_c1b.merge(
    exposure_avg[["iso3", "cbam_exposure_avg"]], on="iso3", how="left"
)
panel_c1b = panel_c1b[panel_c1b["cbam_exposure_avg"].notna()].copy()

# Merge controls (lagged)
ctrl_cols_c1b = ["iso3", "year", "log_gdp_pc_constant2015_lag1",
                 "wgi_composite_estimate_lag1"]
panel_c1b = panel_c1b.merge(controls[ctrl_cols_c1b], on=["iso3", "year"], how="left")

print(f"  Panel C1b: {len(panel_c1b):,} CPI-year rows")
print(f"  CPIs: {panel_c1b['cpi_id'].nunique()}, Countries: {panel_c1b['iso3'].nunique()}")
print(f"  Years: {panel_c1b['year'].min()}-{panel_c1b['year'].max()}")
print(f"  Mean covers_industry: {panel_c1b['covers_industry'].mean():.4f}")

# Post indicators and interactions
for cutoff, label in [(2019, "post_2019"), (2021, "post_2021"), (2023, "post_2023")]:
    panel_c1b[label] = (panel_c1b["year"] >= cutoff).astype(int)
    panel_c1b[f"exposure_x_{label}"] = (
        panel_c1b["cbam_exposure_avg"] * panel_c1b[label]
    )

# Set panel index
panel_c1b_idx = panel_c1b.set_index(["cpi_id", "year"])

CONTROL_VARS = ["log_gdp_pc_constant2015_lag1", "wgi_composite_estimate_lag1"]

# --- Panel DiD regressions ---
c1b_results = {}
INTERACTION_MAP_C1B = {
    "Post-2019": "exposure_x_post_2019",
    "Post-2021": "exposure_x_post_2021",
    "Post-2023": "exposure_x_post_2023",
}

for post_label, interaction in INTERACTION_MAP_C1B.items():
    exog_vars = [interaction] + CONTROL_VARS
    reg_cols = ["covers_industry"] + exog_vars + ["iso3"]
    reg_df = panel_c1b_idx[reg_cols].dropna(subset=["covers_industry"] + exog_vars)

    dep = reg_df["covers_industry"]
    exog = reg_df[exog_vars]
    cluster_df = pd.DataFrame({"cluster": reg_df["iso3"].values}, index=reg_df.index)

    try:
        mod = PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                       drop_absorbed=True)
        res = mod.fit(cov_type="clustered", clusters=cluster_df)
        c1b_results[post_label] = res

        coef = res.params[interaction]
        se = res.std_errors[interaction]
        pv = res.pvalues[interaction]
        n = int(res.nobs)
        n_cpi = reg_df.index.get_level_values("cpi_id").nunique()
        print(f"  {post_label}: beta(exposure × post)={coef:+.5f}{_stars(pv)}, "
              f"SE={se:.5f}, N={n}, CPIs={n_cpi}")
    except Exception as e:
        print(f"  {post_label} FAILED: {e}")

# --- Event study ---
print("\n  --- Event study (relative to 2019) ---")

# Year-specific indicators relative to 2019, each interacted with exposure
event_years = {
    "year_minus2": 2017, "year_minus1": 2018, "year_0": 2019,
    "year_plus1": 2020, "year_plus2": 2021, "year_plus3": 2022,
    "year_plus4": 2023,
}
# Reference year: 2016 (omitted)
for col_name, yr in event_years.items():
    panel_c1b[col_name] = (panel_c1b["year"] == yr).astype(int)
    panel_c1b[f"exposure_x_{col_name}"] = (
        panel_c1b["cbam_exposure_avg"] * panel_c1b[col_name]
    )

event_interactions = [f"exposure_x_{c}" for c in event_years]
panel_c1b_es = panel_c1b.set_index(["cpi_id", "year"])
exog_es_vars = event_interactions + CONTROL_VARS
reg_es_cols = ["covers_industry"] + exog_es_vars + ["iso3"]
reg_es = panel_c1b_es[reg_es_cols].dropna(subset=["covers_industry"] + exog_es_vars)

dep_es = reg_es["covers_industry"]
exog_es = reg_es[exog_es_vars]
cluster_es = pd.DataFrame({"cluster": reg_es["iso3"].values}, index=reg_es.index)

event_study_result = None
try:
    mod_es = PanelOLS(dep_es, exog_es, entity_effects=True, time_effects=True,
                      drop_absorbed=True)
    res_es = mod_es.fit(cov_type="clustered", clusters=cluster_es)
    event_study_result = res_es

    # Print event study coefficients table
    print(f"\n  {'Year':<12} {'Rel.Year':<10} {'Coefficient':>12} {'SE':>10} "
          f"{'95% CI low':>12} {'95% CI high':>12} {'p-value':>10}")
    print("  " + "-" * 78)
    es_table_rows = []
    for col_name, yr in event_years.items():
        ivar = f"exposure_x_{col_name}"
        coef = res_es.params[ivar]
        se = res_es.std_errors[ivar]
        pv = res_es.pvalues[ivar]
        ci_lo = coef - 1.96 * se
        ci_hi = coef + 1.96 * se
        rel_yr = yr - 2019
        rel_label = f"{rel_yr:+d}" if rel_yr != 0 else "0"
        print(f"  {yr:<12} {rel_label:<10} {coef:>+12.5f}{_stars(pv):<3} "
              f"{se:>10.5f} {ci_lo:>12.5f} {ci_hi:>12.5f} {pv:>10.4f}")
        es_table_rows.append({
            "year": yr, "relative_year": rel_yr, "coefficient": coef,
            "se": se, "ci_low": ci_lo, "ci_high": ci_hi, "p_value": pv,
            "stars": _stars(pv),
        })
    print(f"\n  N={int(res_es.nobs)}, CPIs={reg_es.index.get_level_values('cpi_id').nunique()}")

    # Plot event study
    es_df_plot = pd.DataFrame(es_table_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(es_df_plot["relative_year"], es_df_plot["coefficient"],
                yerr=1.96 * es_df_plot["se"], fmt="o-", color="steelblue",
                capsize=4, linewidth=1.5, markersize=6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(-0.5, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("Year relative to 2019", fontsize=11)
    ax.set_ylabel("Coefficient on exposure × year indicator", fontsize=11)
    ax.set_title("Event Study: CBAM Exposure × Year → Industry Coverage\n"
                 "(CPI-level panel DiD, CPI + year FE, clustered by country)",
                 fontsize=11)
    ax.set_xticks(es_df_plot["relative_year"])
    ax.set_xticklabels([f"{yr}\n({r:+d})" for yr, r
                        in zip(es_df_plot["year"], es_df_plot["relative_year"])],
                       fontsize=9)
    fig.tight_layout()
    es_plot_path = os.path.join(LAYER_C, "event_study_industry.png")
    fig.savefig(es_plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved event study plot: {es_plot_path}")

except Exception as e:
    print(f"  Event study FAILED: {e}")

# --- Table 2b: Panel DiD results ---
t2b_rows = []
for post_label, interaction in INTERACTION_MAP_C1B.items():
    res = c1b_results.get(post_label)
    if res is not None:
        coef_str, se_str = _fmt_coef(res.params[interaction],
                                      res.std_errors[interaction],
                                      res.pvalues[interaction])
        n = str(int(res.nobs))
    else:
        coef_str = se_str = n = "—"
    t2b_rows.append({
        "Specification": post_label,
        "Coefficient (exposure × post)": coef_str,
        "SE": se_str,
        "N": n,
        "CPI FE": "Yes",
        "Year FE": "Yes",
        "Controls": "Yes",
    })

t2b_df = pd.DataFrame(t2b_rows)
title_2b = pd.DataFrame(
    [["Table 2b (Primary): CPI-Level Panel DiD — Industry Coverage"] +
     [""] * (len(t2b_df.columns) - 1)],
    columns=t2b_df.columns,
)
note_2b = (
    "The panel DiD identifies from CPI entry events — the transition "
    "from coverage_active=0 to coverage_active=1. Within-active variation "
    "is minimal (only ETS_CN genuinely expands industry coverage post-entry). "
    "The coefficient β captures whether CPIs entering post-CBAM in high-"
    "exposure countries are more likely to cover industry at the time of "
    "their first active observation. This is consistent with the extensive "
    "margin interpretation established in the descriptive analysis."
)
footer_2b = pd.DataFrame(
    [[f"CPI + year FE. SEs clustered at country level. "
      f"Controls: log_gdp_pc_constant2015_lag1, wgi_composite_estimate_lag1. "
      f"*** p<0.01  ** p<0.05  * p<0.1. NOTE: {note_2b}"] +
     [""] * (len(t2b_df.columns) - 1)],
    columns=t2b_df.columns,
)
t2b_out = pd.concat([title_2b, t2b_df.astype(str), footer_2b], ignore_index=True)
t2b_path = os.path.join(LAYER_C, "table2b_c1b_panel_did.csv")
t2b_out.to_csv(t2b_path, index=False)
print(f"  Saved: {t2b_path}")

print(f"\n  NOTE: {note_2b}")
print(f"\n  Step 4 done in {time.time() - t3:.1f}s\n")


# ============================================================================
# STEP 5 — Design C1: Entrant-level regressions (SUPPLEMENTARY)
# ============================================================================
t3b = time.time()
print("\n" + "=" * 72)
print("STEP 5 — Design C1: Entrant-level regressions (SUPPLEMENTARY)")
print("=" * 72)

CONTROL_VARS = ["log_gdp_pc_constant2015_lag1", "wgi_composite_estimate_lag1"]

# Regression sample: non-null industry_at_entry
reg_sample = entrants_final[entrants_final["industry_at_entry"].notna()].copy()
print(f"\n  Regression sample: {len(reg_sample)} entrants with coverage data")

c1_results = {}

# --- Model C: Exposure main effect only (no post interaction) ---
print("\n  --- Model C: Exposure main effect only ---")
dep = reg_sample["industry_at_entry"]
exog_vars_c = ["cbam_exposure_avg"] + CONTROL_VARS
exog_c = reg_sample[exog_vars_c].dropna()
dep_c = dep.loc[exog_c.index]
exog_c = sm.add_constant(exog_c)

try:
    groups_c = reg_sample.loc[exog_c.index, "iso3"]
    mod_c = sm.OLS(dep_c, exog_c).fit(cov_type="cluster", cov_kwds={"groups": groups_c})
    c1_results[("Model_C", "OLS", "none")] = mod_c
    coef = mod_c.params["cbam_exposure_avg"]
    se = mod_c.bse["cbam_exposure_avg"]
    pv = mod_c.pvalues["cbam_exposure_avg"]
    n = int(mod_c.nobs)
    print(f"    OLS: beta(exposure)={coef:+.4f}{_stars(pv)}, SE={se:.4f}, N={n}")
    if n < 30:
        print(f"    WARNING: Small entrant sample (N={n}) for Model C. "
              f"Interpret coefficients as indicative only.")
except Exception as e:
    print(f"    Model C OLS FAILED: {e}")

# --- Model A (OLS) and Model B (Logit) for each post cutoff ---
for post_label, post_col in [("Post-2019", "post_2019_entry"),
                              ("Post-2021", "post_2021_entry"),
                              ("Post-2023", "post_2023_entry")]:
    print(f"\n  --- {post_label} ---")

    interaction_col = f"exposure_x_{post_col}"
    reg_sample[interaction_col] = (reg_sample["cbam_exposure_avg"] *
                                   reg_sample[post_col])

    exog_vars_a = ["cbam_exposure_avg", post_col, interaction_col] + CONTROL_VARS

    # Build clean regression data
    reg_cols = ["industry_at_entry"] + exog_vars_a
    reg_data = reg_sample[reg_cols].dropna()
    dep_a = reg_data["industry_at_entry"]
    exog_a = reg_data[exog_vars_a]
    exog_a_const = sm.add_constant(exog_a)

    n = len(reg_data)
    if n < 30:
        print(f"    WARNING: Small entrant sample (N={n}) for {post_label}. "
              f"Interpret coefficients as indicative only.")

    # Model A — OLS (LPM)
    try:
        groups_a = reg_sample.loc[reg_data.index, "iso3"]
        mod_a = sm.OLS(dep_a, exog_a_const).fit(cov_type="cluster", cov_kwds={"groups": groups_a})
        c1_results[("Model_A", "OLS", post_label)] = mod_a
        for var in ["cbam_exposure_avg", interaction_col]:
            coef = mod_a.params[var]
            se = mod_a.bse[var]
            pv = mod_a.pvalues[var]
            print(f"    OLS: {var:40s} beta={coef:+.4f}{_stars(pv)}, SE={se:.4f}, N={n}")
    except Exception as e:
        print(f"    Model A OLS FAILED: {e}")

    # Model B — Logit with marginal effects
    try:
        # Check if outcome has variation
        if dep_a.nunique() < 2:
            print(f"    Logit SKIPPED: no variation in outcome (all {dep_a.iloc[0]:.0f})")
            continue

        # NOTE: Logit MEs use non-clustered SEs — interpret as descriptive only.
        # OLS with country-clustered SEs is the primary specification.
        mod_b = Logit(dep_a, exog_a_const).fit(disp=0, maxiter=100)
        mfx = mod_b.get_margeff(at="mean")
        c1_results[("Model_B", "Logit", post_label)] = {
            "model": mod_b, "margeff": mfx, "n": n
        }
        for var in ["cbam_exposure_avg", interaction_col]:
            idx = list(exog_a.columns).index(var)
            me = mfx.margeff[idx]
            me_se = mfx.margeff_se[idx]
            me_pv = mfx.pvalues[idx]
            print(f"    Logit ME: {var:40s} ME={me:+.4f}{_stars(me_pv)}, SE={me_se:.4f}, N={n}")
    except Exception as e:
        print(f"    Model B Logit FAILED: {e}")

print(f"\n  Step 5 done in {time.time() - t3b:.1f}s\n")

# ============================================================================
# STEP 6 — Design C2: Country-year outcome panel
# ============================================================================
t4 = time.time()
print("=" * 72)
print("STEP 6 — Design C2: Country-year outcome panel")
print("=" * 72)

# Build country-year entrant counts from entrant data (national only)
nat_entrants = entrants_final[entrants_final["scope"] == "National"].copy()

cy_entrants = nat_entrants.groupby(["iso3", "entry_year"]).agg(
    n_new_entrants=("cpi_id", "count"),
    n_new_industry_entrants=("industry_at_entry", lambda x: (x == 1).sum()),
).reset_index().rename(columns={"entry_year": "year"})

cy_entrants["any_new_industry_entrant"] = (
    cy_entrants["n_new_industry_entrants"] > 0
).astype(int)
cy_entrants["industry_share_new_entrants"] = np.where(
    cy_entrants["n_new_entrants"] > 0,
    cy_entrants["n_new_industry_entrants"] / cy_entrants["n_new_entrants"],
    np.nan,
)

# Build full country-year spine
years = list(range(2016, 2025))
spine = exposure_avg[["iso3", "cbam_exposure_avg"]].assign(key=1).merge(
    pd.DataFrame({"year": years, "key": 1}), on="key"
).drop(columns="key")

cy = spine.merge(cy_entrants, on=["iso3", "year"], how="left")
for col in ["n_new_entrants", "n_new_industry_entrants", "any_new_industry_entrant"]:
    cy[col] = cy[col].fillna(0).astype(int)

# Post indicators and interactions
cy["post_2019"] = (cy["year"] > 2019).astype(int)
cy["post_2021"] = (cy["year"] > 2021).astype(int)
cy["post_2023"] = (cy["year"] > 2023).astype(int)
cy["exposure_x_post2019"] = cy["cbam_exposure_avg"] * cy["post_2019"]
cy["exposure_x_post2021"] = cy["cbam_exposure_avg"] * cy["post_2021"]
cy["exposure_x_post2023"] = cy["cbam_exposure_avg"] * cy["post_2023"]

# Merge controls
ctrl_cols_merge = ["iso3", "year", "log_gdp_pc_constant2015_lag1",
                   "wgi_composite_estimate_lag1"]
cy = cy.merge(controls[ctrl_cols_merge], on=["iso3", "year"], how="left")

# Drop null exposure
cy = cy[cy["cbam_exposure_avg"].notna()].copy()

print(f"  Country-year panel: {len(cy):,} rows, {cy['iso3'].nunique()} countries")
print(f"  any_new_industry_entrant mean: {cy['any_new_industry_entrant'].mean():.4f}")

# TWFE regressions
INTERACTION_MAP_C2 = {
    "Post-2019": "exposure_x_post2019",
    "Post-2021": "exposure_x_post2021",
    "Post-2023": "exposure_x_post2023",
}

panel_df = cy.set_index(["iso3", "year"])
c2_results = {}

for post_label, interaction in INTERACTION_MAP_C2.items():
    exog_vars = [interaction] + CONTROL_VARS
    reg_cols = ["any_new_industry_entrant"] + exog_vars
    reg_df = panel_df[reg_cols].dropna()

    dep = reg_df["any_new_industry_entrant"]
    exog = reg_df[exog_vars]

    try:
        mod = PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                       drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        c2_results[post_label] = res

        coef = res.params[interaction]
        se = res.std_errors[interaction]
        pv = res.pvalues[interaction]
        n = int(res.nobs)
        print(f"  {post_label}: beta(interaction)={coef:+.5f}{_stars(pv)}, "
              f"SE={se:.5f}, N={n}")
    except Exception as e:
        print(f"  {post_label} FAILED: {e}")

print(f"  Step 6 done in {time.time() - t4:.1f}s\n")

# ============================================================================
# STEP 7 — Design C3: Country-sector-year (descriptive)
# ============================================================================
t5 = time.time()
print("=" * 72)
print("STEP 7 — Design C3: Country-sector-year (DESCRIPTIVE ONLY)")
print("=" * 72)
print("  CAVEAT: Design C3 is explicitly descriptive and should not be")
print("  interpreted causally. The near-absence of within-instrument")
print("  variation (only ETS_CN genuinely expanded to cover industry")
print("  post-entry) means the country-sector-year variation primarily")
print("  reflects cohort composition of new entrants, not within-instrument")
print("  expansion. Triple-interaction coefficients reflect cross-sectional")
print("  patterns across entry cohorts, not causal within-unit changes.\n")

# Build country-sector-year panel from national CPIs
cpi_national = cpi[
    (cpi["scope"] == "National") &
    (cpi["is_eea_ets_extension"] != 1) &
    (cpi["instrument_type"] != "Undefined") &
    (cpi["year"] >= 2016) & (cpi["year"] <= 2024)
].copy()

# For each country-year, sum sector coverage across CPIs
csy_rows = []
for (iso3, yr), grp in cpi_national.groupby(["iso3", "year"]):
    for sector in SECTORS:
        col = SECTOR_COL_MAP[sector]
        n_covering = int(grp[col].sum())
        csy_rows.append({
            "iso3": iso3,
            "sector": sector,
            "year": yr,
            "n_cpis_covering_sector": n_covering,
        })

csy = pd.DataFrame(csy_rows)

# Also create rows for countries in exposure universe but no CPIs
all_isos = set(exposure_avg["iso3"])
cpi_isos = set(csy["iso3"])
missing_isos = all_isos - cpi_isos

missing_rows = []
for iso3 in missing_isos:
    for yr in years:
        for sector in SECTORS:
            missing_rows.append({
                "iso3": iso3, "sector": sector, "year": yr,
                "n_cpis_covering_sector": 0,
            })
if missing_rows:
    csy = pd.concat([csy, pd.DataFrame(missing_rows)], ignore_index=True)

# Merge exposure
csy = csy.merge(exposure_avg[["iso3", "cbam_exposure_avg"]], on="iso3", how="left")
csy = csy[csy["cbam_exposure_avg"].notna()].copy()

# Indicators
csy["cbam_relevant_sector"] = (csy["sector"] == "Industry").astype(int)
csy["post_2021"] = (csy["year"] > 2021).astype(int)
csy["exposure_x_post2021"] = csy["cbam_exposure_avg"] * csy["post_2021"]
csy["triple_interaction"] = (csy["cbam_exposure_avg"] * csy["post_2021"] *
                             csy["cbam_relevant_sector"])

# Merge controls
csy = csy.merge(controls[ctrl_cols_merge], on=["iso3", "year"], how="left")

# Entity = iso3 × sector
csy["entity"] = csy["iso3"] + "_" + csy["sector"]

print(f"  Country-sector-year panel: {len(csy):,} rows")
print(f"  Countries: {csy['iso3'].nunique()}, Sectors: {csy['sector'].nunique()}")

# TWFE regression
panel_csy = csy.set_index(["entity", "year"])
exog_vars_c3 = ["exposure_x_post2021", "triple_interaction"] + CONTROL_VARS
reg_cols_c3 = ["n_cpis_covering_sector"] + exog_vars_c3
reg_csy = panel_csy[reg_cols_c3].dropna()

c3_result = None
try:
    dep_c3 = reg_csy["n_cpis_covering_sector"]
    exog_c3 = reg_csy[exog_vars_c3]

    mod_c3 = PanelOLS(dep_c3, exog_c3, entity_effects=True, time_effects=True,
                      drop_absorbed=True)
    # Cluster at country level: extract iso3 from entity (iso3_Sector)
    # Sectors may contain underscores (Road_transport), so strip known suffixes
    sector_suffixes = [f"_{s}" for s in SECTORS]
    def _extract_iso3(entity):
        for sfx in sector_suffixes:
            if entity.endswith(sfx):
                return entity[:-len(sfx)]
        return entity
    cluster_ids = reg_csy.index.get_level_values("entity").map(_extract_iso3)
    cluster_df = pd.DataFrame({"cluster": cluster_ids.values},
                              index=reg_csy.index)
    res_c3 = mod_c3.fit(cov_type="clustered", clusters=cluster_df)
    c3_result = res_c3

    for var in ["exposure_x_post2021", "triple_interaction"]:
        coef = res_c3.params[var]
        se = res_c3.std_errors[var]
        pv = res_c3.pvalues[var]
        n = int(res_c3.nobs)
        print(f"  {var}: beta={coef:+.5f}{_stars(pv)}, SE={se:.5f}, N={n}")
except Exception as e:
    print(f"  Design C3 regression FAILED: {e}")

print(f"\n  CAVEAT (repeated): Descriptive only — see methodology notes.")
print(f"  Step 7 done in {time.time() - t5:.1f}s\n")

# ============================================================================
# OUTPUT TABLES
# ============================================================================
t6 = time.time()
print("=" * 72)
print("OUTPUT TABLES")
print("=" * 72)

# --- TABLE 1: Entrant descriptive by cohort year ---
t1_rows = []
for yr in range(2016, 2025):
    sub = entrants_final[entrants_final["entry_year"] == yr]
    n = len(sub)
    if n == 0:
        t1_rows.append({
            "entry_year": yr, "n_entrants": 0, "n_national": 0, "n_subnational": 0,
            "n_with_coverage_data": 0, "share_covering_industry": np.nan,
            "mean_cbam_exposure_avg": np.nan, "mean_n_sectors_at_entry": np.nan,
        })
        continue
    t1_rows.append({
        "entry_year": yr,
        "n_entrants": n,
        "n_national": (sub["scope"] == "National").sum(),
        "n_subnational": (sub["scope"] == "Subnational").sum(),
        "n_with_coverage_data": sub["industry_at_entry"].notna().sum(),
        "share_covering_industry": (
            sub.loc[sub["industry_at_entry"].notna(), "industry_at_entry"].mean()
            if sub["industry_at_entry"].notna().any() else np.nan
        ),
        "mean_cbam_exposure_avg": sub["cbam_exposure_avg"].mean(),
        "mean_n_sectors_at_entry": (
            sub.loc[sub["n_sectors_at_entry"].notna(), "n_sectors_at_entry"].mean()
            if sub["n_sectors_at_entry"].notna().any() else np.nan
        ),
    })

t1_df = pd.DataFrame(t1_rows)
title_row = pd.DataFrame(
    [["Table 1: New CPI Entrant Characteristics by Entry Cohort"] +
     [""] * (len(t1_df.columns) - 1)],
    columns=t1_df.columns,
)
t1_out = pd.concat([title_row, t1_df.astype(str)], ignore_index=True)
t1_path = os.path.join(LAYER_C, "table1_entrant_descriptive.csv")
t1_out.to_csv(t1_path, index=False)
print(f"  Saved: {t1_path}")

# --- TABLE 2: Design C1 regression results ---
t2_rows = []

# Model C row
res_mc = c1_results.get(("Model_C", "OLS", "none"))
if res_mc is not None:
    coef_c, se_c = _fmt_coef(res_mc.params["cbam_exposure_avg"],
                              res_mc.bse["cbam_exposure_avg"],
                              res_mc.pvalues["cbam_exposure_avg"])
    t2_rows.append({
        "Model": "OLS no interaction (Model C)",
        "Variable": "cbam_exposure_avg",
        "Post-2019": coef_c, "Post-2019 SE": se_c,
        "Post-2021": coef_c, "Post-2021 SE": se_c,
        "Post-2023": coef_c, "Post-2023 SE": se_c,
        "N": str(int(res_mc.nobs)),
        "Note": "Same specification across columns (no post indicator)",
    })

# Model A rows
for post_label, post_col in [("Post-2019", "post_2019_entry"),
                              ("Post-2021", "post_2021_entry"),
                              ("Post-2023", "post_2023_entry")]:
    interaction_col = f"exposure_x_{post_col}"
    res_a = c1_results.get(("Model_A", "OLS", post_label))
    if res_a is not None:
        coef_exp, se_exp = _fmt_coef(res_a.params["cbam_exposure_avg"],
                                      res_a.bse["cbam_exposure_avg"],
                                      res_a.pvalues["cbam_exposure_avg"])
        coef_int, se_int = _fmt_coef(res_a.params[interaction_col],
                                      res_a.bse[interaction_col],
                                      res_a.pvalues[interaction_col])
        n_a = str(int(res_a.nobs))
    else:
        coef_exp = se_exp = coef_int = se_int = n_a = "—"

    t2_rows.append({
        "Model": f"OLS (Model A) — {post_label}",
        "Variable": "cbam_exposure_avg",
        post_label: coef_exp, f"{post_label} SE": se_exp,
        "N": n_a,
    })
    t2_rows.append({
        "Model": f"OLS (Model A) — {post_label}",
        "Variable": f"exposure × {post_label.lower()}",
        post_label: coef_int, f"{post_label} SE": se_int,
        "N": "",
    })

# Model B rows (Logit marginal effects)
for post_label, post_col in [("Post-2019", "post_2019_entry"),
                              ("Post-2021", "post_2021_entry"),
                              ("Post-2023", "post_2023_entry")]:
    interaction_col = f"exposure_x_{post_col}"
    res_b = c1_results.get(("Model_B", "Logit", post_label))
    if res_b is not None:
        mfx = res_b["margeff"]
        exog_names = list(reg_sample[
            ["cbam_exposure_avg", post_col, interaction_col] + CONTROL_VARS
        ].columns)
        idx_exp = exog_names.index("cbam_exposure_avg")
        idx_int = exog_names.index(interaction_col)

        coef_exp, se_exp = _fmt_coef(mfx.margeff[idx_exp],
                                      mfx.margeff_se[idx_exp],
                                      mfx.pvalues[idx_exp])
        coef_int, se_int = _fmt_coef(mfx.margeff[idx_int],
                                      mfx.margeff_se[idx_int],
                                      mfx.pvalues[idx_int])
        n_b = str(res_b["n"])
    else:
        coef_exp = se_exp = coef_int = se_int = n_b = "—"

    t2_rows.append({
        "Model": f"Logit ME (Model B) — {post_label}",
        "Variable": "cbam_exposure_avg",
        post_label: coef_exp, f"{post_label} SE": se_exp,
        "N": n_b,
    })
    t2_rows.append({
        "Model": f"Logit ME (Model B) — {post_label}",
        "Variable": f"exposure × {post_label.lower()}",
        post_label: coef_int, f"{post_label} SE": se_int,
        "N": "",
    })

t2_df = pd.DataFrame(t2_rows)
title2 = pd.DataFrame(
    [["Table 2 (Supplementary): Entrant-Level Industry Coverage Regressions"] +
     [""] * (len(t2_df.columns) - 1)],
    columns=t2_df.columns,
)
footer2 = pd.DataFrame(
    [["Controls: log_gdp_pc_constant2015_lag1, wgi_composite_estimate_lag1. "
      "OLS uses country-clustered SEs. Logit MEs use non-clustered SEs (descriptive only). *** p<0.01  ** p<0.05  * p<0.1"] +
     [""] * (len(t2_df.columns) - 1)],
    columns=t2_df.columns,
)
t2_out = pd.concat([title2, t2_df.astype(str), footer2], ignore_index=True)
t2_path = os.path.join(LAYER_C, "table2_c1_entrant_regressions.csv")
t2_out.to_csv(t2_path, index=False)
print(f"  Saved: {t2_path}")

# --- TABLE 3: Design C2 country-year results ---
t3_rows = []
for post_label, interaction in INTERACTION_MAP_C2.items():
    res = c2_results.get(post_label)
    if res is not None:
        coef_str, se_str = _fmt_coef(res.params[interaction],
                                      res.std_errors[interaction],
                                      res.pvalues[interaction])
        n = str(int(res.nobs))
        r2 = f"{res.rsquared_overall:.4f}"
    else:
        coef_str = se_str = n = r2 = "—"
    t3_rows.append({
        "Specification": f"any_new_industry_entrant × {post_label}",
        "Coefficient": coef_str,
        "SE": se_str,
        "N": n,
        "R2_overall": r2,
    })

t3_df = pd.DataFrame(t3_rows)
title3 = pd.DataFrame(
    [["Table 3 (Secondary): Country-Year Industry Entrant Outcomes"] +
     [""] * (len(t3_df.columns) - 1)],
    columns=t3_df.columns,
)
footer3 = pd.DataFrame(
    [["TWFE with country + year FE. SEs clustered at country level. "
      "Controls: log_gdp_pc_constant2015_lag1, wgi_composite_estimate_lag1. "
      "*** p<0.01  ** p<0.05  * p<0.1"] +
     [""] * (len(t3_df.columns) - 1)],
    columns=t3_df.columns,
)
t3_out = pd.concat([title3, t3_df.astype(str), footer3], ignore_index=True)
t3_path = os.path.join(LAYER_C, "table3_c2_countryyear.csv")
t3_out.to_csv(t3_path, index=False)
print(f"  Saved: {t3_path}")

# --- TABLE 4: Design C3 triple interaction results ---
t4_rows = []
if c3_result is not None:
    for var in ["exposure_x_post2021", "triple_interaction"]:
        coef_str, se_str = _fmt_coef(c3_result.params[var],
                                      c3_result.std_errors[var],
                                      c3_result.pvalues[var])
        t4_rows.append({
            "Variable": var,
            "Coefficient": coef_str,
            "SE": se_str,
            "N": str(int(c3_result.nobs)),
            "R2_overall": f"{c3_result.rsquared_overall:.4f}",
        })
else:
    t4_rows.append({"Variable": "Regression failed", "Coefficient": "—",
                     "SE": "—", "N": "—", "R2_overall": "—"})

t4_df = pd.DataFrame(t4_rows)
caveat = ("NOTE: Descriptive — see methodology notes. Design C3 is explicitly "
          "descriptive and should not be interpreted causally.")
title4 = pd.DataFrame(
    [[f"Table 4 (Supplementary/Descriptive): Sector-Specific Patterns. {caveat}"] +
     [""] * (len(t4_df.columns) - 1)],
    columns=t4_df.columns,
)
footer4 = pd.DataFrame(
    [["Country-sector FE + year FE. SEs clustered at country level. "
      "Controls: log_gdp_pc_constant2015_lag1, wgi_composite_estimate_lag1. "
      f"{caveat} *** p<0.01  ** p<0.05  * p<0.1"] +
     [""] * (len(t4_df.columns) - 1)],
    columns=t4_df.columns,
)
t4_out = pd.concat([title4, t4_df.astype(str), footer4], ignore_index=True)
t4_path = os.path.join(LAYER_C, "table4_c3_sector_comparison.csv")
t4_out.to_csv(t4_path, index=False)
print(f"  Saved: {t4_path}")

# --- TABLE 5: Industry entrants list ---
ind_entrants = entrants_final[entrants_final["industry_at_entry"] == 1].copy()
t5_cols = ["cpi_id", "country", "iso3", "scope", "entry_year",
           "first_coverage_year", "gap_years", "sectors_at_entry",
           "cbam_exposure_avg"]
ind_entrants = ind_entrants[[c for c in t5_cols if c in ind_entrants.columns]]
ind_entrants = ind_entrants.sort_values("cbam_exposure_avg", ascending=False)

t5_path = os.path.join(LAYER_C, "table5_industry_entrants.csv")
ind_entrants.to_csv(t5_path, index=False)
print(f"  Saved: {t5_path} ({len(ind_entrants)} industry entrants)")

# Save full entrant dataset
entrants_path = os.path.join(LAYER_C, "entrant_sample.csv")
entrants_final.to_csv(entrants_path, index=False)
print(f"  Saved: {entrants_path}")

print(f"  Tables done in {time.time() - t6:.1f}s\n")

# ============================================================================
# VALIDATION 5 — Sample size warnings
# ============================================================================
print("=" * 72)
print("VALIDATION 5 — Sample size summary")
print("=" * 72)

for key, res in c1_results.items():
    if isinstance(res, dict):
        n = res["n"]
    elif res is not None:
        n = int(res.nobs)
    else:
        continue
    label = f"{key[0]} {key[1]} {key[2]}"
    if n < 30:
        print(f"  WARNING: Small entrant sample (N={n}) for {label}. "
              f"Interpret coefficients as indicative only.")
    else:
        print(f"  {label}: N={n}")

print(f"\n{'=' * 72}")
print(f"Layer C complete. Total time: {time.time() - t0:.1f}s")
print(f"Output directory: {LAYER_C}")
print(f"{'=' * 72}")
