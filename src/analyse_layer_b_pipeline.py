"""
Layer B — Pipeline Outcomes & Event-History Models
===================================================

World Bank research project: EU CBAM spillovers and carbon pricing.

Layer A built a Bruegel-comparable country-year TWFE benchmark using
national policies and static exposure. Layer B extends the analysis:

1. Pipeline outcomes anchored to master-list policy timing dates
   (cpi_list.xlsx), NOT database-entry dates from S&T reporting.
2. Side-by-side comparison of benchmark (static) vs preferred
   (time-varying, Brexit-corrected) exposure in TWFE regressions.
3. Cox proportional hazard models for pipeline transitions with
   Kaplan-Meier survival curves by exposure tercile.

CRITICAL DISTINCTION — DATABASE ENTRY VS POLICY TIMING
-------------------------------------------------------
First appearance in cpi_panel (coverage_active == 1) reflects the
State & Trends reporting cycle. Master-list dates in cpi_list.xlsx
record when a policy was actually under consideration, under
development, or implemented. This layer uses master-list dates
exclusively for pipeline timing.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

try:
    from lifelines import CoxTimeVaryingFitter, CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("WARNING: lifelines not installed. Run: pip install lifelines")
    print("         Cox models and KM plots will be skipped.")

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = os.path.join(ROOT, "data", "output")
DATA_LOOKUP = os.path.join(ROOT, "data", "lookup")
LAYER_B = os.path.join(DATA_OUT, "layer_b")
os.makedirs(LAYER_B, exist_ok=True)

CPI_PANEL_PATH = os.path.join(DATA_OUT, "cpi_panel.csv")
CPI_LIST_PATH = os.path.join(DATA_LOOKUP, "cpi_list.xlsx")
EXPOSURE_AVG_PATH = os.path.join(DATA_OUT, "cbam_exposure_avg.csv")
EXPOSURE_ANN_PATH = os.path.join(DATA_OUT, "cbam_exposure_annual.csv")
CONTROLS_PATH = os.path.join(DATA_OUT, "controls_panel.csv")

CONTROL_VARS = [
    "log_gdp_pc_constant2015_lag1",
    "log_exports_current_usd_lag1",
    "wgi_composite_estimate_lag1",
]

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


def _fmt_coef(res, var):
    """Return formatted coefficient string: coef***, (SE), N."""
    coef = res.params[var]
    se = res.std_errors[var]
    pv = res.pvalues[var]
    n = int(res.nobs)
    return f"{coef:.4f}{_stars(pv)}", f"({se:.4f})", str(n)


# ============================================================================
# STEP 0 — Load master-list timing from cpi_list.xlsx
# ============================================================================
t0 = time.time()
print("=" * 72)
print("STEP 0 — Load master-list timing from cpi_list.xlsx")
print("=" * 72)

master = pd.read_excel(CPI_LIST_PATH, sheet_name="Sheet1")

# Extract and rename columns
master = master.rename(columns={
    "Unique identifier": "cpi_id",
    "Country/region": "country_master",
    "Scope": "scope_master",
    "Type": "type_master",
    "Start date": "first_implementation_year",
    "End date": "end_year_master",
    "Under consideration start": "first_consideration_year",
    "Under consideration end": "uc_end",
    "Under development start": "first_development_year",
    "Under development end": "ud_end",
})

# Convert date columns from float to nullable integer
DATE_COLS = [
    "first_consideration_year", "first_development_year",
    "first_implementation_year", "end_year_master",
]
for col in DATE_COLS:
    master[col] = pd.to_numeric(master[col], errors="coerce").astype("Int64")

master_timing = master[["cpi_id"] + DATE_COLS].copy()
print(f"  Master-list CPIs loaded: {len(master_timing):,}")
print(f"  With consideration date:  {master_timing['first_consideration_year'].notna().sum()}")
print(f"  With development date:    {master_timing['first_development_year'].notna().sum()}")
print(f"  With implementation date: {master_timing['first_implementation_year'].notna().sum()}")
print(f"  Step 0 done in {time.time() - t0:.1f}s\n")

# ============================================================================
# STEP 1 — Load CPI panel, filter national, merge master-list timing
# ============================================================================
t1 = time.time()
print("=" * 72)
print("STEP 1 — Load CPI panel, filter national, merge master-list timing")
print("=" * 72)

cpi = pd.read_csv(CPI_PANEL_PATH)
print(f"  CPI-year rows loaded:                  {len(cpi):,}")

cpi = cpi[cpi["scope"] == "National"].copy()
print(f"  After scope == National:               {len(cpi):,}")

cpi = cpi[cpi["is_eea_ets_extension"] != 1].copy()
print(f"  After dropping EEA ETS extensions:     {len(cpi):,}")

cpi = cpi[cpi["instrument_type"] != "Undefined"].copy()
print(f"  After dropping instrument_type=Undef:  {len(cpi):,}")

# Merge master-list timing onto CPI panel
cpi = cpi.merge(master_timing, on="cpi_id", how="left")

# Compute pipeline status from master-list dates
cpi["ever_considered"] = (
    cpi["first_consideration_year"].notna() &
    (cpi["year"] >= cpi["first_consideration_year"])
).astype(int)

cpi["ever_developing"] = (
    cpi["first_development_year"].notna() &
    (cpi["year"] >= cpi["first_development_year"])
).astype(int)

cpi["ever_implemented"] = (
    cpi["first_implementation_year"].notna() &
    (cpi["year"] >= cpi["first_implementation_year"]) &
    (cpi["end_year_master"].isna() | (cpi["year"] <= cpi["end_year_master"]))
).astype(int)

print(f"  Pipeline status computed from master-list dates.")
print(f"  ever_considered mean:  {cpi['ever_considered'].mean():.3f}")
print(f"  ever_developing mean:  {cpi['ever_developing'].mean():.3f}")
print(f"  ever_implemented mean: {cpi['ever_implemented'].mean():.3f}")
print(f"  Step 1 done in {time.time() - t1:.1f}s\n")

# ============================================================================
# STEP 2 — Collapse to country-year
# ============================================================================
t2 = time.time()
print("=" * 72)
print("STEP 2 — Collapse to country-year")
print("=" * 72)

g = cpi.groupby(["iso3", "year"])

cy_outcomes = pd.DataFrame({
    "ever_considered": g["ever_considered"].max(),
    "ever_developing": g["ever_developing"].max(),
    "ever_implemented": g["ever_implemented"].max(),
}).reset_index()

# First milestone years per country (minimum across national CPIs)
first_years = cpi.groupby("iso3").agg(
    first_ever_considered_year=("first_consideration_year", "min"),
    first_ever_developing_year=("first_development_year", "min"),
    first_ever_implemented_year=("first_implementation_year", "min"),
).reset_index()

# Convert to nullable integer
for col in ["first_ever_considered_year", "first_ever_developing_year",
            "first_ever_implemented_year"]:
    first_years[col] = first_years[col].astype("Int64")

print(f"  Country-year outcome rows: {len(cy_outcomes):,}")
print(f"  Unique countries with CPIs: {cy_outcomes['iso3'].nunique()}")
print(f"  Step 2 done in {time.time() - t2:.1f}s\n")

# ============================================================================
# STEP 3 — Build spine and merge exposure
# ============================================================================
t3 = time.time()
print("=" * 72)
print("STEP 3 — Build spine and merge exposure")
print("=" * 72)

# Load exposure base (196 non-EU/non-EFTA countries)
exp_avg = pd.read_csv(EXPOSURE_AVG_PATH)
print(f"  Countries in exposure universe: {len(exp_avg):,}")

# Build full spine
years = list(range(2016, 2025))
spine = exp_avg[["iso3", "country_name", "cbam_exposure_avg"]].assign(key=1).merge(
    pd.DataFrame({"year": years, "key": 1}), on="key"
).drop(columns="key")
print(f"  Spine: {len(spine):,} rows ({spine['iso3'].nunique()} countries × {len(years)} years)")

# LEFT JOIN outcomes onto spine
cy = spine.merge(cy_outcomes, on=["iso3", "year"], how="left")
for col in ["ever_considered", "ever_developing", "ever_implemented"]:
    cy[col] = cy[col].fillna(0).astype(int)

# Merge first-year milestones
cy = cy.merge(first_years, on="iso3", how="left")

# Merge annual (time-varying) exposure
exp_ann = pd.read_csv(EXPOSURE_ANN_PATH).rename(columns={"t": "year"})
exp_ann = exp_ann[["iso3", "year", "cbam_exposure_annual"]]
cy = cy.merge(exp_ann, on=["iso3", "year"], how="left")

print(f"  Static exposure non-null:  {cy['cbam_exposure_avg'].notna().sum():,} / {len(cy):,}")
print(f"  Annual exposure non-null:  {cy['cbam_exposure_annual'].notna().sum():,} / {len(cy):,}")
print(f"  Step 3 done in {time.time() - t3:.1f}s\n")

# ============================================================================
# STEP 4 — Merge controls
# ============================================================================
t4 = time.time()
print("=" * 72)
print("STEP 4 — Merge controls")
print("=" * 72)

controls = pd.read_csv(CONTROLS_PATH)
ctrl_cols = ["iso3", "year"] + CONTROL_VARS
cy = cy.merge(controls[ctrl_cols], on=["iso3", "year"], how="left")

print("  Control completeness:")
for c in CONTROL_VARS:
    share = cy[c].notna().mean()
    flag = " *** >20% MISSING ***" if share < 0.80 else ""
    print(f"    {c}: {share:.1%} non-null{flag}")
print(f"  Step 4 done in {time.time() - t4:.1f}s\n")

# ============================================================================
# STEP 5 — Sample restriction and post indicators
# ============================================================================
t5 = time.time()
print("=" * 72)
print("STEP 5 — Sample restriction and post indicators")
print("=" * 72)

cy = cy[(cy["year"] >= 2016) & (cy["year"] <= 2024)].copy()
print(f"  After year 2016-2024:           {len(cy):,} rows")

cy = cy[cy["cbam_exposure_avg"].notna()].copy()
print(f"  After dropping null exposure:   {len(cy):,} rows")
print(f"  Unique countries in sample:     {cy['iso3'].nunique()}")

# Post indicators
cy["post_2019"] = (cy["year"] > 2019).astype(int)
cy["post_2021"] = (cy["year"] > 2021).astype(int)
cy["post_2023"] = (cy["year"] > 2023).astype(int)

# Interaction terms — benchmark (static)
cy["bench_x_post2019"] = cy["cbam_exposure_avg"] * cy["post_2019"]
cy["bench_x_post2021"] = cy["cbam_exposure_avg"] * cy["post_2021"]
cy["bench_x_post2023"] = cy["cbam_exposure_avg"] * cy["post_2023"]

# Interaction terms — preferred (time-varying annual)
cy["pref_x_post2019"] = cy["cbam_exposure_annual"] * cy["post_2019"]
cy["pref_x_post2021"] = cy["cbam_exposure_annual"] * cy["post_2021"]
cy["pref_x_post2023"] = cy["cbam_exposure_annual"] * cy["post_2023"]

# Save panel
panel_path = os.path.join(LAYER_B, "country_year_panel_b.csv")
cy.to_csv(panel_path, index=False)
print(f"  Saved: {panel_path}")
print(f"  Step 5 done in {time.time() - t5:.1f}s\n")

# ============================================================================
# REGRESSIONS PART 1 — TWFE: Benchmark vs Preferred Exposure
# ============================================================================
t6 = time.time()
print("=" * 72)
print("REGRESSIONS PART 1 — TWFE: Benchmark vs Preferred Exposure")
print("=" * 72)

OUTCOMES = ["ever_considered", "ever_developing", "ever_implemented"]
POST_INDICATORS = ["post_2019", "post_2021", "post_2023"]

BENCH_INTERACTION_MAP = {
    "post_2019": "bench_x_post2019",
    "post_2021": "bench_x_post2021",
    "post_2023": "bench_x_post2023",
}
PREF_INTERACTION_MAP = {
    "post_2019": "pref_x_post2019",
    "post_2021": "pref_x_post2021",
    "post_2023": "pref_x_post2023",
}

panel_df = cy.set_index(["iso3", "year"])
results_twfe = {}

for outcome in OUTCOMES:
    for post_var in POST_INDICATORS:
        for exp_type, int_map in [("benchmark", BENCH_INTERACTION_MAP),
                                  ("preferred", PREF_INTERACTION_MAP)]:
            interaction = int_map[post_var]
            exog_vars = [interaction] + CONTROL_VARS

            reg_cols = [outcome] + exog_vars
            reg_df = panel_df[reg_cols].dropna()

            dep = reg_df[outcome]
            exog = reg_df[exog_vars]

            try:
                mod = PanelOLS(
                    dep, exog,
                    entity_effects=True,
                    time_effects=True,
                    drop_absorbed=True,
                )
                res = mod.fit(cov_type="clustered", cluster_entity=True)
                results_twfe[(outcome, post_var, exp_type)] = res

                coef = res.params[interaction]
                se = res.std_errors[interaction]
                pv = res.pvalues[interaction]
                n = int(res.nobs)

                print(f"  {outcome:20s} | {post_var} | {exp_type:10s} | "
                      f"β={coef:+.5f} SE={se:.5f} p={pv:.4f} N={n}")
            except Exception as e:
                print(f"  FAILED: {outcome} | {post_var} | {exp_type} — {e}")
                results_twfe[(outcome, post_var, exp_type)] = None

print(f"\n  TWFE regressions done in {time.time() - t6:.1f}s\n")

# ============================================================================
# REGRESSIONS PART 2 — Cox Proportional Hazard Models
# ============================================================================
t7 = time.time()
print("=" * 72)
print("REGRESSIONS PART 2 — Cox Proportional Hazard Models")
print("=" * 72)

cox_results = {}
km_data = {}

if not HAS_LIFELINES:
    print("  SKIPPED — lifelines not installed.\n")
else:
    # Base cross-section: one row per country with time-invariant covariates
    # Use 2018 values for controls as baseline
    ctrl_2018 = cy[cy["year"] == 2018][["iso3", "cbam_exposure_avg"] + CONTROL_VARS].copy()
    ctrl_2018 = ctrl_2018.rename(columns={
        "log_gdp_pc_constant2015_lag1": "log_gdp_pc",
        "log_exports_current_usd_lag1": "log_exports",
        "wgi_composite_estimate_lag1": "wgi_composite",
    })

    # Rescale exposure to percentage points (0-100) for interpretable HRs.
    # Montenegro becomes 22.18 (not 0.2218), Turkey 3.08 (not 0.0308).
    # HR represents the effect of a 1 percentage-point increase in exposure.
    ctrl_2018["cbam_exposure_pct"] = ctrl_2018["cbam_exposure_avg"] * 100
    print(f"  Exposure rescaled to pct: mean={ctrl_2018['cbam_exposure_pct'].mean():.2f}, "
          f"max={ctrl_2018['cbam_exposure_pct'].max():.2f}")

    # Countries in the panel and their first/last year
    country_years = cy.groupby("iso3").agg(
        first_panel_year=("year", "min"),
        last_panel_year=("year", "max"),
    ).reset_index()

    # Merge milestone years
    base = country_years.merge(
        cy.drop_duplicates("iso3")[["iso3",
                                     "first_ever_considered_year",
                                     "first_ever_developing_year",
                                     "first_ever_implemented_year"]],
        on="iso3", how="left"
    )
    base = base.merge(ctrl_2018, on="iso3", how="left")

    TRANSITIONS = [
        ("consideration", "first_ever_considered_year"),
        ("development", "first_ever_developing_year"),
        ("implementation", "first_ever_implemented_year"),
    ]

    for transition_name, milestone_col in TRANSITIONS:
        print(f"\n  --- Model: {transition_name} ---")

        df = base.copy()

        # Left-censoring: exclude countries that reached THIS milestone before 2016
        milestone_vals = df[milestone_col]
        left_censored = milestone_vals.notna() & (milestone_vals < 2016)
        n_left_censored = left_censored.sum()
        df = df[~left_censored].copy()

        # Spell start: max(2016, first_panel_year)
        df["spell_start"] = df["first_panel_year"].clip(lower=2016)

        # Event: did the milestone occur within the window?
        df["event_year"] = df[milestone_col]
        df["event"] = (
            df["event_year"].notna() &
            (df["event_year"] >= df["spell_start"]) &
            (df["event_year"] <= 2024)
        ).astype(int)

        # Spell end
        df["spell_end"] = df.apply(
            lambda r: int(r["event_year"]) if r["event"] == 1 else 2024,
            axis=1,
        )

        # Duration
        df["duration"] = df["spell_end"] - df["spell_start"] + 1

        # Drop rows with missing covariates
        # Time-varying post_2019 covariate excluded due to implementation
        # issue: CoxTimeVaryingFitter returned HR = 1.000 exactly, indicating
        # the covariate was not contributing. The relative-time encoding
        # (start/stop as years-since-spell-start) strips calendar-year
        # information, making post_2019 collinear with spell structure.
        # Reported as limitation.
        cox_covariates = ["cbam_exposure_pct", "log_gdp_pc", "wgi_composite"]
        df = df.dropna(subset=cox_covariates + ["duration"])
        df = df[df["duration"] > 0].copy()

        n_at_risk = len(df)
        n_events = df["event"].sum()
        n_censored = n_at_risk - n_events

        print(f"    Left-censored excluded: {n_left_censored}")
        print(f"    Countries at risk:      {n_at_risk}")
        print(f"    Events observed:        {n_events}")
        print(f"    Right-censored:         {n_censored}")

        # Store KM data before trying Cox
        km_data[transition_name] = df[["iso3", "duration", "event",
                                        "cbam_exposure_avg"]].copy()

        if n_events < 3:
            print(f"    Too few events ({n_events}) — skipping Cox model.")
            cox_results[transition_name] = None
            continue

        # CoxPHFitter with rescaled exposure (percentage points)
        ph_df = df[["duration", "event"] + cox_covariates].copy()

        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(
                ph_df,
                duration_col="duration",
                event_col="event",
                show_progress=False,
            )
            cox_results[transition_name] = {
                "fitter": cph,
                "type": "CoxPHFitter",
                "training_df": ph_df,
                "n_at_risk": n_at_risk,
                "n_events": n_events,
                "n_censored": n_censored,
                "n_left_censored": n_left_censored,
                "c_index": cph.concordance_index_,
            }
            print(f"    CoxPHFitter converged.")
            print(f"    Concordance: {cph.concordance_index_:.4f}")
            print(cph.summary.to_string())
        except Exception as e:
            print(f"    CoxPHFitter failed: {e}")
            cox_results[transition_name] = None

    # --- Proportional Hazards Assumption Check ---
    print(f"\n  --- Proportional Hazards Assumption Check ---")
    for transition_name, _ in TRANSITIONS:
        res = cox_results.get(transition_name)
        if res is None:
            print(f"\n  {transition_name}: No model fitted.")
            continue
        fitter = res["fitter"]
        print(f"\n  {transition_name} (CoxPHFitter):")
        try:
            training_df = res.get("training_df")
            ph_check = fitter.check_assumptions(
                training_df,
                p_value_threshold=0.05,
                show_plots=False,
            )
            print(f"    Proportional hazards assumption test results printed above.")
            if hasattr(ph_check, '__len__') and len(ph_check) == 0:
                print(f"    No violations detected at p < 0.05.")
        except Exception as e:
            print(f"    check_assumptions() error: {e}")

    # --- Kaplan-Meier Survival Curves ---
    print(f"\n  --- Kaplan-Meier Survival Curves ---")
    for transition_name, _ in TRANSITIONS:
        df_km = km_data.get(transition_name)
        if df_km is None or len(df_km) < 10:
            print(f"  {transition_name}: Insufficient data for KM plot.")
            continue

        # Terciles based on cbam_exposure_avg
        tercile_labels = ["Low", "Medium", "High"]
        df_km = df_km.copy()
        try:
            df_km["tercile"] = pd.qcut(
                df_km["cbam_exposure_avg"], q=3,
                labels=tercile_labels, duplicates="drop",
            )
        except ValueError:
            # Fallback if too many ties for exact terciles
            thresholds = df_km["cbam_exposure_avg"].quantile([1/3, 2/3]).values
            df_km["tercile"] = pd.cut(
                df_km["cbam_exposure_avg"],
                bins=[-np.inf, thresholds[0], thresholds[1], np.inf],
                labels=tercile_labels,
            )

        fig, ax = plt.subplots(figsize=(8, 5))
        kmf = KaplanMeierFitter()

        groups = df_km["tercile"].unique()
        for grp in sorted(groups, key=lambda x: tercile_labels.index(str(x))):
            mask = df_km["tercile"] == grp
            kmf.fit(
                df_km.loc[mask, "duration"],
                event_observed=df_km.loc[mask, "event"],
                label=f"{grp} exposure",
            )
            kmf.plot_survival_function(ax=ax)

        # Log-rank test: high vs low
        high_mask = df_km["tercile"] == "High"
        low_mask = df_km["tercile"] == "Low"
        if high_mask.sum() > 0 and low_mask.sum() > 0:
            lr = logrank_test(
                df_km.loc[high_mask, "duration"],
                df_km.loc[low_mask, "duration"],
                event_observed_A=df_km.loc[high_mask, "event"],
                event_observed_B=df_km.loc[low_mask, "event"],
            )
            lr_p = lr.p_value
            ax.set_title(f"KM Survival: {transition_name.title()} Transition\n"
                         f"Log-rank test (High vs Low): p={lr_p:.4f}")
        else:
            ax.set_title(f"KM Survival: {transition_name.title()} Transition")

        ax.set_xlabel("Years since spell start")
        ax.set_ylabel("Survival probability (no transition)")
        ax.legend(loc="best")
        fig.tight_layout()

        km_path = os.path.join(LAYER_B, f"km_{transition_name}.png")
        fig.savefig(km_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {km_path}")

    print(f"\n  Cox models done in {time.time() - t7:.1f}s\n")

    # ========================================================================
    # REGRESSIONS PART 3 — Discrete-time hazard models (cloglog)
    # ========================================================================
    t_dt = time.time()
    print("=" * 72)
    print("REGRESSIONS PART 3 — Discrete-time hazard models (cloglog)")
    print("=" * 72)

    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.families.links import CLogLog
    import statsmodels.api as sm_api

    # Load time-varying covariates for the person-year expansion
    exp_ann_dt = pd.read_csv(EXPOSURE_ANN_PATH).rename(columns={"t": "year"})
    exp_ann_dt = exp_ann_dt[["iso3", "year", "cbam_exposure_annual"]].copy()
    exp_ann_dt["cbam_exposure_annual_pct"] = exp_ann_dt["cbam_exposure_annual"] * 100

    ctrl_dt = pd.read_csv(CONTROLS_PATH)
    ctrl_dt = ctrl_dt[["iso3", "year",
                        "log_gdp_pc_constant2015_lag1",
                        "wgi_composite_estimate_lag1"]].copy()

    dt_rows = []

    for transition_name, milestone_col in TRANSITIONS:
        print(f"\n  --- Discrete-time: {transition_name} ---")

        # Reconstruct spell data (same logic as Cox block above)
        df_dt = base.copy()
        milestone_vals = df_dt[milestone_col]
        left_censored = milestone_vals.notna() & (milestone_vals < 2016)
        df_dt = df_dt[~left_censored].copy()
        df_dt["spell_start"] = df_dt["first_panel_year"].clip(lower=2016)
        df_dt["event_year"] = df_dt[milestone_col]
        df_dt["event"] = (
            df_dt["event_year"].notna() &
            (df_dt["event_year"] >= df_dt["spell_start"]) &
            (df_dt["event_year"] <= 2024)
        ).astype(int)
        df_dt["spell_end"] = df_dt.apply(
            lambda r: int(r["event_year"]) if r["event"] == 1 else 2024,
            axis=1,
        )

        # Expand to person-year risk set
        py_rows = []
        for _, r in df_dt.iterrows():
            s, e = int(r["spell_start"]), int(r["spell_end"])
            for yr in range(s, e + 1):
                py_rows.append({
                    "iso3": r["iso3"],
                    "year": yr,
                    "event": 1 if (yr == e and r["event"] == 1) else 0,
                    "spell_start": s,
                    "duration_year": yr - s + 1,
                })
        py_df = pd.DataFrame(py_rows)

        # Merge time-varying exposure
        py_df = py_df.merge(exp_ann_dt[["iso3", "year", "cbam_exposure_annual_pct"]],
                            on=["iso3", "year"], how="left")
        # Merge time-varying controls
        py_df = py_df.merge(ctrl_dt, on=["iso3", "year"], how="left")

        # Duration dummies (non-parametric baseline hazard)
        max_dur = int(py_df["duration_year"].max())
        for d in range(1, max_dur + 1):
            py_df[f"dur_{d}"] = (py_df["duration_year"] == d).astype(int)
        dur_cols = [f"dur_{d}" for d in range(1, max_dur + 1)]

        # Drop rows with missing covariates
        cov_cols = ["cbam_exposure_annual_pct",
                    "log_gdp_pc_constant2015_lag1",
                    "wgi_composite_estimate_lag1"]
        py_clean = py_df.dropna(subset=cov_cols).copy()

        n_obs = len(py_clean)
        n_events = py_clean["event"].sum()
        n_countries = py_clean["iso3"].nunique()
        print(f"    Person-years: {n_obs}, events: {n_events}, countries: {n_countries}")

        if n_events < 3:
            print(f"    Too few events ({n_events}) — skipping.")
            dt_rows.append({
                "Transition": transition_name, "N_person_years": n_obs,
                "N_events": n_events, "N_countries": n_countries,
                "HR_cloglog": np.nan, "CI_lower": np.nan, "CI_upper": np.nan,
                "p_value": np.nan, "Stars": "",
            })
            continue

        exog_cols = ["cbam_exposure_annual_pct"] + cov_cols[1:] + dur_cols
        exog = sm_api.add_constant(py_clean[exog_cols])
        dep = py_clean["event"]
        groups = py_clean["iso3"]

        try:
            mod = GLM(dep, exog, family=Binomial(link=CLogLog()))
            res = mod.fit(cov_type="cluster", cov_kwds={"groups": groups})

            coef_exp = res.params["cbam_exposure_annual_pct"]
            se_exp = res.bse["cbam_exposure_annual_pct"]
            pv_exp = res.pvalues["cbam_exposure_annual_pct"]
            hr = np.exp(coef_exp)
            ci_lo = np.exp(coef_exp - 1.96 * se_exp)
            ci_hi = np.exp(coef_exp + 1.96 * se_exp)

            # Compare with Cox HR
            cox_hr_str = "—"
            cox_res = cox_results.get(transition_name)
            if cox_res is not None:
                cox_summary = cox_res["fitter"].summary
                if "cbam_exposure_pct" in cox_summary.index:
                    cox_hr_val = np.exp(cox_summary.loc["cbam_exposure_pct", "coef"])
                    cox_hr_str = f"{cox_hr_val:.4f}"

            print(f"    Cloglog HR: {hr:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]{_stars(pv_exp)}")
            print(f"    Cox HR:     {cox_hr_str}")

            dt_rows.append({
                "Transition": transition_name, "N_person_years": n_obs,
                "N_events": int(n_events), "N_countries": n_countries,
                "HR_cloglog": hr, "CI_lower": ci_lo, "CI_upper": ci_hi,
                "p_value": pv_exp, "Stars": _stars(pv_exp),
                "Cox_HR": cox_hr_str,
            })
        except Exception as e:
            print(f"    GLM cloglog failed: {e}")
            dt_rows.append({
                "Transition": transition_name, "N_person_years": n_obs,
                "N_events": int(n_events), "N_countries": n_countries,
                "HR_cloglog": np.nan, "CI_lower": np.nan, "CI_upper": np.nan,
                "p_value": np.nan, "Stars": "", "Cox_HR": "—",
            })

    dt_df = pd.DataFrame(dt_rows)
    dt_path = os.path.join(LAYER_B, "table_discrete_time_hazard.csv")
    dt_df.to_csv(dt_path, index=False)
    print(f"\n  Saved: {dt_path}")
    print(f"  Discrete-time models done in {time.time() - t_dt:.1f}s\n")

    # ========================================================================
    # REGRESSIONS PART 4 — Cox robustness (penalizer & robust variance)
    # ========================================================================
    t_cr = time.time()
    print("=" * 72)
    print("REGRESSIONS PART 4 — Cox robustness (penalizer & robust variance)")
    print("=" * 72)

    cox_robust_rows = []

    for transition_name, milestone_col in TRANSITIONS:
        print(f"\n  --- Cox robustness: {transition_name} ---")

        # Reconstruct spell data
        df_cr = base.copy()
        milestone_vals = df_cr[milestone_col]
        left_censored = milestone_vals.notna() & (milestone_vals < 2016)
        df_cr = df_cr[~left_censored].copy()
        df_cr["spell_start"] = df_cr["first_panel_year"].clip(lower=2016)
        df_cr["event_year"] = df_cr[milestone_col]
        df_cr["event"] = (
            df_cr["event_year"].notna() &
            (df_cr["event_year"] >= df_cr["spell_start"]) &
            (df_cr["event_year"] <= 2024)
        ).astype(int)
        df_cr["spell_end"] = df_cr.apply(
            lambda r: int(r["event_year"]) if r["event"] == 1 else 2024,
            axis=1,
        )
        df_cr["duration"] = df_cr["spell_end"] - df_cr["spell_start"] + 1

        cox_covariates = ["cbam_exposure_pct", "log_gdp_pc", "wgi_composite"]
        df_cr = df_cr.dropna(subset=cox_covariates + ["duration"])
        df_cr = df_cr[df_cr["duration"] > 0].copy()

        n_events = int(df_cr["event"].sum())
        n_covariates = len(cox_covariates)
        epv = n_events / n_covariates if n_covariates > 0 else 0
        epv_flag = " *** EPV < 10 ***" if epv < 10 else ""
        print(f"    Events: {n_events}, Covariates: {n_covariates}, "
              f"EPV: {epv:.1f}{epv_flag}")

        if n_events < 3:
            print(f"    Too few events — skipping.")
            continue

        ph_df = df_cr[["duration", "event"] + cox_covariates].copy()

        configs = [
            (0.01, False, "penalizer=0.01, robust=False (original)"),
            (0.01, True,  "penalizer=0.01, robust=True"),
            (0.0,  True,  "penalizer=0.0,  robust=True"),
        ]

        for pen, robust, desc in configs:
            try:
                cph = CoxPHFitter(penalizer=pen)
                cph.fit(ph_df, duration_col="duration", event_col="event",
                        show_progress=False, robust=robust)
                summary = cph.summary
                cov = "cbam_exposure_pct"
                hr = np.exp(summary.loc[cov, "coef"])
                ci_cols_lo = [c for c in summary.columns if "lower" in c.lower()]
                ci_cols_hi = [c for c in summary.columns if "upper" in c.lower()]
                ci_lo = np.exp(summary.loc[cov, ci_cols_lo[0]]) if ci_cols_lo else np.nan
                ci_hi = np.exp(summary.loc[cov, ci_cols_hi[0]]) if ci_cols_hi else np.nan
                p = summary.loc[cov, "p"]

                print(f"    {desc}: HR={hr:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] "
                      f"p={p:.4f}{_stars(p)}")

                cox_robust_rows.append({
                    "Transition": transition_name,
                    "Penalizer": pen, "Robust": robust,
                    "Description": desc,
                    "HR": hr, "CI_lower": ci_lo, "CI_upper": ci_hi,
                    "p_value": p, "Stars": _stars(p),
                    "N_events": n_events, "EPV": epv,
                })
            except Exception as e:
                print(f"    {desc}: FAILED — {e}")

    cr_df = pd.DataFrame(cox_robust_rows)
    cr_path = os.path.join(LAYER_B, "cox_robustness_comparison.csv")
    cr_df.to_csv(cr_path, index=False)
    print(f"\n  Saved: {cr_path}")
    print(f"  Cox robustness done in {time.time() - t_cr:.1f}s\n")

# ============================================================================
# OUTPUT TABLES
# ============================================================================
t8 = time.time()
print("=" * 72)
print("OUTPUT TABLES")
print("=" * 72)

# --- Table 1: TWFE comparison — benchmark vs preferred exposure ---
print("\n  Building Table 1: TWFE Exposure Comparison...")

t1_columns = ["Outcome"]
for post_var in POST_INDICATORS:
    post_label = post_var.replace("post_", "Post-")
    t1_columns.append(f"Benchmark {post_label}")
    t1_columns.append(f"Preferred {post_label}")

t1_rows = []
for outcome in OUTCOMES:
    coef_row = {"Outcome": outcome}
    se_row = {"Outcome": ""}
    n_row = {"Outcome": ""}
    for post_var in POST_INDICATORS:
        post_label = post_var.replace("post_", "Post-")
        for exp_type, int_map, col_prefix in [
            ("benchmark", BENCH_INTERACTION_MAP, "Benchmark"),
            ("preferred", PREF_INTERACTION_MAP, "Preferred"),
        ]:
            col_name = f"{col_prefix} {post_label}"
            interaction = int_map[post_var]
            res = results_twfe.get((outcome, post_var, exp_type))
            if res is not None:
                c, s, n = _fmt_coef(res, interaction)
                coef_row[col_name] = c
                se_row[col_name] = s
                n_row[col_name] = f"N={n}"
            else:
                coef_row[col_name] = "—"
                se_row[col_name] = ""
                n_row[col_name] = ""
    t1_rows.extend([coef_row, se_row, n_row])

t1_df = pd.DataFrame(t1_rows, columns=t1_columns)

title1 = "Table 1: Pipeline Outcomes — Benchmark vs Preferred Exposure"
footer1 = (
    "Benchmark = static 2014-2024 average, EU27+UK perimeter. "
    "Preferred = time-varying annual, Brexit-corrected perimeter. "
    "All regressions include country and year FE, controls (log GDP pc, "
    "log exports, WGI composite, all lagged). SEs clustered at country level. "
    "*** p<0.01  ** p<0.05  * p<0.1"
)

title1_row = pd.DataFrame(
    [[title1] + [""] * (len(t1_columns) - 1)], columns=t1_columns
)
footer1_row = pd.DataFrame(
    [[footer1] + [""] * (len(t1_columns) - 1)], columns=t1_columns
)
t1_out = pd.concat([title1_row, t1_df, footer1_row], ignore_index=True)
t1_path = os.path.join(LAYER_B, "table1_twfe_exposure_comparison.csv")
t1_out.to_csv(t1_path, index=False)
print(f"  Saved: {t1_path}")

# --- Table 2: Cox hazard model results ---
print("\n  Building Table 2: Cox Hazard Results...")

if HAS_LIFELINES and any(v is not None for v in cox_results.values()):
    t2_columns = ["Covariate", "Consideration HR [95% CI]",
                   "Development HR [95% CI]", "Implementation HR [95% CI]"]
    t2_rows = []

    # Collect all covariates across models
    all_covs = ["cbam_exposure_pct", "log_gdp_pc", "wgi_composite"]

    for cov in all_covs:
        row = {"Covariate": cov}
        for transition_name, col_name in [
            ("consideration", "Consideration HR [95% CI]"),
            ("development", "Development HR [95% CI]"),
            ("implementation", "Implementation HR [95% CI]"),
        ]:
            res = cox_results.get(transition_name)
            if res is not None:
                fitter = res["fitter"]
                summary = fitter.summary
                if cov in summary.index:
                    hr = np.exp(summary.loc[cov, "coef"])
                    # Find CI columns
                    ci_cols = [c for c in summary.columns if "lower" in c.lower()]
                    ci_lower_col = ci_cols[0] if ci_cols else None
                    ci_cols_upper = [c for c in summary.columns if "upper" in c.lower()]
                    ci_upper_col = ci_cols_upper[0] if ci_cols_upper else None

                    if ci_lower_col and ci_upper_col:
                        ci_lo = np.exp(summary.loc[cov, ci_lower_col])
                        ci_hi = np.exp(summary.loc[cov, ci_upper_col])
                    else:
                        ci_lo = ci_hi = np.nan

                    p = summary.loc[cov, "p"]
                    row[col_name] = f"{hr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]{_stars(p)}"
                else:
                    row[col_name] = "—"
            else:
                row[col_name] = "—"
        t2_rows.append(row)

    # Add c-index row
    cidx_row = {"Covariate": "C-index"}
    for transition_name, col_name in [
        ("consideration", "Consideration HR [95% CI]"),
        ("development", "Development HR [95% CI]"),
        ("implementation", "Implementation HR [95% CI]"),
    ]:
        res = cox_results.get(transition_name)
        if res is not None:
            cidx = res.get("c_index", np.nan)
            cidx_row[col_name] = f"{cidx:.4f}" if not np.isnan(cidx) else "—"
        else:
            cidx_row[col_name] = "—"
    t2_rows.append(cidx_row)

    t2_df = pd.DataFrame(t2_rows, columns=t2_columns)

    title2 = "Table 2: Cox Proportional Hazard Models — Pipeline Transitions"
    footer2 = (
        "HR > 1 indicates higher hazard (faster transition). "
        "C-index reported in bottom row. "
        "Exposure rescaled to percentage points (0-100) for interpretability. "
        "HR represents effect of 1ppt increase. "
        "Covariates: cbam_exposure_pct (static benchmark, ppt scale), log GDP per capita and "
        "WGI composite (2018 baseline values). "
        "Time-varying post_2019 covariate excluded due to implementation issue "
        "(HR = 1.000 under CoxTimeVaryingFitter). Reported as limitation. "
        "*** p<0.01  ** p<0.05  * p<0.1"
    )
    title2_row = pd.DataFrame(
        [[title2] + [""] * (len(t2_columns) - 1)], columns=t2_columns
    )
    footer2_row = pd.DataFrame(
        [[footer2] + [""] * (len(t2_columns) - 1)], columns=t2_columns
    )
    t2_out = pd.concat([title2_row, t2_df, footer2_row], ignore_index=True)
    t2_path = os.path.join(LAYER_B, "table2_cox_hazard.csv")
    t2_out.to_csv(t2_path, index=False)
    print(f"  Saved: {t2_path}")
else:
    print("  Table 2 skipped — no Cox results available.")

# --- Table 3: Pipeline timing descriptive ---
print("\n  Building Table 3: Pipeline Timing Descriptive...")

# All countries with at least one milestone
desc = cy.drop_duplicates("iso3")[
    ["iso3", "country_name", "cbam_exposure_avg",
     "first_ever_considered_year", "first_ever_developing_year",
     "first_ever_implemented_year"]
].copy()

has_milestone = (
    desc["first_ever_considered_year"].notna() |
    desc["first_ever_developing_year"].notna() |
    desc["first_ever_implemented_year"].notna()
)
desc = desc[has_milestone].sort_values("cbam_exposure_avg", ascending=False)

t3_path = os.path.join(LAYER_B, "table3_pipeline_timing.csv")
desc.to_csv(t3_path, index=False)
print(f"  Saved: {t3_path} ({len(desc)} countries)")
print(f"  Tables done in {time.time() - t8:.1f}s\n")

# ============================================================================
# VALIDATION
# ============================================================================
print("=" * 72)
print("VALIDATION")
print("=" * 72)

# --- 1. Master-list timing vs database entry comparison ---
print("\n1. MASTER-LIST TIMING vs DATABASE ENTRY COMPARISON")
print("   (first_development_year from master-list vs first year coverage_active==1 in panel)")

cpi_full = pd.read_csv(CPI_PANEL_PATH)
cpi_full = cpi_full.merge(master_timing, on="cpi_id", how="left")

# First year coverage_active == 1 per CPI
db_entry = cpi_full[cpi_full["coverage_active"] == 1].groupby("cpi_id")["year"].min()
db_entry = db_entry.reset_index().rename(columns={"year": "first_db_year"})

compare = master_timing.merge(db_entry, on="cpi_id", how="inner")
compare["diff"] = compare["first_db_year"] - compare["first_development_year"]
compare = compare.dropna(subset=["diff"])
compare = compare[compare["diff"] != 0].sort_values("diff", key=abs, ascending=False)

if len(compare) > 0:
    print(f"   Found {len(compare)} CPIs where dates differ. Top 5:")
    for _, row in compare.head(5).iterrows():
        print(f"     {row['cpi_id']:25s} master_dev={row['first_development_year']}  "
              f"db_entry={row['first_db_year']}  diff={row['diff']:+.0f} years")
else:
    print("   No differences found (or no overlapping CPIs with both dates).")

# --- 2. Hazard model sample sizes ---
print("\n2. HAZARD MODEL SAMPLE SIZES")
if HAS_LIFELINES:
    for transition_name, _ in TRANSITIONS:
        res = cox_results.get(transition_name)
        if res is not None:
            print(f"   {transition_name:20s}: at_risk={res['n_at_risk']}, "
                  f"events={res['n_events']}, censored={res['n_censored']}, "
                  f"left_excluded={res['n_left_censored']}")
        else:
            print(f"   {transition_name:20s}: model not fitted")
else:
    print("   Skipped — lifelines not installed.")

# --- 3. Proportional hazards assumption ---
print("\n3. PROPORTIONAL HAZARDS ASSUMPTION")
if HAS_LIFELINES:
    for transition_name, _ in TRANSITIONS:
        res = cox_results.get(transition_name)
        if res is None:
            print(f"   {transition_name}: No model fitted.")
            continue
        fitter = res["fitter"]
        training_df = res.get("training_df")
        try:
            fitter.check_assumptions(training_df, show_plots=False,
                                     p_value_threshold=0.05)
        except Exception as e:
            print(f"   {transition_name}: check_assumptions error — {e}")
else:
    print("   Skipped — lifelines not installed.")

# --- 4. Sensitivity: Post-2021 and Post-2023 signal comparison ---
print("\n4. SENSITIVITY: POST INDICATOR SIGNAL COMPARISON")
print("   (Does the later, more legally binding milestone show stronger signal?)")
for outcome in OUTCOMES:
    print(f"\n   {outcome}:")
    for exp_type in ["benchmark", "preferred"]:
        int_map = BENCH_INTERACTION_MAP if exp_type == "benchmark" else PREF_INTERACTION_MAP
        coefs = {}
        for post_var in POST_INDICATORS:
            interaction = int_map[post_var]
            res = results_twfe.get((outcome, post_var, exp_type))
            if res is not None:
                coefs[post_var] = res.params[interaction]
            else:
                coefs[post_var] = np.nan

        c19 = coefs.get("post_2019", np.nan)
        c21 = coefs.get("post_2021", np.nan)
        c23 = coefs.get("post_2023", np.nan)

        stronger_21 = "YES" if abs(c21) > abs(c19) else "no"
        stronger_23 = "YES" if abs(c23) > abs(c19) else "no"

        print(f"     {exp_type:10s}: Post-2019={c19:+.5f}, "
              f"Post-2021={c21:+.5f} (>{'>'}2019? {stronger_21}), "
              f"Post-2023={c23:+.5f} (>{'>'}2019? {stronger_23})")

# --- 5. Exposure sensitivity comparison ---
print("\n5. EXPOSURE SENSITIVITY COMPARISON")
print("   (Benchmark vs Preferred for ever_implemented with Post-2021)")

bench_res = results_twfe.get(("ever_implemented", "post_2021", "benchmark"))
pref_res = results_twfe.get(("ever_implemented", "post_2021", "preferred"))

if bench_res is not None and pref_res is not None:
    bench_coef = bench_res.params[BENCH_INTERACTION_MAP["post_2021"]]
    pref_coef = pref_res.params[PREF_INTERACTION_MAP["post_2021"]]
    diff = bench_coef - pref_coef
    print(f"   Benchmark exposure coefficient: {bench_coef:+.5f}")
    print(f"   Preferred exposure coefficient: {pref_coef:+.5f}")
    print(f"   Difference: {diff:+.5f}")
    print(f"   This reflects Brexit correction and time-varying perimeter adjustment.")
else:
    print("   One or both regressions unavailable.")

# ============================================================================
# DONE
# ============================================================================
print(f"\n{'=' * 72}")
print(f"Layer B complete. Total time: {time.time() - t0:.1f}s")
print(f"Output directory: {LAYER_B}")
print(f"{'=' * 72}")
