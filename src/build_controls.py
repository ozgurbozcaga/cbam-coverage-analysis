"""
build_controls.py

Constructs the country-year control variables panel for the World Bank
research project on EU CBAM spillovers and carbon pricing.

Sources
-------
  data/lookup/wdi.xlsx   (World Development Indicators)
  data/lookup/wgi.xlsx   (Worldwide Governance Indicators)

Outputs
-------
  data/output/controls_panel.csv
  data/output/controls_panel_data_dictionary.csv

Run from project root:
  python3 src/build_controls.py

Symlinks required before first run:
  ln -s "/mnt/c/Users/ozgur/OneDrive/Documents/WBG/CBAM_coverage/P_Data_Extract_From_World_Development_Indicators.xlsx" data/lookup/wdi.xlsx
  ln -s "/mnt/c/Users/ozgur/OneDrive/Documents/WBG/CBAM_coverage/wgidataset_with_sourcedata-2025.xlsx" data/lookup/wgi.xlsx
"""

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ===========================================================================
# CONFIGURATION
# ===========================================================================

WDI_PATH = "data/lookup/wdi.xlsx"
WGI_PATH = "data/lookup/wgi.xlsx"
CPI_PANEL_PATH = "data/output/cpi_panel.csv"
OUTPUT_PANEL = "data/output/controls_panel.csv"
OUTPUT_DICT = "data/output/controls_panel_data_dictionary.csv"

# Years to load from source (2014 needed to compute 2015 lags)
SOURCE_YEARS = list(range(2014, 2025))
# Final output years
OUTPUT_YEARS = list(range(2015, 2025))

# WDI series to extract: series_code -> variable_name
WDI_SERIES = {
    "NY.GDP.PCAP.KD":    "gdp_pc_constant2015",
    "NY.GDP.PCAP.CD":    "gdp_pc_current",
    "NE.EXP.GNFS.CD":   "exports_current_usd",
    "NE.EXP.GNFS.ZS":   "exports_pct_gdp",
    "NE.TRD.GNFS.ZS":   "trade_openness",
    "NY.GDP.DEFL.ZS.AD":    "gdp_deflator_linked",
    "NY.GDP.DEFL.KD.ZG.AD": "gdp_deflator_growth_linked",
}

# Variables for which we compute _lag1
WDI_LAG_VARS = [
    "gdp_pc_constant2015",
    "gdp_pc_current",
    "exports_current_usd",
    "exports_pct_gdp",
    "trade_openness",
]

# Log-transformed WDI variables (before logging)
LOG_SOURCE_VARS = [
    "gdp_pc_constant2015",
    "gdp_pc_constant2015_lag1",
    "exports_current_usd",
    "exports_current_usd_lag1",
]

# WGI sheets and corresponding short codes
WGI_SHEETS = ["ge", "rq", "rl", "cc", "va", "pv"]

# WGI variables for _lag1
WGI_LAG_VARS = [f"wgi_{s}_estimate" for s in WGI_SHEETS]

# WGI score columns (no lag needed in spec, but extract for output)
WGI_SCORE_VARS = [f"wgi_{s}_score" for s in ["ge", "rq", "rl", "cc"]]

# Four governance dimensions used to build composite
COMPOSITE_COMPONENTS = ["wgi_ge_estimate", "wgi_rq_estimate",
                        "wgi_rl_estimate", "wgi_cc_estimate"]


# ===========================================================================
# WDI LOADING AND RESHAPING
# ===========================================================================

def load_wdi() -> pd.DataFrame:
    """Load WDI Excel sheet 'Data' as-is (wide, one row per country-series)."""
    df = pd.read_excel(WDI_PATH, sheet_name="Data", engine="openpyxl",
                       dtype=str)
    return df


def reshape_wdi(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Melt WDI from wide (country × series) to long (iso3, year, variable, value).

    Year columns are formatted '2014 [YR2014]'; we extract the first 4 chars
    as integer year.  Missing values coded as '..' are replaced with NaN.
    Only series in WDI_SERIES and years in SOURCE_YEARS are kept.
    """
    # Identify year columns
    year_cols = [c for c in df_wide.columns if str(c)[:4].isdigit()]

    # Keep only the series we need
    df = df_wide[df_wide["Series Code"].isin(WDI_SERIES)].copy()

    # Melt to long
    id_vars = ["Country Name", "Country Code", "Series Name", "Series Code"]
    df_long = df.melt(id_vars=id_vars, value_vars=year_cols,
                      var_name="year_raw", value_name="value")

    # Parse year
    df_long["year"] = df_long["year_raw"].str[:4].astype(int)
    df_long = df_long[df_long["year"].isin(SOURCE_YEARS)].copy()

    # Replace '..' with NaN, then cast value to float
    df_long["value"] = df_long["value"].replace("..", np.nan)
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

    # Map series code to variable name
    df_long["variable"] = df_long["Series Code"].map(WDI_SERIES)

    # Rename country columns
    df_long = df_long.rename(columns={
        "Country Code": "iso3",
        "Country Name": "country_name",
    })

    return df_long[["iso3", "country_name", "year", "variable", "value"]]


def pivot_wdi(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot WDI long -> wide: one row per (iso3, year), one column per variable.

    country_name is carried along by taking the first value per iso3.
    """
    # Pivot values
    df_pivot = df_long.pivot_table(
        index=["iso3", "year"],
        columns="variable",
        values="value",
        aggfunc="first",
    ).reset_index()
    df_pivot.columns.name = None

    # Add country_name (stable per iso3)
    country_map = (df_long[["iso3", "country_name"]]
                   .drop_duplicates("iso3")
                   .set_index("iso3")["country_name"])
    df_pivot.insert(1, "country_name", df_pivot["iso3"].map(country_map))

    # Ensure all expected variable columns exist
    for var in WDI_SERIES.values():
        if var not in df_pivot.columns:
            df_pivot[var] = np.nan

    return df_pivot


# ===========================================================================
# WGI LOADING
# ===========================================================================

def load_wgi_sheet(sheet: str) -> pd.DataFrame:
    """
    Load one WGI sheet and return a long DataFrame with columns:
      iso3, year, wgi_{sheet}_estimate, wgi_{sheet}_score

    The WGI Excel uses these exact column names:
      'Economy (code)'                      -> iso3
      'Year'                                -> year
      'Governance estimate (approx. -2.5 to +2.5)' -> estimate
      'Governance score (0-100)'            -> score

    Missing values '..' are replaced with NaN.
    """
    df = pd.read_excel(WGI_PATH, sheet_name=sheet, engine="openpyxl",
                       dtype=str)

    # Normalise column names for robustness (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    # Locate the relevant columns: exact match (case-insensitive) first,
    # falling back to substring match to handle minor formatting variation.
    def find_col(partial: str) -> str:
        # 1. Exact match
        for c in df.columns:
            if c.lower() == partial.lower():
                return c
        # 2. Substring match
        matches = [c for c in df.columns if partial.lower() in c.lower()]
        if not matches:
            raise KeyError(
                f"Sheet '{sheet}': could not find column matching '{partial}'. "
                f"Available: {list(df.columns)}"
            )
        return matches[0]

    iso3_col     = find_col("Economy (code)")
    year_col     = find_col("Year")
    estimate_col = find_col("Governance estimate")
    score_col    = find_col("Governance score")

    df = df[[iso3_col, year_col, estimate_col, score_col]].copy()
    df.columns = ["iso3", "year", f"wgi_{sheet}_estimate", f"wgi_{sheet}_score"]

    # Replace '..' with NaN and cast to numeric
    for col in [f"wgi_{sheet}_estimate", f"wgi_{sheet}_score"]:
        df[col] = df[col].replace("..", np.nan)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cast year to int, filter to SOURCE_YEARS
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[df["year"].isin(SOURCE_YEARS)].copy()

    # Drop rows with no iso3
    df = df.dropna(subset=["iso3"])
    df["iso3"] = df["iso3"].str.strip()

    return df.reset_index(drop=True)


def load_all_wgi() -> pd.DataFrame:
    """
    Load all 6 WGI sheets and merge into a single wide DataFrame:
      iso3, year, wgi_ge_estimate, wgi_ge_score, ..., wgi_pv_estimate, wgi_pv_score
    """
    frames = []
    for sheet in WGI_SHEETS:
        frames.append(load_wgi_sheet(sheet))

    # Merge all sheets on iso3 × year
    df = frames[0]
    for other in frames[1:]:
        df = df.merge(other, on=["iso3", "year"], how="outer")

    return df.sort_values(["iso3", "year"]).reset_index(drop=True)


# ===========================================================================
# LAG, LOG, AND COMPOSITE CONSTRUCTION
# ===========================================================================

def build_lags(panel: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """
    Add _lag1 columns for each variable in `variables`.

    CRITICAL: sorts by ['iso3', 'year'] before computing lags to guarantee
    correct temporal ordering. Uses groupby('iso3').shift(1) so that lags
    never cross country boundaries.
    """
    panel = panel.sort_values(["iso3", "year"]).copy()

    for var in variables:
        if var not in panel.columns:
            panel[f"{var}_lag1"] = np.nan
            continue
        panel[f"{var}_lag1"] = (
            panel.groupby("iso3", sort=False)[var].shift(1)
        )

    return panel


def build_log_vars(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-transformed columns for LOG_SOURCE_VARS.

    Uses np.log(x) on raw values.  Zeros and negatives produce NaN.
    No +1 offset is applied because GDP per capita and exports should not be
    zero for any valid observation.
    """
    panel = panel.copy()
    for var in LOG_SOURCE_VARS:
        if var not in panel.columns:
            panel[f"log_{var}"] = np.nan
            continue
        panel[f"log_{var}"] = np.where(panel[var] > 0, np.log(panel[var]), np.nan)

    return panel


def build_composite(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Construct wgi_composite_estimate as the simple mean of the four core
    governance estimates: ge, rq, rl, cc.

    A composite is computed whenever at least one component is non-null
    (nanmean behaviour).  Rows where all four are NaN remain NaN.
    Then wgi_composite_estimate_lag1 is computed via the same sort-safe
    groupby shift.
    """
    panel = panel.copy()
    available = [c for c in COMPOSITE_COMPONENTS if c in panel.columns]
    if available:
        panel["wgi_composite_estimate"] = panel[available].mean(axis=1,
                                                                skipna=True)
        # Where all components are NaN, the mean is also NaN (correct)
        all_null = panel[available].isna().all(axis=1)
        panel.loc[all_null, "wgi_composite_estimate"] = np.nan
    else:
        panel["wgi_composite_estimate"] = np.nan

    panel = build_lags(panel, ["wgi_composite_estimate"])
    return panel


# ===========================================================================
# VALIDATION
# ===========================================================================

def validate_and_report(panel: pd.DataFrame) -> None:
    """Print a full validation summary to stdout."""
    sep = "=" * 70

    print(f"\n{sep}")
    print("VALIDATION REPORT — controls_panel")
    print(sep)

    # ------------------------------------------------------------------
    # 1. Panel dimensions
    # ------------------------------------------------------------------
    print("\n[1] Panel dimensions (output years 2015-2024)")
    out = panel[panel["year"].isin(OUTPUT_YEARS)]
    print(f"    Total rows       : {len(out):,}")
    print(f"    Unique countries : {out['iso3'].nunique()}")
    print(f"    Years covered    : {sorted(out['year'].unique())}")

    # ------------------------------------------------------------------
    # 2. WDI coverage by series and year
    # ------------------------------------------------------------------
    print(f"\n[2] WDI coverage — non-null observations by variable × year")
    wdi_vars = list(WDI_SERIES.values())
    wdi_cov = (out[["year"] + [v for v in wdi_vars if v in out.columns]]
               .groupby("year")
               .count())
    print(wdi_cov.to_string())

    print("\n    ** Flagged year-series combinations with < 150 non-null obs:")
    flagged = False
    for var in wdi_vars:
        if var not in wdi_cov.columns:
            continue
        low = wdi_cov[wdi_cov[var] < 150][[var]]
        if not low.empty:
            for yr, row in low.iterrows():
                print(f"       {var:35s} year={yr}  n={int(row[var])}")
                flagged = True
    if not flagged:
        print("       None — all series × year have >= 150 observations.")

    # ------------------------------------------------------------------
    # 3. WGI coverage by dimension and year
    # ------------------------------------------------------------------
    print(f"\n[3] WGI coverage — non-null estimate observations by sheet × year")
    wgi_est_vars = [f"wgi_{s}_estimate" for s in WGI_SHEETS]
    wgi_cov = (out[["year"] + [v for v in wgi_est_vars if v in out.columns]]
               .groupby("year")
               .count())
    print(wgi_cov.to_string())

    print("\n    ** Flagged year-dimension combinations with < 150 non-null obs:")
    flagged = False
    for var in wgi_est_vars:
        if var not in wgi_cov.columns:
            continue
        low = wgi_cov[wgi_cov[var] < 150][[var]]
        if not low.empty:
            for yr, row in low.iterrows():
                print(f"       {var:35s} year={yr}  n={int(row[var])}")
                flagged = True
    if not flagged:
        print("       None — all dimensions × year have >= 150 observations.")

    # ------------------------------------------------------------------
    # 4. Lag verification spot-check: Germany (DEU)
    # ------------------------------------------------------------------
    print(f"\n[4] Lag verification spot-check — Germany (DEU)")
    deu = panel[panel["iso3"] == "DEU"].sort_values("year")
    cols = ["year", "gdp_pc_constant2015", "gdp_pc_constant2015_lag1",
            "wgi_ge_estimate", "wgi_ge_estimate_lag1"]
    available = [c for c in cols if c in deu.columns]
    if deu.empty:
        print("    DEU not found in panel.")
    else:
        print(deu[available].to_string(index=False))
        # Spot-check: lag[t] == current[t-1]
        deu = deu.reset_index(drop=True)
        issues = []
        for t in range(1, len(deu)):
            for base in ["gdp_pc_constant2015", "wgi_ge_estimate"]:
                lag_col = f"{base}_lag1"
                if base not in deu.columns or lag_col not in deu.columns:
                    continue
                cur_val  = deu.at[t - 1, base]
                lag_val  = deu.at[t, lag_col]
                if pd.isna(cur_val) and pd.isna(lag_val):
                    continue
                if not np.isclose(cur_val, lag_val, equal_nan=True):
                    issues.append(
                        f"       MISMATCH: {base} year={deu.at[t,'year']} "
                        f"lag={lag_val:.4f} vs prior={cur_val:.4f}"
                    )
        if issues:
            print("\n    *** LAG ERRORS DETECTED:")
            for msg in issues:
                print(msg)
        else:
            print("\n    Lag values verified correct for DEU.")

    # ------------------------------------------------------------------
    # 5. GDP deflator coverage and range
    # ------------------------------------------------------------------
    print(f"\n[5] GDP deflator (gdp_deflator_linked) coverage and range")
    if "gdp_deflator_linked" in out.columns:
        defl_cov = out.groupby("year")["gdp_deflator_linked"].count()
        print("    Non-null obs by year:")
        print(defl_cov.to_string())
        defl_2020 = out.loc[out["year"] == 2020, "gdp_deflator_linked"].dropna()
        if not defl_2020.empty:
            print(f"\n    2020 deflator — min={defl_2020.min():.2f}, "
                  f"max={defl_2020.max():.2f}, "
                  f"mean={defl_2020.mean():.2f}  "
                  f"(values near 100 expected if base year is recent)")
        else:
            print("    No 2020 deflator data found.")
    else:
        print("    gdp_deflator_linked not in panel.")

    # ------------------------------------------------------------------
    # 6. Log variable validity
    # ------------------------------------------------------------------
    print(f"\n[6] Log variable validity")
    log_cols = [f"log_{v}" for v in LOG_SOURCE_VARS]
    for lc in log_cols:
        if lc not in out.columns:
            print(f"    {lc}: MISSING from panel")
            continue
        n_nan  = out[lc].isna().sum()
        n_inf  = np.isinf(out[lc]).sum()
        print(f"    {lc:42s}  NaN={n_nan:5d}  -inf={n_inf:4d}")

    # Flag negative source values
    for src in ["gdp_pc_constant2015", "gdp_pc_current", "exports_current_usd"]:
        if src not in out.columns:
            continue
        n_neg = (out[src] < 0).sum()
        if n_neg > 0:
            print(f"    *** WARNING: {src} has {n_neg} negative values — "
                  "check before log transform.")

    # ------------------------------------------------------------------
    # 7. WGI composite
    # ------------------------------------------------------------------
    print(f"\n[7] WGI composite estimate by year")
    if "wgi_composite_estimate" in out.columns:
        comp_stats = out.groupby("year")["wgi_composite_estimate"].agg(
            ["mean", "std", "count"]
        )
        comp_stats.columns = ["mean", "std", "n_non_null"]
        print(comp_stats.round(3).to_string())
        all_null = out[COMPOSITE_COMPONENTS].isna().all(axis=1).sum()
        print(f"\n    Rows where all 4 components are NaN "
              f"(composite = NaN): {all_null}")
    else:
        print("    wgi_composite_estimate not in panel.")

    # ------------------------------------------------------------------
    # 8. Merge readiness
    # ------------------------------------------------------------------
    print(f"\n[8] Merge readiness — controls_panel vs cpi_panel")
    try:
        cpi = pd.read_csv(CPI_PANEL_PATH, encoding="utf-8-sig",
                          usecols=["iso3", "year"])
        cpi_pairs = cpi.drop_duplicates()
        ctrl_pairs = out[["iso3", "year"]].drop_duplicates()

        merged = cpi_pairs.merge(ctrl_pairs, on=["iso3", "year"],
                                 how="left", indicator=True)
        matched   = (merged["_merge"] == "both").sum()
        unmatched = (merged["_merge"] == "left_only").sum()
        print(f"    Unique (iso3, year) pairs in cpi_panel   : {len(cpi_pairs)}")
        print(f"    Matched to controls_panel                : {matched}")
        print(f"    Unmatched (in cpi_panel, not in controls): {unmatched}")
        if unmatched > 0:
            miss_iso3 = (merged[merged["_merge"] == "left_only"]["iso3"]
                         .unique().tolist())
            print(f"    Unmatched iso3 codes: {miss_iso3}")
    except FileNotFoundError:
        print(f"    {CPI_PANEL_PATH} not found — skipping merge check.")

    print(f"\n{sep}\n")


# ===========================================================================
# DATA DICTIONARY
# ===========================================================================

def build_data_dictionary() -> pd.DataFrame:
    rows = [
        # Identifiers
        ("iso3", "ISO 3166-1 alpha-3 country code", "WDI / WGI", "", "string", "no", "Primary merge key onto cpi_panel"),
        ("country_name", "Country name", "WDI", "", "string", "no", ""),
        ("year", "Calendar year", "", "", "integer", "no", "2015-2024 in output"),

        # WDI — current year
        ("gdp_pc_constant2015", "GDP per capita, constant 2015 USD", "WDI", "NY.GDP.PCAP.KD", "USD (constant 2015)", "no", ""),
        ("gdp_pc_constant2015_lag1", "GDP per capita, constant 2015 USD, lagged 1 year", "WDI", "NY.GDP.PCAP.KD", "USD (constant 2015)", "t-1", ""),
        ("gdp_pc_current", "GDP per capita, current USD", "WDI", "NY.GDP.PCAP.CD", "USD (current)", "no", ""),
        ("gdp_pc_current_lag1", "GDP per capita, current USD, lagged 1 year", "WDI", "NY.GDP.PCAP.CD", "USD (current)", "t-1", ""),
        ("exports_current_usd", "Total exports of goods and services, current USD", "WDI", "NE.EXP.GNFS.CD", "USD (current)", "no", ""),
        ("exports_current_usd_lag1", "Total exports, current USD, lagged 1 year", "WDI", "NE.EXP.GNFS.CD", "USD (current)", "t-1", ""),
        ("exports_pct_gdp", "Exports of goods and services as % of GDP", "WDI", "NE.EXP.GNFS.ZS", "% of GDP", "no", ""),
        ("exports_pct_gdp_lag1", "Exports as % of GDP, lagged 1 year", "WDI", "NE.EXP.GNFS.ZS", "% of GDP", "t-1", ""),
        ("trade_openness", "Trade (exports + imports) as % of GDP", "WDI", "NE.TRD.GNFS.ZS", "% of GDP", "no", "Standard openness measure"),
        ("trade_openness_lag1", "Trade as % of GDP, lagged 1 year", "WDI", "NE.TRD.GNFS.ZS", "% of GDP", "t-1", ""),
        ("gdp_deflator_linked", "GDP deflator, linked series (index)", "WDI", "NY.GDP.DEFL.ZS.AD", "Index", "no",
         "Linked/harmonised series for cross-country comparability. "
         "Use to deflate nominal prices: real_price = nominal_price_usd / (gdp_deflator_linked / 100). "
         "Preferred over NY.GDP.DEFL.ZS whose base year varies by country."),
        ("gdp_deflator_growth_linked", "GDP deflator annual growth, linked series (%)", "WDI", "NY.GDP.DEFL.KD.ZG.AD", "% per year", "no",
         "Inflation measured via linked GDP deflator; preferred over NY.GDP.DEFL.KD.ZG."),

        # Log-transformed
        ("log_gdp_pc_constant2015", "Natural log of GDP per capita (constant 2015 USD)", "WDI", "NY.GDP.PCAP.KD", "log USD", "no", "log(gdp_pc_constant2015); NaN if source<=0"),
        ("log_gdp_pc_constant2015_lag1", "Natural log of lagged GDP per capita (constant 2015 USD)", "WDI", "NY.GDP.PCAP.KD", "log USD", "t-1", "log(gdp_pc_constant2015_lag1)"),
        ("log_exports_current_usd", "Natural log of total exports (current USD)", "WDI", "NE.EXP.GNFS.CD", "log USD", "no", "log(exports_current_usd)"),
        ("log_exports_current_usd_lag1", "Natural log of lagged exports (current USD)", "WDI", "NE.EXP.GNFS.CD", "log USD", "t-1", "log(exports_current_usd_lag1)"),

        # WGI estimates
        ("wgi_ge_estimate", "Government Effectiveness estimate", "WGI", "ge", "approx. -2.5 to +2.5", "no", "Standard WGI estimate; higher = more effective"),
        ("wgi_ge_estimate_lag1", "Government Effectiveness estimate, lagged 1 year", "WGI", "ge", "approx. -2.5 to +2.5", "t-1", ""),
        ("wgi_rq_estimate", "Regulatory Quality estimate", "WGI", "rq", "approx. -2.5 to +2.5", "no", ""),
        ("wgi_rq_estimate_lag1", "Regulatory Quality estimate, lagged 1 year", "WGI", "rq", "approx. -2.5 to +2.5", "t-1", ""),
        ("wgi_rl_estimate", "Rule of Law estimate", "WGI", "rl", "approx. -2.5 to +2.5", "no", ""),
        ("wgi_rl_estimate_lag1", "Rule of Law estimate, lagged 1 year", "WGI", "rl", "approx. -2.5 to +2.5", "t-1", ""),
        ("wgi_cc_estimate", "Control of Corruption estimate", "WGI", "cc", "approx. -2.5 to +2.5", "no", ""),
        ("wgi_cc_estimate_lag1", "Control of Corruption estimate, lagged 1 year", "WGI", "cc", "approx. -2.5 to +2.5", "t-1", ""),
        ("wgi_va_estimate", "Voice and Accountability estimate", "WGI", "va", "approx. -2.5 to +2.5", "no", "Lower priority control"),
        ("wgi_va_estimate_lag1", "Voice and Accountability estimate, lagged 1 year", "WGI", "va", "approx. -2.5 to +2.5", "t-1", ""),
        ("wgi_pv_estimate", "Political Stability and Absence of Violence estimate", "WGI", "pv", "approx. -2.5 to +2.5", "no", "Lower priority control"),
        ("wgi_pv_estimate_lag1", "Political Stability estimate, lagged 1 year", "WGI", "pv", "approx. -2.5 to +2.5", "t-1", ""),

        # WGI scores
        ("wgi_ge_score", "Government Effectiveness score (percentile rank 0-100)", "WGI", "ge", "0-100", "no", ""),
        ("wgi_rq_score", "Regulatory Quality score (percentile rank 0-100)", "WGI", "rq", "0-100", "no", ""),
        ("wgi_rl_score", "Rule of Law score (percentile rank 0-100)", "WGI", "rl", "0-100", "no", ""),
        ("wgi_cc_score", "Control of Corruption score (percentile rank 0-100)", "WGI", "cc", "0-100", "no", ""),

        # Composite
        ("wgi_composite_estimate", "Composite governance estimate (simple mean of ge, rq, rl, cc)", "WGI (derived)", "", "approx. -2.5 to +2.5", "no",
         "Simple mean of wgi_ge_estimate, wgi_rq_estimate, wgi_rl_estimate, wgi_cc_estimate. "
         "NaN only when all four components are NaN."),
        ("wgi_composite_estimate_lag1", "Composite governance estimate, lagged 1 year", "WGI (derived)", "", "approx. -2.5 to +2.5", "t-1", ""),
    ]

    dd = pd.DataFrame(rows, columns=[
        "variable_name", "description", "source", "series_code",
        "units", "lag", "notes"
    ])
    return dd


# ===========================================================================
# FINAL COLUMN ORDERING
# ===========================================================================

FINAL_COLS = [
    "iso3", "country_name", "year",
    # WDI current + lag
    "gdp_pc_constant2015", "gdp_pc_constant2015_lag1",
    "gdp_pc_current", "gdp_pc_current_lag1",
    "exports_current_usd", "exports_current_usd_lag1",
    "exports_pct_gdp", "exports_pct_gdp_lag1",
    "trade_openness", "trade_openness_lag1",
    "gdp_deflator_linked",
    "gdp_deflator_growth_linked",
    # Log transforms
    "log_gdp_pc_constant2015", "log_gdp_pc_constant2015_lag1",
    "log_exports_current_usd", "log_exports_current_usd_lag1",
    # WGI estimates + lags
    "wgi_ge_estimate", "wgi_ge_estimate_lag1",
    "wgi_rq_estimate", "wgi_rq_estimate_lag1",
    "wgi_rl_estimate", "wgi_rl_estimate_lag1",
    "wgi_cc_estimate", "wgi_cc_estimate_lag1",
    "wgi_va_estimate", "wgi_va_estimate_lag1",
    "wgi_pv_estimate", "wgi_pv_estimate_lag1",
    # WGI scores
    "wgi_ge_score", "wgi_rq_score", "wgi_rl_score", "wgi_cc_score",
    "wgi_va_score", "wgi_pv_score",
    # Composite
    "wgi_composite_estimate", "wgi_composite_estimate_lag1",
]


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load and process WDI
    # ------------------------------------------------------------------
    print("Loading WDI...")
    wdi_raw  = load_wdi()
    wdi_long = reshape_wdi(wdi_raw)
    wdi_wide = pivot_wdi(wdi_long)
    print(f"  WDI loaded: {len(wdi_wide):,} country-year rows "
          f"({wdi_wide['iso3'].nunique()} countries)")

    # ------------------------------------------------------------------
    # Step 2: Load and process WGI
    # ------------------------------------------------------------------
    print("Loading WGI...")
    wgi_wide = load_all_wgi()
    print(f"  WGI loaded: {len(wgi_wide):,} country-year rows "
          f"({wgi_wide['iso3'].nunique()} countries)")

    # ------------------------------------------------------------------
    # Step 3: Merge WDI + WGI on iso3 × year
    # ------------------------------------------------------------------
    print("Merging WDI and WGI...")
    panel = wdi_wide.merge(wgi_wide, on=["iso3", "year"], how="outer")
    # Recover country_name for WGI-only rows where possible
    country_map = (wdi_long[["iso3", "country_name"]]
                   .drop_duplicates("iso3")
                   .set_index("iso3")["country_name"])
    panel["country_name"] = panel["iso3"].map(country_map)
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    print(f"  Merged panel: {len(panel):,} rows")

    # ------------------------------------------------------------------
    # Step 4: Build lags (requires 2014 to be present for 2015 lags)
    # ------------------------------------------------------------------
    print("Building lag variables...")
    all_lag_vars = WDI_LAG_VARS + WGI_LAG_VARS
    panel = build_lags(panel, all_lag_vars)

    # ------------------------------------------------------------------
    # Step 5: Build log variables
    # ------------------------------------------------------------------
    print("Building log-transformed variables...")
    panel = build_log_vars(panel)

    # ------------------------------------------------------------------
    # Step 6: Build composite governance index and its lag
    # ------------------------------------------------------------------
    print("Building WGI composite...")
    panel = build_composite(panel)

    # ------------------------------------------------------------------
    # Step 7: Drop 2014 (used only for lag computation)
    # ------------------------------------------------------------------
    panel = panel[panel["year"].isin(OUTPUT_YEARS)].copy()

    # ------------------------------------------------------------------
    # Step 8: Enforce final column ordering (keep any extras at end)
    # ------------------------------------------------------------------
    ordered = [c for c in FINAL_COLS if c in panel.columns]
    extras  = [c for c in panel.columns if c not in FINAL_COLS]
    panel = panel[ordered + extras]

    # ------------------------------------------------------------------
    # Step 9: Sort and save
    # ------------------------------------------------------------------
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    panel.to_csv(OUTPUT_PANEL, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_PANEL}  ({len(panel):,} rows × {len(panel.columns)} cols)")

    # ------------------------------------------------------------------
    # Step 10: Data dictionary
    # ------------------------------------------------------------------
    dd = build_data_dictionary()
    dd.to_csv(OUTPUT_DICT, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_DICT}")

    # ------------------------------------------------------------------
    # Step 11: Validation report
    # ------------------------------------------------------------------
    validate_and_report(panel)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
