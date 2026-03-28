"""
===============================================================================
CBAM Trade Exposure Variable Construction
===============================================================================
Methodology:
    Constructs a country-level measure of economic exposure to the EU Carbon
    Border Adjustment Mechanism (CBAM) using bilateral trade data from BACI
    (Base pour l'Analyse du Commerce International), HS12 revision.

    For each exporting country i and year t, CBAM exposure is defined as:

        cbam_exposure_annual(i, t) =
            Σ v[i→CBAM_ZONE, k∈CBAM]  /  Σ v[i→*, k∈all]

    where v is the trade value in thousand USD, the numerator sums exports
    to the CBAM zone (EU27 + EEA-EFTA, plus GBR pre-2020) for CBAM-covered
    HS6 products, and the denominator sums all exports to all destinations
    in all products.

    The variable measures the share of a country's total export revenue that
    is potentially subject to CBAM pricing pressure, capturing both product
    coverage (only CBAM sectors) and market concentration (only CBAM zone
    destination).

    cbam_exposure_avg is the simple (unweighted) mean of cbam_exposure_annual
    over 2014–2024, used as the primary cross-sectional exposure measure in
    regressions testing CBAM's effect on domestic carbon pricing adoption.

CBAM sectors covered (EU CBAM Regulation 2023/956, Annex I):
    - Iron and steel (HS chapters 72–73 selected codes)
    - Aluminium (HS chapter 76)
    - Cement (HS chapter 25 selected codes)
    - Fertilisers (HS chapter 31)
    - Electricity (HS 271600)
    - Hydrogen (HS 280410)

Data sources:
    - BACI HS12 V202601 (CEPII): bilateral trade flows, 2014–2024
    - BACI country code lookup: CEPII country_codes.csv

Author: [placeholder]
Date:   2026-03-27
===============================================================================
"""

import os
import sys
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
BACI_SRC_DIR = DATA_DIR / "baci_hs12_src"
LOOKUP_FILE  = DATA_DIR / "lookup" / "country_codes.csv"
OUTPUT_DIR   = DATA_DIR / "output"

BACI_FILENAME_PATTERN = "BACI_HS12_Y{year}_V202601.csv"

OUTPUT_ANNUAL = OUTPUT_DIR / "cbam_exposure_annual.csv"
OUTPUT_AVG    = OUTPUT_DIR / "cbam_exposure_avg.csv"

# --- Year range ---
YEAR_START = 2014
YEAR_END   = 2024

# --- Destination importer perimeter (time-varying) ---
# The EU CBAM creates a unified carbon border across EU27 + EEA-EFTA states
# (NOR, ISL, LIE). These countries participate in the EU ETS via the EEA
# agreement and are inside the CBAM zone. Switzerland (CHE) has a separate
# bilateral ETS linking agreement and is NOT in the CBAM zone.
#
# Brexit: GBR left the EU on 31 January 2020. BACI records full-year flows;
# since GBR was outside the EU for 11 of 12 months in 2020, it is excluded
# from 2020 onward. 2019 is the last year GBR is in the importer perimeter.
#
# Importer perimeter by year:
#   2014–2019 : EU27 + EEA-EFTA + GBR  (31 countries)
#   2020–2024 : EU27 + EEA-EFTA only   (30 countries)

EU27_ISO3 = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
]
EEA_EFTA_ISO3 = ["NOR", "ISL", "LIE"]
CBAM_ZONE_ISO3     = EU27_ISO3 + EEA_EFTA_ISO3           # 30 countries
CBAM_ZONE_GBR_ISO3 = CBAM_ZONE_ISO3 + ["GBR"]            # 31 countries

# LIE (Liechtenstein) is not in the BACI country lookup — its trade is
# recorded under CHE in BACI. Expected resolved counts are therefore 29/30.
CBAM_ZONE_ISO3_COUNT_EXPECTED     = 29
CBAM_ZONE_GBR_ISO3_COUNT_EXPECTED = 30

# Last year GBR is included in the CBAM zone importer perimeter
GBR_LAST_YEAR = 2019

# --- CBAM HS6 product codes (as integers) ---
CBAM_HS6_CODES = [
    # Iron and steel — Chapter 72
    720110, 720120, 720150, 720210, 720211, 720219, 720220, 720230, 720241,
    720249, 720250, 720260, 720270, 720280, 720291, 720292, 720293, 720299,
    720310, 720390, 720410, 720421, 720429, 720430, 720450, 720510, 720521,
    720529, 720610, 720690, 720711, 720712, 720719, 720720, 720810, 720825,
    720826, 720827, 720836, 720837, 720838, 720839, 720840, 720851, 720852,
    720853, 720854, 720890, 720915, 720916, 720917, 720918, 720925, 720926,
    720927, 720928, 720990, 721011, 721012, 721020, 721030, 721041, 721049,
    721050, 721061, 721069, 721070, 721090, 721113, 721114, 721119, 721123,
    721129, 721190, 721210, 721220, 721230, 721240, 721250, 721260, 721310,
    721320, 721391, 721399, 721410, 721420, 721430, 721491, 721499, 721510,
    721550, 721590, 721610, 721621, 721622, 721631, 721632, 721633, 721640,
    721650, 721661, 721669, 721691, 721699, 721710, 721720, 721730, 721790,
    721810, 721891, 721899, 721911, 721912, 721913, 721914, 721921, 721922,
    721923, 721924, 721931, 721932, 721933, 721934, 721935, 721990, 722011,
    722012, 722020, 722090, 722100, 722211, 722219, 722220, 722230, 722240,
    722300, 722410, 722490, 722511, 722519, 722520, 722530, 722540, 722550,
    722591, 722592, 722599, 722611, 722619, 722620, 722691, 722692, 722693,
    722694, 722699, 722710, 722720, 722790, 722810, 722820, 722830, 722840,
    722850, 722860, 722870, 722880, 722920, 722990,
    # Iron and steel — Chapter 73 (selected)
    730110, 730120, 730210, 730240, 730290, 730300, 730410, 730420, 730431,
    730439, 730441, 730449, 730451, 730459, 730490, 730511, 730512, 730519,
    730520, 730531, 730539, 730590, 730610, 730621, 730629, 730630, 730640,
    730650, 730661, 730669, 730690,
    # Aluminium — Chapter 76
    760110, 760120, 760200, 760310, 760390, 760410, 760421, 760429, 760491,
    760499, 760511, 760519, 760521, 760529, 760611, 760612, 760691, 760692,
    760711, 760719, 760720, 760810, 760820, 760900, 761010, 761090, 761100,
    761290, 761300, 761410, 761490, 761510, 761520, 761610, 761620, 761630,
    761641, 761649, 761691, 761699,
    # Cement — Chapter 25 (selected)
    252310, 252321, 252329, 252330, 252390,
    # Fertilisers — Chapter 31
    310100, 310210, 310221, 310229, 310230, 310240, 310250, 310260, 310270,
    310280, 310290, 310310, 310390, 310410, 310420, 310430, 310490, 310510,
    310520, 310530, 310540, 310551, 310559, 310560, 310590,
    # Electricity
    271600,
    # Hydrogen
    280410,
]

# --- Countries excluded from the exporter panel ---
# EU27 members export TO the EU single market and are not subject to CBAM
# as third-country exporters. EFTA/EEA countries (NOR, ISL, LIE) and
# Switzerland (CHE) are exempt from CBAM under the regulation.
EXPORTER_EXCLUDE_ISO3 = [
    # EU27
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
    # EFTA/EEA exempt from CBAM
    "NOR", "ISL", "LIE", "CHE",
]

# --- Plausibility check countries ---
EXPECTED_TOP25_ISO3  = ["TUR", "UKR", "MNE", "BIH"]
UNEXPECTED_TOP25_ISO3 = ["DEU", "FRA", "ITA", "POL"]

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def load_country_codes() -> pd.DataFrame:
    """
    Load the BACI country codes lookup, deduplicate on iso3 keeping the row
    with the lowest numeric country_code (removes Belgium-Luxembourg composite),
    and assert no duplicate ISO3 codes remain.
    """
    df = pd.read_csv(LOOKUP_FILE, dtype={"country_code": int, "country_iso3": str}, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Sort so lowest country_code comes first, then keep first occurrence per iso3
    df = df.sort_values("country_code").drop_duplicates(subset="country_iso3", keep="first")
    df = df.reset_index(drop=True)

    dup_check = df["country_iso3"].duplicated().sum()
    assert dup_check == 0, (
        f"Duplicate ISO3 codes remain after deduplication: "
        f"{df.loc[df['country_iso3'].duplicated(keep=False), 'country_iso3'].unique().tolist()}"
    )

    return df


def get_cbam_zone_numeric_codes(
    country_df: pd.DataFrame,
) -> tuple[list[int], list[int]]:
    """
    Resolve CBAM zone ISO3 lists to BACI numeric country codes.
    Returns (cbam_zone_codes, cbam_zone_gbr_codes).
    Asserts expected counts are met for both lists; prints missing ISO3 codes.
    """
    lookup = dict(zip(country_df["country_iso3"], country_df["country_code"]))

    def _resolve(iso3_list: list[str], label: str, expected: int) -> list[int]:
        codes, missing = [], []
        for iso3 in iso3_list:
            code = lookup.get(iso3)
            if code is None:
                missing.append(iso3)
            else:
                codes.append(int(code))
        if missing:
            print(f"  WARNING: {label} — ISO3 codes not found in lookup: {missing}")
        assert len(codes) == expected, (
            f"Expected {expected} numeric codes for {label}, "
            f"found {len(codes)}. Missing ISO3: {missing}"
        )
        return codes

    cbam_zone_codes     = _resolve(CBAM_ZONE_ISO3,     "CBAM zone",       CBAM_ZONE_ISO3_COUNT_EXPECTED)
    cbam_zone_gbr_codes = _resolve(CBAM_ZONE_GBR_ISO3, "CBAM zone + GBR", CBAM_ZONE_GBR_ISO3_COUNT_EXPECTED)

    return cbam_zone_codes, cbam_zone_gbr_codes


def _build_file_list(verbose: bool = True) -> list[str]:
    """
    Build the list of expected BACI CSV file paths for YEAR_START..YEAR_END.
    Asserts each file exists; prints a warning for any missing year.
    Returns the list of existing file paths as strings.
    """
    files = []
    for year in range(YEAR_START, YEAR_END + 1):
        path = BACI_SRC_DIR / BACI_FILENAME_PATTERN.format(year=year)
        if path.exists():
            files.append(str(path))
        else:
            warnings.warn(f"  WARNING: expected BACI file not found: {path}")
            if verbose:
                print(f"  WARNING: expected BACI file not found: {path}")

    assert len(files) > 0, "No BACI source files found. Check BACI_SRC_DIR and file naming."
    return files


def query_baci_cbam_exports(
    con: duckdb.DuckDBPyConnection,
    file_list: list[str],
    cbam_zone_codes: list[int],
    cbam_zone_gbr_codes: list[int],
) -> pd.DataFrame:
    """
    Query A: For each (t, i), sum v where the importer j belongs to the
    year-appropriate destination perimeter AND k ∈ CBAM HS6 codes.

    Importer perimeter:
        years <= GBR_LAST_YEAR : CBAM zone + GBR (EU27 + EEA-EFTA + GBR)
        years >  GBR_LAST_YEAR : CBAM zone only  (EU27 + EEA-EFTA)

    Uses a _cbam_importer_codes(year, code) temp table so the join is a
    single equi-join on (t = year AND j_int = code) — no CASE expression
    needed in the main query.
    Returns DataFrame with columns: t, i, cbam_exports_eu.
    """
    # Build the time-varying importer table: CBAM zone + GBR for years
    # up to GBR_LAST_YEAR, CBAM zone only thereafter
    importer_rows: list[list[int]] = []
    for year in range(YEAR_START, YEAR_END + 1):
        codes_for_year = cbam_zone_gbr_codes if year <= GBR_LAST_YEAR else cbam_zone_codes
        for code in codes_for_year:
            importer_rows.append([year, code])

    con.execute(
        "CREATE OR REPLACE TEMP TABLE _cbam_importer_codes "
        "(year INTEGER, code INTEGER)"
    )
    con.executemany(
        "INSERT INTO _cbam_importer_codes VALUES (?, ?)",
        importer_rows,
    )

    # CBAM product codes temp table
    con.execute("CREATE OR REPLACE TEMP TABLE _cbam_codes (code INTEGER)")
    con.executemany(
        "INSERT INTO _cbam_codes VALUES (?)",
        [[c] for c in CBAM_HS6_CODES],
    )

    file_str = ", ".join(f"'{f}'" for f in file_list)

    sql = f"""
        SELECT
            baci.t,
            baci.i,
            SUM(baci.v) AS cbam_exports_eu
        FROM read_csv_auto([{file_str}], union_by_name=true) AS baci
        INNER JOIN _cbam_importer_codes AS imp
            ON baci.t = imp.year
            AND TRY_CAST(baci.j AS INTEGER) = imp.code
        WHERE
            TRY_CAST(baci.k AS INTEGER) IN (SELECT code FROM _cbam_codes)
        GROUP BY baci.t, baci.i
    """
    result = con.execute(sql).fetchdf()
    return result


def query_baci_total_exports(
    con: duckdb.DuckDBPyConnection,
    file_list: list[str],
) -> pd.DataFrame:
    """
    Query B: For each (t, i), sum all v regardless of destination or product.
    Returns DataFrame with columns: t, i, total_exports
    """
    file_str = ", ".join(f"'{f}'" for f in file_list)

    sql = f"""
        SELECT
            t,
            i,
            SUM(v) AS total_exports
        FROM read_csv_auto([{file_str}], union_by_name=true)
        GROUP BY t, i
    """
    result = con.execute(sql).fetchdf()
    return result


def build_exposure_panel(
    cbam_df: pd.DataFrame,
    total_df: pd.DataFrame,
    country_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge CBAM exports and total exports, compute exposure ratios, map ISO3
    and country names, and build annual and average-level output DataFrames.

    Returns:
        annual_df : panel with cbam_exposure_annual per (country, year)
        avg_df    : cross-section with cbam_exposure_avg per country
    """
    # Right join on total_exports so all (i, t) pairs are retained
    panel = total_df.merge(cbam_df, on=["t", "i"], how="left")
    panel["cbam_exports_eu"] = panel["cbam_exports_eu"].fillna(0.0)

    # Compute annual exposure share
    panel["cbam_exposure_annual"] = np.where(
        panel["total_exports"] > 0,
        panel["cbam_exports_eu"] / panel["total_exports"],
        0.0,
    )

    # Map numeric exporter codes to ISO3 and country name
    code_to_iso3  = dict(zip(country_df["country_code"], country_df["country_iso3"]))
    code_to_name  = dict(zip(country_df["country_code"], country_df["country_name"]))

    unmapped = set(panel["i"].unique()) - set(code_to_iso3.keys())
    if unmapped:
        print(f"  WARNING: {len(unmapped)} exporter code(s) not found in lookup table: "
              f"{sorted(unmapped)}")

    panel["iso3"]         = panel["i"].map(code_to_iso3)
    panel["country_name"] = panel["i"].map(code_to_name)

    # Exclude EU27 and EFTA/EEA-exempt countries from the exporter panel
    n_before = len(panel)
    panel = panel[~panel["iso3"].isin(EXPORTER_EXCLUDE_ISO3)].copy()
    n_excluded = n_before - len(panel)
    print(f"      Excluded {n_excluded:,} rows for EU27/EFTA exporters "
          f"({len(EXPORTER_EXCLUDE_ISO3)} ISO3 codes)")

    # --- Annual output ---
    annual_df = (
        panel[["t", "i", "iso3", "country_name",
               "cbam_exports_eu", "total_exports", "cbam_exposure_annual"]]
        .rename(columns={
            "cbam_exports_eu": "cbam_exports_eu_thousand_usd",
            "total_exports":   "total_exports_thousand_usd",
        })
        .sort_values(["iso3", "t"])
        .reset_index(drop=True)
    )

    # --- Average output ---
    # NOTE: cbam_exposure_avg is the simple mean of cbam_exposure_annual over
    # 2014–2024. The importer perimeter changes mid-period: EU27 + EEA-EFTA
    # + GBR for 2014–2019 and EU27 + EEA-EFTA only for 2020–2024. The average
    # therefore spans two slightly different destination markets. This is
    # intentional — it captures the full pre/post-Brexit exposure trajectory —
    # but should be noted when interpreting cross-sectional regressions.
    avg_df = (
        panel.groupby(["i", "iso3", "country_name"], dropna=False)
        .agg(
            cbam_exposure_avg             =("cbam_exposure_annual", "mean"),
            cbam_exports_eu_avg_thousand_usd=("cbam_exports_eu",    "mean"),
            total_exports_avg_thousand_usd  =("total_exports",      "mean"),
        )
        .reset_index()
        .drop(columns="i")
        .sort_values("cbam_exposure_avg", ascending=False)
        .reset_index(drop=True)
    )

    return annual_df, avg_df


def validate_and_report(
    annual_df: pd.DataFrame,
    avg_df: pd.DataFrame,
) -> None:
    """
    Print validation statistics and plausibility checks.
    """
    sep = "-" * 60

    print(f"\n{sep}")
    print("VALIDATION REPORT")
    print(sep)

    # 1. Total observations
    n_obs = len(annual_df)
    print(f"\n1. Total country-year observations:  {n_obs:,}")

    # 2. Unique exporters
    n_countries = annual_df["iso3"].nunique()
    print(f"2. Unique exporter countries (ISO3): {n_countries:,}")

    # 3. Years and obs per year
    year_counts = (
        annual_df.groupby("t").size()
        .reset_index(name="n_obs")
        .sort_values("t")
    )
    print(f"\n3. Observations per year:")
    for _, row in year_counts.iterrows():
        print(f"     {int(row['t'])}: {int(row['n_obs']):>5,} country-year obs")

    # 4. Top 20 countries by cbam_exposure_avg
    print(f"\n4. Top 20 countries by cbam_exposure_avg (2014–2024 mean):")
    top20 = avg_df.head(20).copy()
    print(f"   {'Rank':<5} {'ISO3':<6} {'Country':<35} {'cbam_exposure_avg':>18}")
    print(f"   {'----':<5} {'----':<6} {'-------':<35} {'----------------':>18}")
    for rank, (_, row) in enumerate(top20.iterrows(), start=1):
        iso3 = row["iso3"] if pd.notna(row["iso3"]) else "(unknown)"
        name = row["country_name"] if pd.notna(row["country_name"]) else "(unknown)"
        pct  = row["cbam_exposure_avg"] * 100
        print(f"   {rank:<5} {iso3:<6} {name:<35} {pct:>17.2f}%")

    # 5. Plausibility check: expected high-exposure countries in top 25
    top25_iso3 = set(avg_df.head(25)["iso3"].dropna().tolist())
    found   = [c for c in EXPECTED_TOP25_ISO3 if c in top25_iso3]
    missing = [c for c in EXPECTED_TOP25_ISO3 if c not in top25_iso3]
    if not missing:
        print(f"\n5. Plausibility check (TUR/UKR/MNE/BIH in top 25): PASS")
        print(f"     All found: {found}")
    else:
        print(f"\n5. Plausibility check (TUR/UKR/MNE/BIH in top 25): FAIL")
        print(f"     Found:   {found}")
        print(f"     Missing: {missing}")

    # 6. Sanity check: EU27 and EFTA/EEA-exempt countries must be absent from
    #    the entire output (not just top 25) due to the exclusion filter
    all_iso3_in_output = set(avg_df["iso3"].dropna().tolist())
    excluded_present = [c for c in EXPORTER_EXCLUDE_ISO3 if c in all_iso3_in_output]
    if not excluded_present:
        print(f"\n6. Sanity check (EU27/EFTA absent from full output): PASS")
        print(f"     No excluded exporter (EU27 or EFTA/EEA-exempt) appears in output")
    else:
        print(f"\n6. Sanity check (EU27/EFTA absent from full output): FAIL")
        print(f"     Excluded exporters found in output: {excluded_present}")

    # 7. True-zero countries
    n_zero = (avg_df["cbam_exposure_avg"] == 0.0).sum()
    print(f"\n7. Countries with cbam_exposure_avg = 0 (no CBAM exports to CBAM zone): {n_zero:,}")

    # 8. Distribution summary
    vals = avg_df["cbam_exposure_avg"].dropna()
    percentiles = np.percentile(vals, [0, 25, 50, 75, 90, 100])
    labels = ["min", "p25", "median", "p75", "p90", "max"]
    print(f"\n8. Distribution of cbam_exposure_avg:")
    for label, pct_val in zip(labels, percentiles):
        print(f"     {label:<8}: {pct_val * 100:.4f}%")

    print(f"\n{sep}")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    overall_start = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load country codes ---
    t0 = time.time()
    print("[1/5] Loading country codes lookup...")
    country_df = load_country_codes()
    cbam_zone_codes, cbam_zone_gbr_codes = get_cbam_zone_numeric_codes(country_df)
    print(f"      {len(country_df):,} countries loaded, "
          f"{len(cbam_zone_codes)} CBAM zone codes + GBR resolved "
          f"(GBR included through {GBR_LAST_YEAR})  "
          f"({time.time() - t0:.1f}s)")

    # --- Step 2: Build and validate file list ---
    t0 = time.time()
    print("[2/5] Checking BACI source files...")
    file_list = _build_file_list(verbose=True)
    print(f"      {len(file_list)} files confirmed present for "
          f"{YEAR_START}–{YEAR_END}  ({time.time() - t0:.1f}s)")

    # --- Step 3: Run DuckDB queries ---
    t0 = time.time()
    print("[3/5] Running DuckDB queries across all BACI files...")
    con = duckdb.connect()

    print("      Query A: CBAM exports to CBAM zone...")
    qa_start = time.time()
    cbam_df = query_baci_cbam_exports(con, file_list, cbam_zone_codes, cbam_zone_gbr_codes)
    print(f"        → {len(cbam_df):,} (exporter, year) pairs with CBAM exports  "
          f"({time.time() - qa_start:.1f}s)")

    print("      Query B: Total exports...")
    qb_start = time.time()
    total_df = query_baci_total_exports(con, file_list)
    print(f"        → {len(total_df):,} (exporter, year) pairs total  "
          f"({time.time() - qb_start:.1f}s)")

    con.close()
    print(f"      Queries complete  ({time.time() - t0:.1f}s)")

    # --- Step 4: Build exposure panel ---
    t0 = time.time()
    print("[4/5] Building exposure panel and computing averages...")
    annual_df, avg_df = build_exposure_panel(cbam_df, total_df, country_df)
    print(f"      Annual panel: {len(annual_df):,} rows  |  "
          f"Average cross-section: {len(avg_df):,} rows  "
          f"({time.time() - t0:.1f}s)")

    # --- Comparison with previous output (if exists) ---
    if OUTPUT_AVG.exists():
        print("\n--- Old vs New exposure comparison (top 20) ---")
        old_avg = pd.read_csv(OUTPUT_AVG)
        old_top = old_avg[["iso3", "cbam_exposure_avg"]].rename(
            columns={"cbam_exposure_avg": "old"}
        )
        new_top = avg_df[["iso3", "cbam_exposure_avg"]].rename(
            columns={"cbam_exposure_avg": "new"}
        )
        compare = new_top.merge(old_top, on="iso3", how="left")
        compare["diff_pp"] = (compare["new"] - compare["old"]) * 100
        compare = compare.sort_values("new", ascending=False).head(20)
        print(f"    {'ISO3':<6} {'Old':>10} {'New':>10} {'Diff (pp)':>10}")
        print(f"    {'----':<6} {'---':>10} {'---':>10} {'---------':>10}")
        for _, r in compare.iterrows():
            old_val = f"{r['old'] * 100:.2f}%" if pd.notna(r["old"]) else "N/A"
            new_val = f"{r['new'] * 100:.2f}%"
            diff_val = f"{r['diff_pp']:+.2f}" if pd.notna(r["diff_pp"]) else "N/A"
            print(f"    {r['iso3']:<6} {old_val:>10} {new_val:>10} {diff_val:>10}")
        print()

    # --- Step 5: Save outputs ---
    t0 = time.time()
    print("[5/5] Saving output files...")
    annual_df.to_csv(OUTPUT_ANNUAL, index=False)
    avg_df.to_csv(OUTPUT_AVG, index=False)
    print(f"      {OUTPUT_ANNUAL}")
    print(f"      {OUTPUT_AVG}")
    print(f"      ({time.time() - t0:.1f}s)")

    # --- Validation report ---
    validate_and_report(annual_df, avg_df)

    print(f"\nTotal elapsed time: {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
