"""
===============================================================================
Bruegel-Equivalent CBAM Exposure Construction (Replication)
===============================================================================
Builds a CBAM exposure measure following Bruegel's methodology (Section 2.1,
footnote 3) for replication and comparison purposes.

Key differences from our preferred exposure (build_cbam_exposure.py):
    - Destination perimeter: EU27 + NOR + ISL + LIE + CHE (EU + EFTA)
      Our preferred: EU27 + NOR + ISL + LIE (EEA-EFTA), no CHE
    - Static perimeter: same countries for ALL years (no Brexit adjustment)
      Our preferred: GBR included 2014-2019, excluded 2020-2024
    - Two variants produced:
      (a) EU+EFTA without GBR (strict reading of "EU + EFTA")
      (b) EU+EFTA+GBR for all years (inclusive reading)

This script does NOT modify any existing output files.

Author: [placeholder]
Date:   2026-03-28
===============================================================================
"""

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
BACI_SRC_DIR = DATA_DIR / "baci_hs12_src"
LOOKUP_FILE  = DATA_DIR / "lookup" / "country_codes.csv"
OUTPUT_DIR   = DATA_DIR / "output" / "bruegel_replication"

BACI_FILENAME_PATTERN = "BACI_HS12_Y{year}_V202601.csv"

YEAR_START = 2014
YEAR_END   = 2024

# --- Bruegel importer perimeter: EU + EFTA = EU27 + NOR + ISL + LIE + CHE ---
# Note: CHE is included here (unlike our preferred measure) because Bruegel
# defines their perimeter as "EU + EFTA" which includes Switzerland.

EU27_ISO3 = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
]

EU_EFTA_ISO3     = EU27_ISO3 + ["NOR", "ISL", "LIE", "CHE"]       # 31 ISO3
EU_EFTA_GBR_ISO3 = EU_EFTA_ISO3 + ["GBR"]                          # 32 ISO3

# --- Exporter exclusions: same as main analysis ---
EXPORTER_EXCLUDE_ISO3 = [
    # EU27
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
    # EFTA/EEA exempt from CBAM
    "NOR", "ISL", "LIE", "CHE",
]

# --- CBAM HS6 product codes (identical to build_cbam_exposure.py) ---
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

# --- Bruegel benchmark values for comparison ---
BRUEGEL_BENCHMARKS = {
    "MNE": 0.242,   # Montenegro: 24.2%
    "MOZ": 0.150,   # Mozambique: 15%
    "BIH": 0.134,   # Bosnia: 13.4%
    "TUR": 0.040,   # Türkiye: ~4%
}


# ==============================================================================
# FUNCTIONS
# ==============================================================================

def load_country_codes() -> pd.DataFrame:
    df = pd.read_csv(LOOKUP_FILE, dtype={"country_code": int, "country_iso3": str},
                     encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.sort_values("country_code").drop_duplicates(subset="country_iso3", keep="first")
    df = df.reset_index(drop=True)
    assert df["country_iso3"].duplicated().sum() == 0, "Duplicate ISO3 codes in lookup"
    return df


def resolve_importer_codes(
    country_df: pd.DataFrame,
    iso3_list: list[str],
    label: str,
) -> list[int]:
    lookup = dict(zip(country_df["country_iso3"], country_df["country_code"]))
    codes, missing = [], []
    for iso3 in iso3_list:
        code = lookup.get(iso3)
        if code is None:
            missing.append(iso3)
        else:
            codes.append(int(code))
    if missing:
        print(f"  NOTE: {label} — ISO3 not in BACI lookup (trade under other codes): {missing}")
    return codes


def build_file_list() -> list[str]:
    files = []
    for year in range(YEAR_START, YEAR_END + 1):
        path = BACI_SRC_DIR / BACI_FILENAME_PATTERN.format(year=year)
        if path.exists():
            files.append(str(path))
        else:
            print(f"  WARNING: missing BACI file: {path}")
    assert len(files) > 0, "No BACI source files found"
    return files


def query_cbam_exports_static(
    con: duckdb.DuckDBPyConnection,
    file_list: list[str],
    importer_codes: list[int],
) -> pd.DataFrame:
    """CBAM exports to a STATIC importer perimeter (same codes every year)."""
    con.execute("CREATE OR REPLACE TEMP TABLE _imp_codes (code INTEGER)")
    con.executemany("INSERT INTO _imp_codes VALUES (?)", [[c] for c in importer_codes])

    con.execute("CREATE OR REPLACE TEMP TABLE _cbam_codes (code INTEGER)")
    con.executemany("INSERT INTO _cbam_codes VALUES (?)", [[c] for c in CBAM_HS6_CODES])

    file_str = ", ".join(f"'{f}'" for f in file_list)
    sql = f"""
        SELECT
            baci.t,
            baci.i,
            SUM(baci.v) AS cbam_exports
        FROM read_csv_auto([{file_str}], union_by_name=true) AS baci
        WHERE
            TRY_CAST(baci.j AS INTEGER) IN (SELECT code FROM _imp_codes)
            AND TRY_CAST(baci.k AS INTEGER) IN (SELECT code FROM _cbam_codes)
        GROUP BY baci.t, baci.i
    """
    return con.execute(sql).fetchdf()


def query_total_exports(
    con: duckdb.DuckDBPyConnection,
    file_list: list[str],
) -> pd.DataFrame:
    file_str = ", ".join(f"'{f}'" for f in file_list)
    sql = f"""
        SELECT t, i, SUM(v) AS total_exports
        FROM read_csv_auto([{file_str}], union_by_name=true)
        GROUP BY t, i
    """
    return con.execute(sql).fetchdf()


def build_exposure(
    cbam_df: pd.DataFrame,
    total_df: pd.DataFrame,
    country_df: pd.DataFrame,
    col_prefix: str = "bruegel",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build annual and average exposure DataFrames.
    Also computes one-year lagged exposure for Bruegel Table A2 replication.
    """
    panel = total_df.merge(cbam_df, on=["t", "i"], how="left")
    panel["cbam_exports"] = panel["cbam_exports"].fillna(0.0)

    panel["exposure_annual"] = np.where(
        panel["total_exports"] > 0,
        panel["cbam_exports"] / panel["total_exports"],
        0.0,
    )

    # Map to ISO3 and country name
    code_to_iso3 = dict(zip(country_df["country_code"], country_df["country_iso3"]))
    code_to_name = dict(zip(country_df["country_code"], country_df["country_name"]))
    panel["iso3"]         = panel["i"].map(code_to_iso3)
    panel["country_name"] = panel["i"].map(code_to_name)

    # Exclude EU27 + EFTA/EEA
    n_before = len(panel)
    panel = panel[~panel["iso3"].isin(EXPORTER_EXCLUDE_ISO3)].copy()
    print(f"      Excluded {n_before - len(panel):,} rows for EU27/EFTA exporters")

    # --- Annual output with lagged exposure ---
    panel = panel.sort_values(["i", "t"]).reset_index(drop=True)
    panel["exposure_lag1"] = panel.groupby("i")["exposure_annual"].shift(1)

    annual_cols = {
        "t": "t",
        "iso3": "iso3",
        "country_name": "country_name",
        "cbam_exports": f"{col_prefix}_cbam_exports_thousand_usd",
        "total_exports": "total_exports_thousand_usd",
        "exposure_annual": f"{col_prefix}_exposure_annual",
        "exposure_lag1": f"{col_prefix}_exposure_lag1",
    }
    annual_df = (
        panel[list(annual_cols.keys())]
        .rename(columns=annual_cols)
        .sort_values(["iso3", "t"])
        .reset_index(drop=True)
    )

    # --- Average output ---
    avg_df = (
        panel.groupby(["i", "iso3", "country_name"], dropna=False)
        .agg(
            exposure_avg=("exposure_annual", "mean"),
            cbam_exports_avg=("cbam_exports", "mean"),
            total_exports_avg=("total_exports", "mean"),
        )
        .reset_index()
        .drop(columns="i")
    )
    avg_df = avg_df.rename(columns={
        "exposure_avg": f"{col_prefix}_exposure_avg",
        "cbam_exports_avg": f"{col_prefix}_cbam_exports_avg_thousand_usd",
        "total_exports_avg": "total_exports_avg_thousand_usd",
    })
    avg_df = (
        avg_df.sort_values(f"{col_prefix}_exposure_avg", ascending=False)
        .reset_index(drop=True)
    )

    return annual_df, avg_df


def print_top20(avg_df: pd.DataFrame, exp_col: str, label: str) -> None:
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"Top 20 — {label}")
    print(sep)
    top20 = avg_df.head(20)
    print(f"  {'Rank':<5} {'ISO3':<6} {'Country':<30} {'Exposure':>10}")
    print(f"  {'----':<5} {'----':<6} {'-------':<30} {'--------':>10}")
    for rank, (_, row) in enumerate(top20.iterrows(), start=1):
        iso3 = row["iso3"] if pd.notna(row["iso3"]) else "???"
        name = row["country_name"] if pd.notna(row["country_name"]) else "???"
        pct  = row[exp_col] * 100
        print(f"  {rank:<5} {iso3:<6} {name:<30} {pct:>9.2f}%")


def print_bruegel_comparison(avg_df: pd.DataFrame, exp_col: str, label: str) -> None:
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"Bruegel benchmark comparison — {label}")
    print(sep)
    print(f"  {'Country':<15} {'Bruegel':>10} {'Ours':>10} {'Diff (pp)':>10}")
    print(f"  {'-------':<15} {'-------':>10} {'----':>10} {'---------':>10}")
    for iso3, bval in BRUEGEL_BENCHMARKS.items():
        row = avg_df.loc[avg_df["iso3"] == iso3]
        if len(row) == 0:
            print(f"  {iso3:<15} {bval*100:>9.1f}%      N/A       N/A")
            continue
        our_val = row[exp_col].values[0]
        diff = (our_val - bval) * 100
        print(f"  {iso3:<15} {bval*100:>9.1f}% {our_val*100:>9.2f}% {diff:>+9.2f}")


def print_preferred_comparison(
    bruegel_avg: pd.DataFrame,
    bruegel_col: str,
    label: str,
) -> None:
    """Compare Bruegel exposure against our preferred measure."""
    preferred_path = DATA_DIR / "output" / "cbam_exposure_avg.csv"
    if not preferred_path.exists():
        print(f"\n  Preferred exposure file not found at {preferred_path}, skipping comparison.")
        return

    preferred = pd.read_csv(preferred_path)
    merged = bruegel_avg[["iso3", bruegel_col]].merge(
        preferred[["iso3", "cbam_exposure_avg"]], on="iso3", how="inner"
    )
    merged["diff_pp"] = (merged[bruegel_col] - merged["cbam_exposure_avg"]) * 100
    merged = merged.sort_values(bruegel_col, ascending=False)

    sep = "-" * 70
    print(f"\n{sep}")
    print(f"Bruegel vs preferred exposure — {label} (top 20)")
    print(sep)
    print(f"  {'ISO3':<6} {'Bruegel':>10} {'Preferred':>10} {'Diff (pp)':>10}")
    print(f"  {'----':<6} {'-------':>10} {'---------':>10} {'---------':>10}")
    for _, row in merged.head(20).iterrows():
        b = row[bruegel_col] * 100
        p = row["cbam_exposure_avg"] * 100
        d = row["diff_pp"]
        print(f"  {row['iso3']:<6} {b:>9.2f}% {p:>9.2f}% {d:>+9.2f}")

    # Summary stats
    print(f"\n  All countries (N={len(merged)}):")
    print(f"    Mean diff:   {merged['diff_pp'].mean():+.3f} pp")
    print(f"    Median diff: {merged['diff_pp'].median():+.3f} pp")
    print(f"    Std diff:    {merged['diff_pp'].std():.3f} pp")
    print(f"    Correlation: {merged[bruegel_col].corr(merged['cbam_exposure_avg']):.4f}")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    overall_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load country codes ---
    print("[1/5] Loading country codes...")
    country_df = load_country_codes()

    eu_efta_codes     = resolve_importer_codes(country_df, EU_EFTA_ISO3, "EU+EFTA")
    eu_efta_gbr_codes = resolve_importer_codes(country_df, EU_EFTA_GBR_ISO3, "EU+EFTA+GBR")
    print(f"      EU+EFTA: {len(eu_efta_codes)} numeric codes resolved")
    print(f"      EU+EFTA+GBR: {len(eu_efta_gbr_codes)} numeric codes resolved")

    # --- Step 2: Build file list ---
    print("[2/5] Checking BACI source files...")
    file_list = build_file_list()
    print(f"      {len(file_list)} files for {YEAR_START}–{YEAR_END}")

    # --- Step 3: Run queries ---
    con = duckdb.connect()

    # Total exports (shared by both variants)
    print("[3/5] Querying total exports...")
    t0 = time.time()
    total_df = query_total_exports(con, file_list)
    print(f"      {len(total_df):,} (exporter, year) pairs  ({time.time() - t0:.1f}s)")

    # Variant (a): EU+EFTA without GBR
    print("      Querying CBAM exports to EU+EFTA (no GBR)...")
    t0 = time.time()
    cbam_a = query_cbam_exports_static(con, file_list, eu_efta_codes)
    print(f"      {len(cbam_a):,} pairs  ({time.time() - t0:.1f}s)")

    # Variant (b): EU+EFTA+GBR
    print("      Querying CBAM exports to EU+EFTA+GBR...")
    t0 = time.time()
    cbam_b = query_cbam_exports_static(con, file_list, eu_efta_gbr_codes)
    print(f"      {len(cbam_b):,} pairs  ({time.time() - t0:.1f}s)")

    con.close()

    # --- Step 4: Build exposure panels ---
    print("[4/5] Building exposure panels...")

    print("    Variant (a): EU+EFTA without GBR")
    annual_a, avg_a = build_exposure(cbam_a, total_df, country_df, col_prefix="bruegel")

    print("    Variant (b): EU+EFTA+GBR")
    annual_b, avg_b = build_exposure(cbam_b, total_df, country_df, col_prefix="bruegel_gbr")

    # --- Step 5: Save outputs ---
    print("[5/5] Saving outputs...")
    out_avg_a    = OUTPUT_DIR / "bruegel_exposure_avg.csv"
    out_annual_a = OUTPUT_DIR / "bruegel_exposure_annual.csv"
    out_avg_b    = OUTPUT_DIR / "bruegel_exposure_avg_incl_gbr.csv"
    out_annual_b = OUTPUT_DIR / "bruegel_exposure_annual_incl_gbr.csv"

    avg_a.to_csv(out_avg_a, index=False)
    annual_a.to_csv(out_annual_a, index=False)
    avg_b.to_csv(out_avg_b, index=False)
    annual_b.to_csv(out_annual_b, index=False)
    print(f"      {out_avg_a}")
    print(f"      {out_annual_a}")
    print(f"      {out_avg_b}")
    print(f"      {out_annual_b}")

    # --- Reports ---
    print_top20(avg_a, "bruegel_exposure_avg", "EU+EFTA without GBR (variant a)")
    print_top20(avg_b, "bruegel_gbr_exposure_avg", "EU+EFTA+GBR (variant b)")

    print_bruegel_comparison(avg_a, "bruegel_exposure_avg", "EU+EFTA without GBR")
    print_bruegel_comparison(avg_b, "bruegel_gbr_exposure_avg", "EU+EFTA+GBR")

    print_preferred_comparison(avg_a, "bruegel_exposure_avg", "EU+EFTA without GBR")
    print_preferred_comparison(avg_b, "bruegel_gbr_exposure_avg", "EU+EFTA+GBR")

    print(f"\nTotal elapsed: {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
