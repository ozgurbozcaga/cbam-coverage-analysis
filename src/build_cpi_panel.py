"""
build_cpi_panel.py

Constructs the analytical CPI panel for the World Bank research project on
EU CBAM spillovers and carbon pricing sectoral coverage.

Unit of observation: one row per CPI per year (2016-2026).

Sources
-------
  data/lookup/cpi_list.xlsx
  data/lookup/cpi_price_usd.xlsx
  data/lookup/sectoral_coverage.xlsx
  data/output/cbam_exposure_annual.csv
  data/output/cbam_exposure_avg.csv

Outputs
-------
  data/output/cpi_panel.csv
  data/output/cpi_panel_data_dictionary.csv

Run from project root:
  python3 src/build_cpi_panel.py
"""

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ===========================================================================
# CONFIGURATION
# ===========================================================================

PANEL_YEARS = list(range(2016, 2027))

# Nine canonical sector tokens and their output column names
SECTOR_COL_MAP: dict[str, str] = {
    "Power":          "covers_power",
    "Industry":       "covers_industry",
    "Road transport": "covers_road_transport",
    "Aviation":       "covers_aviation",
    "Shipping":       "covers_shipping",
    "Buildings":      "covers_buildings",
    "Waste":          "covers_waste",
    "Agriculture":    "covers_agriculture",
    "Forestry":       "covers_forestry",
}

# S&T editions are the canonical annual observations.
# For years with two editions (2016-2020), only the S&T edition is used.
EDITION_TO_YEAR: dict[str, int] = {
    "S&T16": 2016,
    "S&T17": 2017,
    "S&T18": 2018,
    "S&T19": 2019,
    "S&T20": 2020,
    "S&T21": 2021,
    "S&T22": 2022,
    "S&T23": 2023,
    "S&T24": 2024,
    "S&T25": 2025,
    "S&T26": 2026,
}

# EEA/EFTA entries that participate in EU ETS but are CBAM-exempt exporters.
# Their cbam_exposure values will be null — expected and correct.
EEA_ETS_EXTENSION_IDS: list[str] = ["ETS_EU_IS", "ETS_EU_LI", "ETS_EU_NO"]

# Subnational and EEA jurisdiction -> ISO3 mapping.
# Checked first (before country_codes.csv) in resolve_iso3().
JURISDICTION_TO_ISO3: dict[str, str] = {
    # Canadian provinces
    "Alberta": "CAN",
    "British Columbia": "CAN",
    "Ontario": "CAN",
    "Quebec": "CAN",
    "Saskatchewan": "CAN",
    "Nova Scotia": "CAN",
    "New Brunswick": "CAN",
    "Newfoundland and Labrador": "CAN",
    "Northwest Territories": "CAN",
    "Prince Edward Island": "CAN",
    # US states
    "California": "USA",
    "Washington": "USA",
    "Massachusetts": "USA",
    "Virginia": "USA",
    "Oregon": "USA",
    "Colorado": "USA",
    "Minneapolis": "USA",
    # Chinese pilot cities/provinces
    "Beijing": "CHN",
    "Shanghai": "CHN",
    "Shenzhen": "CHN",
    "Guangdong": "CHN",
    "Tianjin": "CHN",
    "Hubei": "CHN",
    "Chongqing": "CHN",
    "Fujian": "CHN",
    # Japanese prefectures
    "Tokyo": "JPN",
    "Saitama": "JPN",
    # Mexican states
    "Baja California": "MEX",
    "Tamaulipas": "MEX",
    "Zacatecas": "MEX",
    "Nuevo Leon": "MEX",
    "Queretaro": "MEX",
    "Southern Mexico": "MEX",
    "Yucatan": "MEX",
    "Guanajuato": "MEX",
    "Durango": "MEX",
    "Colima": "MEX",
    "Mexico City": "MEX",
    "Morelos": "MEX",
    "San Luis Potosi": "MEX",
    # EEA/EFTA EU ETS participants
    "Iceland": "ISL",       # ETS_EU_IS
    "Liechtenstein": "LIE", # ETS_EU_LI
    "Norway": "NOR",        # ETS_EU_NO
}

# Manual overrides for country names in CPI files that do not match
# country_codes.csv exactly.  None means no single ISO3 applies.
COUNTRY_NAME_OVERRIDES: dict[str, str | None] = {
    "Bolivia": "BOL",
    "Bosnia and Herzegovina": "BIH",
    "Côte d'Ivoire": "CIV",
    "Dominican Republic": "DOM",
    "Korea, Rep.": "KOR",
    "Moldova": "MDA",
    "Russia": "RUS",
    "Taiwan, China": "TWN",
    "Turkey": "TUR",
    "United States": "USA",
    "United states": "USA",
    "European Union": None,   # No single ISO3; null CBAM exposure is correct
}

# Fallback income group for CPIs absent from cpi_price_usd.xlsx.
# Matched on the 'country' column after the price-file merge.
COUNTRY_INCOME_GROUP: dict[str, str] = {
    'Albania': 'Upper middle income',
    'Argentina': 'Upper middle income',
    'Australia': 'High income',
    'Bahrain': 'High income',
    'Bolivia': 'Lower middle income',
    'Bosnia and Herzegovina': 'Upper middle income',
    'Botswana': 'Upper middle income',
    'Brazil': 'Upper middle income',
    'Brunei Darussalam': 'High income',
    'Canada': 'High income',
    'Chile': 'High income',
    'China': 'Upper middle income',
    'Colombia': 'Upper middle income',
    'Côte d Ivoire': 'Lower middle income',
    'Denmark': 'High income',
    'Dominican Republic': 'Upper middle income',
    'Estonia': 'High income',
    'European Union': 'High income',
    'Finland': 'High income',
    'France': 'High income',
    'Gabon': 'Upper middle income',
    'Georgia': 'Upper middle income',
    'Germany': 'High income',
    'Guyana': 'Upper middle income',
    'Hungary': 'High income',
    'Iceland': 'High income',
    'Indonesia': 'Lower middle income',
    'Ireland': 'High income',
    'Japan': 'High income',
    'Kenya': 'Lower middle income',
    'Korea, Rep.': 'High income',
    'Latvia': 'High income',
    'Liechtenstein': 'High income',
    'Luxembourg': 'High income',
    'Malaysia': 'Upper middle income',
    'Mauritania': 'Lower middle income',
    'Mexico': 'Upper middle income',
    'Moldova': 'Lower middle income',
    'Morocco': 'Lower middle income',
    'Netherlands': 'High income',
    'New Zealand': 'High income',
    'Nigeria': 'Lower middle income',
    'North Macedonia': 'Upper middle income',
    'Norway': 'High income',
    'Pakistan': 'Lower middle income',
    'Paraguay': 'Upper middle income',
    'Philippines': 'Lower middle income',
    'Poland': 'High income',
    'Portugal': 'High income',
    'Russia': 'Upper middle income',
    'Senegal': 'Lower middle income',
    'Serbia': 'Upper middle income',
    'Slovenia': 'High income',
    'Spain': 'High income',
    'Sweden': 'High income',
    'Switzerland': 'High income',
    'Taiwan, China': 'High income',
    'Thailand': 'Upper middle income',
    'Turkey': 'Upper middle income',
    'Ukraine': 'Lower middle income',
    'United States': 'High income',
    'Uruguay': 'High income',
    'Viet Nam': 'Lower middle income',
}

# ---------------------------------------------------------------------------
# Data dictionary entries — (variable_name, description, source, dtype, notes)
# ---------------------------------------------------------------------------
_DATA_DICTIONARY_ROWS: list[tuple] = [
    ("cpi_id",
     "Unique CPI identifier",
     "cpi_list.xlsx / sectoral_coverage.xlsx",
     "str", "Primary key component"),
    ("cpi_name",
     "Name of the carbon pricing instrument",
     "cpi_list.xlsx", "str", ""),
    ("year",
     "Publication year of the observation",
     "Derived", "int", "Panel range 2016-2026"),
    ("instrument_type",
     "Type of instrument: ETS, Carbon tax, or Undefined",
     "cpi_list.xlsx", "str", ""),
    ("country",
     "Country or region name",
     "cpi_list.xlsx", "str", ""),
    ("jurisdiction",
     "Jurisdiction covered (may be subnational or multi-country)",
     "cpi_list.xlsx", "str", ""),
    ("scope",
     "Geographic scope: National, Subnational, or Regional",
     "cpi_list.xlsx", "str", "Normalised to title case"),
    ("start_year",
     "Year the CPI became implemented (Start date)",
     "cpi_list.xlsx", "Int64", "Nullable integer"),
    ("end_year",
     "Year the CPI was abolished or ended (End date)",
     "cpi_list.xlsx", "Int64",
     "Nullable integer; NaN = still active or no end date recorded"),
    ("is_eea_ets_extension",
     "Binary: 1 for ETS_EU_IS, ETS_EU_LI, ETS_EU_NO (EFTA participation in EU ETS)",
     "cpi_list.xlsx", "int",
     "CBAM exposure is null for these; expected and correct"),
    ("status_implemented",
     "Binary: 1 if CPI is in implemented status in this year",
     "Derived from cpi_list.xlsx date columns", "int",
     "Uses date-range derivation; takes priority over all other statuses"),
    ("status_under_development",
     "Binary: 1 if CPI is under development in this year",
     "Derived from cpi_list.xlsx date columns", "int",
     "Priority 2 in status hierarchy; mutually exclusive with implemented"),
    ("status_under_consideration",
     "Binary: 1 if CPI is under consideration in this year",
     "Derived from cpi_list.xlsx date columns", "int",
     "Priority 3 in status hierarchy; mutually exclusive with higher statuses"),
    ("status_active",
     "Binary: 1 if any of implemented/under_development/under_consideration = 1",
     "Derived", "int", ""),
    ("status_abolished",
     "Binary: 1 if end_year is not NaN and year > end_year",
     "Derived", "int", ""),
    ("status_level",
     "Ordinal status: 3=Implemented, 2=Under development, 1=Under consideration, 0=None/Abolished",
     "Derived", "int", "Mutually exclusive; uses priority hierarchy"),
    ("covers_power",
     "Binary: 1 if CPI covers the power sector",
     "sectoral_coverage.xlsx", "int",
     "0 if CPI not present in S&T edition for that year"),
    ("covers_industry",
     "Binary: 1 if CPI covers the industry sector.  PRIMARY OUTCOME VARIABLE",
     "sectoral_coverage.xlsx", "int", ""),
    ("covers_road_transport",
     "Binary: 1 if CPI covers the road transport sector",
     "sectoral_coverage.xlsx", "int", ""),
    ("covers_aviation",
     "Binary: 1 if CPI covers the aviation sector",
     "sectoral_coverage.xlsx", "int", ""),
    ("covers_shipping",
     "Binary: 1 if CPI covers the shipping sector",
     "sectoral_coverage.xlsx", "int", ""),
    ("covers_buildings",
     "Binary: 1 if CPI covers the buildings sector",
     "sectoral_coverage.xlsx", "int", ""),
    ("covers_waste",
     "Binary: 1 if CPI covers the waste sector",
     "sectoral_coverage.xlsx", "int", ""),
    ("covers_agriculture",
     "Binary: 1 if CPI covers the agriculture sector",
     "sectoral_coverage.xlsx", "int", ""),
    ("covers_forestry",
     "Binary: 1 if CPI covers the forestry sector",
     "sectoral_coverage.xlsx", "int", ""),
    ("n_sectors_covered",
     "Integer count of sectors covered (0-9)",
     "Derived from sectoral_coverage.xlsx", "int", ""),
    ("sector_string",
     "Original comma-separated sector string (audit trail)",
     "sectoral_coverage.xlsx", "str",
     "NaN if CPI not present in S&T edition for that year"),
    ("coverage_active",
     "Binary: 1 if CPI appears in sectoral coverage data for this year",
     "Derived", "int",
     "0 = not present in S&T edition; sector binaries are 0 but row is retained"),
    ("price_usd",
     "Annual carbon price in USD (converted from LCU at April 1 exchange rate)",
     "cpi_price_usd.xlsx", "float",
     "NaN if no price data available for that CPI-year"),
    ("log_price_usd",
     "log(price_usd); NaN where price_usd <= 0 or missing",
     "Derived", "float", "NaN propagates from price_usd"),
    ("income_group",
     "World Bank income group of the country",
     "cpi_price_usd.xlsx + COUNTRY_INCOME_GROUP fallback", "str",
     "Filled from price file first; COUNTRY_INCOME_GROUP dict used as fallback"),
    ("region",
     "World Bank region",
     "cpi_price_usd.xlsx", "str", ""),
    ("currency",
     "Original currency code before USD conversion",
     "cpi_price_usd.xlsx", "str", ""),
    ("post_2019",
     "Binary: 1 if year > 2019 (Green Deal announced July 2019; "
     "April 1 data reference predates announcement)",
     "Derived", "int", ""),
    ("post_2021",
     "Binary: 1 if year > 2021 (CBAM regulation proposed July 2021; "
     "April 1 data reference predates proposal)",
     "Derived", "int", ""),
    ("post_2023",
     "Binary: 1 if year > 2023 (CBAM entered into force October 2023; "
     "April 1 data reference predates entry into force)",
     "Derived", "int", ""),
    ("post_ge2019",
     "Sensitivity variant: 1 if year >= 2019 (inclusive milestone)",
     "Derived", "int", ""),
    ("post_ge2021",
     "Sensitivity variant: 1 if year >= 2021 (inclusive milestone)",
     "Derived", "int", ""),
    ("post_ge2023",
     "Sensitivity variant: 1 if year >= 2023 (inclusive milestone)",
     "Derived", "int", ""),
    ("first_year_industry",
     "First year this CPI covered industry sector within 2016-2026 panel. "
     "0 = never-treated.",
     "Derived", "int",
     "If covers_industry=1 in 2016, set to 2016 (always-treated at panel start)"),
    ("never_treated_industry",
     "Binary: 1 if covers_industry=0 in ALL years 2016-2026",
     "Derived", "int",
     "Clean control group for Callaway & Sant'Anna DiD estimator"),
    ("always_treated_industry",
     "Binary: 1 if first_year_industry == 2016 (had industry coverage in panel start year)",
     "Derived", "int",
     "Treated from the start of the panel; excluded from C&S switcher analysis"),
    ("switcher_type",
     "Categorical: always_treated / never_treated / new_cpi_with_industry / genuine_expansion",
     "Derived", "str",
     "new_cpi_with_industry = first covered industry in same year it entered coverage data; "
     "genuine_expansion = was active without industry before adding industry coverage"),
    ("first_year_implemented",
     "First year this CPI reached implemented status within 2016-2026. 0 = never.",
     "Derived", "int", ""),
    ("never_implemented",
     "Binary: 1 if CPI never reached implemented status in the 2016-2026 window",
     "Derived", "int", ""),
    ("cbam_exposure_annual",
     "Annual CBAM exposure: share of exports going to EU in CBAM-covered sectors",
     "cbam_exposure_annual.csv", "float",
     "Merged on iso3 + year; NaN for unmatched ISO3 or years outside 2014-2024"),
    ("cbam_exposure_avg",
     "Average CBAM exposure 2014-2024 (static regressor)",
     "cbam_exposure_avg.csv", "float",
     "Merged on iso3 only; NaN for unmatched ISO3"),
    ("cbam_exports_eu_thousand_usd",
     "Annual CBAM-covered exports to EU in thousand USD (numerator)",
     "cbam_exposure_annual.csv", "float", ""),
    ("total_exports_thousand_usd",
     "Annual total exports in thousand USD (denominator)",
     "cbam_exposure_annual.csv", "float", ""),
    ("iso3",
     "ISO3 country code resolved from jurisdiction/country columns",
     "Derived via country_codes.csv and manual overrides", "str",
     "None for unresolved jurisdictions (e.g. European Union); "
     "null CBAM exposure expected for those rows"),
]


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_cpi_list() -> pd.DataFrame:
    """
    Load cpi_list.xlsx.

    Normalises scope to title case.
    Converts all date columns (Start date, End date, etc.) to nullable Int64.

    Returns 162-row DataFrame with columns:
      cpi_id, cpi_name, instrument_type, jurisdiction, country, scope,
      start_year_raw, end_year_raw, uc_start_raw, uc_end_raw,
      ud_start_raw, ud_end_raw, status_static
    """
    df = pd.read_excel(
        "data/lookup/cpi_list.xlsx",
        engine="openpyxl",
        sheet_name="Sheet1",
    )
    df = df.rename(columns={
        "Unique identifier":          "cpi_id",
        "CPI Name":                   "cpi_name",
        "Type":                       "instrument_type",
        "Jurisdiction covered":       "jurisdiction",
        "Country/region":             "country",
        "Scope":                      "scope",
        "Start date":                 "start_year_raw",
        "End date":                   "end_year_raw",
        "Under consideration start":  "uc_start_raw",
        "Under consideration end":    "uc_end_raw",
        "Under development start":    "ud_start_raw",
        "Under development end":      "ud_end_raw",
        "Status":                     "status_static",
    })

    # Normalise scope to title case ('subnational' -> 'Subnational')
    df["scope"] = df["scope"].str.strip().str.title()

    # Convert year floats to nullable Int64
    for col in ("start_year_raw", "end_year_raw",
                "uc_start_raw", "uc_end_raw",
                "ud_start_raw", "ud_end_raw"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def load_price_panel() -> pd.DataFrame:
    """
    Load cpi_price_usd.xlsx and reshape from wide to long for years 2016-2026.

    Returns DataFrame with columns:
      cpi_id, year, price_usd, income_group, region, currency
    """
    df = pd.read_excel(
        "data/lookup/cpi_price_usd.xlsx",
        engine="openpyxl",
        sheet_name="Sheet1",
    )
    df = df.rename(columns={
        "Unique ID":            "cpi_id",
        "Country income group": "income_group",
        "Region":               "region",
        "Currency":             "currency",
    })

    year_cols = [c for c in df.columns if isinstance(c, int) and 2016 <= c <= 2026]
    id_cols   = ["cpi_id", "income_group", "region", "currency"]

    long = df[id_cols + year_cols].melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="year",
        value_name="price_usd",
    )
    long["year"] = long["year"].astype(int)
    return long.reset_index(drop=True)


def load_sector_panel() -> pd.DataFrame:
    """
    Load sectoral_coverage.xlsx using only the S&T (State & Trends) editions
    as canonical annual observations.

    Row 0 of the sheet is the publication-year header; rows 1+ are CPI data.
    CPI IDs are normalised (spaces -> underscores) to match cpi_list.xlsx.

    Returns long DataFrame with columns:
      cpi_id, year, sector_string, coverage_active,
      covers_power, covers_industry, ..., n_sectors_covered
    """
    raw = pd.read_excel(
        "data/lookup/sectoral_coverage.xlsx",
        engine="openpyxl",
        sheet_name="Sheet1",
        header=None,
    )

    # Row 0 is the header row (contains column names such as 'S&T16')
    col_names = raw.iloc[0].tolist()
    data = raw.iloc[1:].copy()
    data.columns = col_names
    data = data.rename(columns={"Report ID": "cpi_id"})
    data = data.dropna(subset=["cpi_id"]).copy()

    # Normalise CPI IDs: replace spaces with underscores to align with cpi_list
    data["cpi_id"] = data["cpi_id"].astype(str).str.strip().str.replace(" ", "_")

    # Drop the publication-year metadata row (Report ID = "Publication year")
    data = data[data["cpi_id"] != "Publication_year"].copy()

    # Keep only S&T edition columns
    st_cols = [col for col in col_names if col in EDITION_TO_YEAR]
    data = data[["cpi_id"] + st_cols].copy()

    # Melt to long format
    long = data.melt(
        id_vars="cpi_id",
        value_vars=st_cols,
        var_name="edition",
        value_name="sector_string",
    )
    long["year"] = long["edition"].map(EDITION_TO_YEAR).astype(int)
    long = long.drop(columns="edition").reset_index(drop=True)

    # Parse sector strings into binary columns
    parsed = long["sector_string"].apply(parse_sector_string)
    sector_df = pd.DataFrame(parsed.tolist(), index=long.index)
    long = pd.concat([long, sector_df], axis=1)

    long["coverage_active"] = (~long["sector_string"].isna()).astype(int)

    # Fill sector binaries with 0 where sector_string is NaN
    for col in SECTOR_COL_MAP.values():
        long[col] = long[col].fillna(0).astype(int)

    long["n_sectors_covered"] = long[list(SECTOR_COL_MAP.values())].sum(axis=1).astype(int)

    return long.reset_index(drop=True)


def parse_sector_string(s) -> dict[str, int]:
    """
    Parse a comma-separated sector string into a dict of 9 binary flags.

    Parameters
    ----------
    s : str or NaN
        e.g. 'Power, Industry, Aviation'

    Returns
    -------
    dict keyed by covers_* column names, values 0 or 1.
    """
    result = {col: 0 for col in SECTOR_COL_MAP.values()}
    if pd.isna(s) or not isinstance(s, str) or s.strip() == "":
        return result
    tokens = {t.strip() for t in s.split(",")}
    for sector, col in SECTOR_COL_MAP.items():
        if sector in tokens:
            result[col] = 1
    return result


# ===========================================================================
# ISO3 RESOLUTION
# ===========================================================================

_country_codes: pd.DataFrame | None = None


def _get_country_codes() -> pd.DataFrame:
    global _country_codes
    if _country_codes is None:
        _country_codes = pd.read_csv(
            "data/lookup/country_codes.csv", encoding="utf-8-sig"
        )
    return _country_codes


def resolve_iso3(jurisdiction: str, country: str, cpi_id: str) -> str | None:
    """
    Resolve the ISO3 code for a CPI using a three-step lookup.

    Step 1: Check JURISDICTION_TO_ISO3 using the 'jurisdiction' column.
    Step 2: Check COUNTRY_NAME_OVERRIDES using the 'country' column.
    Step 3: Match 'country' against country_name in country_codes.csv.
    Step 4: Log a warning and return None if still unresolved.

    Missing ISO3 -> null CBAM exposure, which is acceptable for some CPIs.
    """
    # Step 1: jurisdiction dictionary (handles subnational and EEA entries)
    if pd.notna(jurisdiction) and str(jurisdiction).strip() in JURISDICTION_TO_ISO3:
        return JURISDICTION_TO_ISO3[str(jurisdiction).strip()]

    if pd.notna(country):
        country_str = str(country).strip()

        # Step 2: manual name overrides
        if country_str in COUNTRY_NAME_OVERRIDES:
            return COUNTRY_NAME_OVERRIDES[country_str]

        # Step 3: country_codes.csv exact match
        cc = _get_country_codes()
        match = cc.loc[cc["country_name"] == country_str, "country_iso3"]
        if not match.empty:
            return match.iloc[0]

    # Step 4: unresolved
    print(f"  WARNING: unresolved ISO3 for cpi_id='{cpi_id}' "
          f"(jurisdiction={jurisdiction!r}, country={country!r})")
    return None


# ===========================================================================
# DERIVED VARIABLES
# ===========================================================================

def derive_status_variables(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-varying status binary flags and status_level using the date
    columns from cpi_list.xlsx.

    Priority hierarchy (mutually exclusive):
      3 — Implemented       : year >= start_year AND (end_year NaN OR year <= end_year)
      2 — Under development : year >= ud_start   AND (ud_end   NaN OR year <= ud_end)
      1 — Under consideration: year >= uc_start  AND (uc_end   NaN OR year <= uc_end)
      0 — None / Abolished

    Exactly one status flag is set to 1 per CPI-year row.
    """
    y        = panel["year"].astype(float)
    s_start  = pd.to_numeric(panel["start_year"],   errors="coerce")
    s_end    = pd.to_numeric(panel["end_year"],      errors="coerce")
    ud_start = pd.to_numeric(panel["ud_start_raw"],  errors="coerce")
    ud_end   = pd.to_numeric(panel["ud_end_raw"],    errors="coerce")
    uc_start = pd.to_numeric(panel["uc_start_raw"],  errors="coerce")
    uc_end   = pd.to_numeric(panel["uc_end_raw"],    errors="coerce")

    # Priority 1: Implemented
    imp = (
        s_start.notna() &
        (y >= s_start) &
        (s_end.isna() | (y <= s_end))
    )
    # Priority 2: Under development (only if not already Implemented)
    ud = (
        ~imp &
        ud_start.notna() &
        (y >= ud_start) &
        (ud_end.isna() | (y <= ud_end))
    )
    # Priority 3: Under consideration (only if not Implemented or Under dev)
    uc = (
        ~imp & ~ud &
        uc_start.notna() &
        (y >= uc_start) &
        (uc_end.isna() | (y <= uc_end))
    )

    panel["status_implemented"]         = imp.astype(int)
    panel["status_under_development"]   = ud.astype(int)
    panel["status_under_consideration"] = uc.astype(int)
    panel["status_active"]              = (imp | ud | uc).astype(int)
    panel["status_abolished"] = (
        s_end.notna() & (y > s_end)
    ).astype(int)

    panel["status_level"] = np.where(
        imp, 3,
        np.where(ud, 2,
        np.where(uc, 1, 0))
    ).astype(int)

    return panel


def build_staggered_vars(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add staggered adoption variables required for Callaway & Sant'Anna DiD.

    first_year_industry
        First year covers_industry == 1 within 2016-2026.
        0 if never covers industry (never-treated).

    never_treated_industry
        1 if covers_industry == 0 in ALL years 2016-2026.

    always_treated_industry
        1 if covers_industry == 1 in the very first year the CPI appears
        with coverage data (coverage_active == 1).

    first_year_implemented
        First year status_implemented == 1 within 2016-2026. 0 if never.

    never_implemented
        1 if CPI never reached implemented status in the panel window.
    """
    panel = panel.sort_values(["cpi_id", "year"]).copy()

    # --- first_year_industry ---
    first_ind = (
        panel.loc[panel["covers_industry"] == 1, ["cpi_id", "year"]]
        .groupby("cpi_id")["year"]
        .min()
        .rename("first_year_industry")
    )
    panel = panel.merge(first_ind, on="cpi_id", how="left")
    panel["first_year_industry"] = (
        panel["first_year_industry"].fillna(0).astype(int)
    )

    # --- never_treated_industry ---
    panel["never_treated_industry"] = (
        panel["first_year_industry"] == 0
    ).astype(int)

    # --- always_treated_industry ---
    # 1 if CPI had industry coverage in the very first year of the panel (2016)
    panel["always_treated_industry"] = (
        panel["first_year_industry"] == 2016
    ).astype(int)

    # --- switcher_type ---
    # First year each CPI appears with any coverage data (coverage_active == 1)
    first_active_yr = (
        panel.loc[panel["coverage_active"] == 1, ["cpi_id", "year"]]
        .groupby("cpi_id")["year"]
        .min()
        .rename("_first_active_yr")
    )
    panel = panel.merge(first_active_yr, on="cpi_id", how="left")

    def _switcher_type(row):
        fyi = row["first_year_industry"]
        fay = row["_first_active_yr"]
        if fyi == 2016:
            return "always_treated"
        if fyi == 0:
            return "never_treated"
        # switcher: fyi > 2016
        if pd.isna(fay) or fay == fyi:
            return "new_cpi_with_industry"
        # fay < fyi: was active before it covered industry
        return "genuine_expansion"

    cpi_type = (
        panel.drop_duplicates("cpi_id")[["cpi_id", "first_year_industry", "_first_active_yr"]]
        .apply(_switcher_type, axis=1)
        .rename("switcher_type")
    )
    cpi_type_df = (
        panel.drop_duplicates("cpi_id")[["cpi_id"]]
        .copy()
        .assign(switcher_type=cpi_type.values)
    )
    panel = panel.drop(columns=["_first_active_yr"])
    panel = panel.merge(cpi_type_df, on="cpi_id", how="left")

    # --- first_year_implemented ---
    first_imp = (
        panel.loc[panel["status_implemented"] == 1, ["cpi_id", "year"]]
        .groupby("cpi_id")["year"]
        .min()
        .rename("first_year_implemented")
    )
    panel = panel.merge(first_imp, on="cpi_id", how="left")
    panel["first_year_implemented"] = (
        panel["first_year_implemented"].fillna(0).astype(int)
    )

    # --- never_implemented ---
    panel["never_implemented"] = (
        panel["first_year_implemented"] == 0
    ).astype(int)

    return panel


def merge_cbam_exposure(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Merge annual (by iso3 + year) and average (by iso3) CBAM exposure.

    Years 2025-2026 will have null cbam_exposure_annual because the exposure
    file only covers 2014-2024.  EEA/EFTA entries (ISL, LIE, NOR) will also
    have null exposure — this is expected and correct.
    """
    annual = pd.read_csv(
        "data/output/cbam_exposure_annual.csv", encoding="utf-8-sig"
    ).rename(columns={"t": "year"})

    avg = pd.read_csv(
        "data/output/cbam_exposure_avg.csv", encoding="utf-8-sig"
    )[["iso3", "cbam_exposure_avg",
       "cbam_exports_eu_avg_thousand_usd",
       "total_exports_avg_thousand_usd"]]

    annual_cols = [
        "iso3", "year",
        "cbam_exposure_annual",
        "cbam_exports_eu_thousand_usd",
        "total_exports_thousand_usd",
    ]
    panel = panel.merge(annual[annual_cols], on=["iso3", "year"], how="left")
    panel = panel.merge(avg, on="iso3", how="left")

    return panel


# ===========================================================================
# VALIDATION REPORT
# ===========================================================================

def validate_and_report(panel: pd.DataFrame) -> None:
    """Print the comprehensive validation summary to stdout."""
    SEP  = "=" * 72
    SEP2 = "-" * 72
    years = sorted(panel["year"].unique())

    print(f"\n{SEP}")
    print("  CPI PANEL — VALIDATION REPORT")
    print(SEP)

    # ------------------------------------------------------------------
    # 1. Panel dimensions
    # ------------------------------------------------------------------
    print(f"\n[1] PANEL DIMENSIONS")
    print(f"    Total rows    : {len(panel):,}")
    print(f"    Unique CPIs   : {panel['cpi_id'].nunique():,}")
    print(f"    Years covered : {years}")

    # ------------------------------------------------------------------
    # 2. Rows per year
    # ------------------------------------------------------------------
    print(f"\n[2] ROWS PER YEAR")
    rpy = panel.groupby("year").size()
    baseline = rpy.median()
    for yr, n in rpy.items():
        flag = "  *** SIGNIFICANTLY FEWER ***" if n < baseline * 0.9 else ""
        print(f"    {yr}: {n:>4} rows{flag}")

    # ------------------------------------------------------------------
    # 3. Instrument type breakdown
    # ------------------------------------------------------------------
    print(f"\n[3] INSTRUMENT TYPE BREAKDOWN  (counted at year=2016)")
    sub = panel[panel["year"] == panel["year"].min()]
    n_total = len(sub)
    for it, grp in sub.groupby("instrument_type", dropna=False):
        pct = len(grp) / n_total * 100
        print(f"    {str(it):<22}: {len(grp):>4} ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # 4. Scope breakdown
    # ------------------------------------------------------------------
    print(f"\n[4] SCOPE BREAKDOWN  (counted at year=2016)")
    for sc, grp in sub.groupby("scope", dropna=False):
        print(f"    {str(sc):<15}: {len(grp):>4}")

    # ------------------------------------------------------------------
    # 5. Sector coverage counts per year
    # ------------------------------------------------------------------
    print(f"\n[5] SECTOR COVERAGE BY YEAR  (number of CPIs covering each sector)")
    sector_labels = list(SECTOR_COL_MAP.keys())
    sector_cols   = list(SECTOR_COL_MAP.values())
    yr_width = 5
    hdr_year = "".join(f"{y:>{yr_width}}" for y in years)
    print(f"    {'Sector':<22}{hdr_year}")
    print(f"    {SEP2[:22+yr_width*len(years)]}")
    for label, col in zip(sector_labels, sector_cols):
        counts = [int(panel[panel["year"] == y][col].sum()) for y in years]
        row = "".join(f"{c:>{yr_width}}" for c in counts)
        print(f"    {label:<22}{row}")

    # ------------------------------------------------------------------
    # 6. Industry coverage trajectory
    # ------------------------------------------------------------------
    print(f"\n[6] INDUSTRY COVERAGE TRAJECTORY")
    prev = None
    for y in years:
        n = int(panel[panel["year"] == y]["covers_industry"].sum())
        if prev is None:
            print(f"    {y}: {n:>3} CPIs")
        else:
            delta = n - prev
            flag  = "  *** SIGNIFICANT JUMP ***" if abs(delta) >= 5 else ""
            print(f"    {y}: {n:>3} CPIs  (change: {delta:+d}){flag}")
        prev = n

    # ------------------------------------------------------------------
    # 7. Staggered adoption summary
    # ------------------------------------------------------------------
    print(f"\n[7] STAGGERED ADOPTION SUMMARY  (Callaway & Sant'Anna)")
    cpi_lvl = panel.drop_duplicates("cpi_id")
    n_never  = int(cpi_lvl["never_treated_industry"].sum())
    n_always = int(cpi_lvl["always_treated_industry"].sum())
    switchers = cpi_lvl[
        (cpi_lvl["never_treated_industry"]  == 0) &
        (cpi_lvl["always_treated_industry"] == 0)
    ]
    n_switch = len(switchers)
    print(f"    Never-treated  (never_treated_industry=1)   : {n_never}")
    print(f"    Always-treated (always_treated_industry=1)  : {n_always}")
    print(f"    Switchers      (added industry in 2016-2026): {n_switch}")
    if n_switch > 0:
        print(f"    Switcher cohort distribution (first_year_industry):")
        for yr, cnt in (
            switchers["first_year_industry"].value_counts().sort_index().items()
        ):
            print(f"      {yr}: {cnt} CPI(s)")

    print(f"\n    switcher_type distribution (CPI level):")
    for st, cnt in cpi_lvl["switcher_type"].value_counts().items():
        print(f"      {st:<28}: {cnt}")

    genuine = cpi_lvl[cpi_lvl["switcher_type"] == "genuine_expansion"][
        ["cpi_id", "country", "first_year_industry"]
    ]
    if len(genuine):
        print(f"\n    Genuine-expansion CPIs ({len(genuine)}):")
        for _, row in genuine.iterrows():
            print(f"      {row['cpi_id']:<30} country={row['country']!r}  "
                  f"first_year_industry={row['first_year_industry']}")
    else:
        print(f"\n    No genuine-expansion CPIs found.")

    # ------------------------------------------------------------------
    # 8. CBAM exposure merge results
    # ------------------------------------------------------------------
    print(f"\n[8] CBAM EXPOSURE MERGE RESULTS")
    all_iso3 = panel["iso3"].dropna().unique()
    matched_iso3 = (
        panel.loc[panel["cbam_exposure_annual"].notna(), "iso3"].dropna().unique()
    )
    print(f"    Unique ISO3 codes in panel              : {len(all_iso3)}")
    print(f"    ISO3 codes matched to annual exposure   : {len(matched_iso3)}")
    ann_nn = int(panel["cbam_exposure_annual"].notna().sum())
    print(f"    CPI-year rows with non-null cbam_exposure_annual: {ann_nn:,}")

    # EEA/EFTA flag
    for cpi in EEA_ETS_EXTENSION_IDS:
        print(f"    {cpi}: EFTA exempt — null exposure expected and correct")

    # Unmatched ISO3 (excluding EEA/EFTA)
    eea_iso3 = {"ISL", "LIE", "NOR"}
    unmatched = [
        iso for iso in all_iso3
        if iso not in matched_iso3 and iso not in eea_iso3
    ]
    if unmatched:
        print(f"    Other ISO3 not in exposure data ({len(unmatched)}):")
        for iso in sorted(unmatched):
            cpi_ids = panel[panel["iso3"] == iso]["cpi_id"].unique().tolist()
            print(f"      {iso}: {cpi_ids}")

    # CPIs with null ISO3
    null_iso3_cpis = (
        panel[panel["iso3"].isna()]["cpi_id"].unique().tolist()
    )
    if null_iso3_cpis:
        print(f"    CPIs with null ISO3 ({len(null_iso3_cpis)}): {null_iso3_cpis}")

    # ------------------------------------------------------------------
    # 9. Price data coverage
    # ------------------------------------------------------------------
    print(f"\n[9] PRICE DATA COVERAGE")
    print(f"    CPI-year rows with non-null price_usd: "
          f"{int(panel['price_usd'].notna().sum()):,}")
    for y in years:
        sub_p = panel.loc[panel["year"] == y, "price_usd"].dropna()
        if len(sub_p):
            print(
                f"    {y}: n={len(sub_p):>3}, "
                f"range=[{sub_p.min():.2f}, {sub_p.max():.2f}], "
                f"mean={sub_p.mean():.2f}, median={sub_p.median():.2f}"
            )
        else:
            print(f"    {y}: no price data")

    # ------------------------------------------------------------------
    # 10. Income group breakdown
    # ------------------------------------------------------------------
    print(f"\n[10] INCOME GROUP BREAKDOWN  (CPIs with price data, deduplicated)")
    ig_df = panel[panel["price_usd"].notna()].drop_duplicates("cpi_id")
    for ig, grp in ig_df.groupby("income_group", dropna=False):
        print(f"    {str(ig):<28}: {len(grp):>3} CPIs")

    # ------------------------------------------------------------------
    # 11. Subnational CPIs
    # ------------------------------------------------------------------
    print(f"\n[11] SUBNATIONAL CPIs")
    sub_df = panel[panel["scope"] == "Subnational"].drop_duplicates("cpi_id")
    print(f"    Total subnational CPIs: {len(sub_df)}")
    for iso, grp in sub_df.groupby("iso3", dropna=False):
        ids = sorted(grp["cpi_id"].unique())
        print(f"    {str(iso):<6}: {ids}")

    # ------------------------------------------------------------------
    # 12. Status level distribution per year
    # ------------------------------------------------------------------
    print(f"\n[12] STATUS LEVEL DISTRIBUTION PER YEAR")
    labels = {3: "Implemented(3)", 2: "UnderDev(2)",
              1: "UnderCon(1)",  0: "None(0)"}
    hdr_yr = "".join(f"{y:>{yr_width}}" for y in years)
    print(f"    {'Status':<16}{hdr_yr}")
    print(f"    {SEP2[:16+yr_width*len(years)]}")
    for level in (3, 2, 1, 0):
        counts = [
            int((panel[(panel["year"] == y) &
                       (panel["status_level"] == level)]).shape[0])
            for y in years
        ]
        row = "".join(f"{c:>{yr_width}}" for c in counts)
        print(f"    {labels[level]:<16}{row}")

    print(f"\n{SEP}\n")


# ===========================================================================
# DATA DICTIONARY OUTPUT
# ===========================================================================

def save_data_dictionary() -> None:
    """Write cpi_panel_data_dictionary.csv."""
    dd = pd.DataFrame(
        _DATA_DICTIONARY_ROWS,
        columns=["variable_name", "description", "source", "dtype", "notes"],
    )
    dd.to_csv("data/output/cpi_panel_data_dictionary.csv", index=False)
    print(f"  Data dictionary: {len(dd)} variables -> "
          "data/output/cpi_panel_data_dictionary.csv")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    t0 = time.perf_counter()
    SEP = "=" * 72
    print(SEP)
    print("  BUILD CPI PANEL")
    print(SEP)

    # ------------------------------------------------------------------
    # Step 1: Load source files
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading source files...")
    cpi_list    = load_cpi_list()
    price_long  = load_price_panel()
    sector_long = load_sector_panel()
    print(f"  cpi_list.xlsx          : {len(cpi_list)} CPIs")
    print(f"  cpi_price_usd.xlsx     : {price_long['cpi_id'].nunique()} CPIs, "
          f"{len(price_long):,} long rows (2016-2026)")
    print(f"  sectoral_coverage.xlsx : {sector_long['cpi_id'].nunique()} CPIs, "
          f"{len(sector_long):,} long rows")

    # ------------------------------------------------------------------
    # Step 2: Resolve ISO3 codes
    # ------------------------------------------------------------------
    print("\n[Step 2] Resolving ISO3 codes (warnings follow)...")
    cpi_list["iso3"] = cpi_list.apply(
        lambda r: resolve_iso3(r["jurisdiction"], r["country"], r["cpi_id"]),
        axis=1,
    )
    n_resolved = cpi_list["iso3"].notna().sum()
    print(f"  ISO3 resolved: {n_resolved}/{len(cpi_list)} CPIs  "
          f"({len(cpi_list) - n_resolved} unresolved -> null CBAM exposure)")

    # ------------------------------------------------------------------
    # Step 3: Build base panel (all CPI-year combinations)
    # ------------------------------------------------------------------
    print("\n[Step 3] Building base CPI x year panel...")
    all_cpi_ids = sorted(
        set(cpi_list["cpi_id"].dropna()) |
        set(sector_long["cpi_id"].dropna())
    )
    base = pd.DataFrame(
        [(cpi_id, year) for cpi_id in all_cpi_ids for year in PANEL_YEARS],
        columns=["cpi_id", "year"],
    )
    print(f"  Base panel: {len(base):,} rows  "
          f"({len(all_cpi_ids)} CPIs x {len(PANEL_YEARS)} years)")

    # ------------------------------------------------------------------
    # Step 4: Merge CPI list metadata
    # ------------------------------------------------------------------
    print("\n[Step 4] Merging CPI list metadata...")
    meta_cols = [
        "cpi_id", "cpi_name", "instrument_type",
        "country", "jurisdiction", "scope",
        "start_year_raw", "end_year_raw",
        "uc_start_raw", "uc_end_raw",
        "ud_start_raw", "ud_end_raw",
        "iso3",
    ]
    panel = base.merge(cpi_list[meta_cols], on="cpi_id", how="left")

    # Rename implementation date columns to their final output names
    panel = panel.rename(columns={
        "start_year_raw": "start_year",
        "end_year_raw":   "end_year",
    })

    # EEA/EFTA extension flag
    panel["is_eea_ets_extension"] = (
        panel["cpi_id"].isin(EEA_ETS_EXTENSION_IDS).astype(int)
    )

    # ------------------------------------------------------------------
    # Step 5: Derive time-varying status variables
    # ------------------------------------------------------------------
    print("\n[Step 5] Deriving time-varying status variables...")
    panel = derive_status_variables(panel)

    # ------------------------------------------------------------------
    # Step 6: Merge sector coverage
    # ------------------------------------------------------------------
    print("\n[Step 6] Merging sector coverage...")
    sector_cols = (
        ["cpi_id", "year", "sector_string", "coverage_active", "n_sectors_covered"]
        + list(SECTOR_COL_MAP.values())
    )
    panel = panel.merge(sector_long[sector_cols], on=["cpi_id", "year"], how="left")

    # Rows without sector data: fill with zeros and coverage_active=0
    for col in list(SECTOR_COL_MAP.values()) + ["n_sectors_covered", "coverage_active"]:
        panel[col] = panel[col].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Step 7: Merge price data
    # ------------------------------------------------------------------
    print("\n[Step 7] Merging price data...")
    price_cols = ["cpi_id", "year", "price_usd", "income_group", "region", "currency"]
    panel = panel.merge(price_long[price_cols], on=["cpi_id", "year"], how="left")
    panel["log_price_usd"] = np.where(
        panel["price_usd"] > 0,
        np.log(panel["price_usd"]),
        np.nan,
    )

    # Fallback income_group fill for CPIs absent from price file
    null_ig_before = panel["income_group"].isna().sum()
    panel["income_group"] = panel["income_group"].fillna(
        panel["country"].map(COUNTRY_INCOME_GROUP)
    )
    null_ig_after = panel["income_group"].isna().sum()
    print(f"  income_group: {null_ig_before} null before fallback fill, "
          f"{null_ig_after} null after")
    print("  income_group distribution (all CPI-year rows):")
    for ig, cnt in panel["income_group"].value_counts(dropna=False).items():
        print(f"    {str(ig):<28}: {cnt:,}")
    still_null = (
        panel[panel["income_group"].isna()]["cpi_id"].unique().tolist()
    )
    if still_null:
        print(f"  CPIs still missing income_group ({len(still_null)}): {still_null}")
    else:
        print("  No CPIs with null income_group remaining.")

    # ------------------------------------------------------------------
    # Step 8: Add treatment timing indicators
    # ------------------------------------------------------------------
    print("\n[Step 8] Adding treatment timing indicators...")
    # Primary: strict > (April 1 data reference predates mid-year announcements)
    panel["post_2019"] = (panel["year"] > 2019).astype(int)
    panel["post_2021"] = (panel["year"] > 2021).astype(int)
    panel["post_2023"] = (panel["year"] > 2023).astype(int)

    # Sensitivity variants (>= milestone, for robustness)
    panel["post_ge2019"] = (panel["year"] >= 2019).astype(int)
    panel["post_ge2021"] = (panel["year"] >= 2021).astype(int)
    panel["post_ge2023"] = (panel["year"] >= 2023).astype(int)

    # ------------------------------------------------------------------
    # Step 9: Build staggered adoption variables
    # ------------------------------------------------------------------
    print("\n[Step 9] Building staggered adoption variables...")
    panel = build_staggered_vars(panel)

    # ------------------------------------------------------------------
    # Step 10: Merge CBAM exposure
    # ------------------------------------------------------------------
    print("\n[Step 10] Merging CBAM exposure...")
    panel = merge_cbam_exposure(panel)

    # ------------------------------------------------------------------
    # Step 11: Enforce output column order; drop internal helpers
    # ------------------------------------------------------------------
    print("\n[Step 11] Finalising output columns...")
    output_cols = [
        # Identifiers
        "cpi_id", "cpi_name", "year",
        # Instrument characteristics
        "instrument_type", "country", "jurisdiction", "scope",
        "start_year", "end_year", "is_eea_ets_extension",
        # Time-varying status
        "status_implemented", "status_under_development",
        "status_under_consideration", "status_active",
        "status_abolished", "status_level",
        # Sector coverage
        "covers_power", "covers_industry", "covers_road_transport",
        "covers_aviation", "covers_shipping", "covers_buildings",
        "covers_waste", "covers_agriculture", "covers_forestry",
        "n_sectors_covered", "sector_string", "coverage_active",
        # Price
        "price_usd", "log_price_usd", "income_group", "region", "currency",
        # Treatment timing
        "post_2019", "post_2021", "post_2023",
        "post_ge2019", "post_ge2021", "post_ge2023",
        # Staggered adoption
        "first_year_industry", "never_treated_industry",
        "always_treated_industry", "switcher_type",
        "first_year_implemented", "never_implemented",
        # CBAM exposure
        "cbam_exposure_annual", "cbam_exposure_avg",
        "cbam_exports_eu_thousand_usd", "total_exports_thousand_usd",
        # Country identifier
        "iso3",
    ]
    panel = (
        panel[output_cols]
        .sort_values(["cpi_id", "year"])
        .reset_index(drop=True)
    )

    # Re-cast start_year / end_year to nullable Int64 for output
    for col in ("start_year", "end_year"):
        panel[col] = pd.to_numeric(panel[col], errors="coerce").astype("Int64")

    # ------------------------------------------------------------------
    # Step 12: Validate
    # ------------------------------------------------------------------
    validate_and_report(panel)

    # ------------------------------------------------------------------
    # Step 13: Save outputs
    # ------------------------------------------------------------------
    print("[Step 13] Saving outputs...")
    panel.to_csv("data/output/cpi_panel.csv", index=False)
    print(f"  Panel saved: {len(panel):,} rows x {len(panel.columns)} columns "
          "-> data/output/cpi_panel.csv")
    save_data_dictionary()

    elapsed = time.perf_counter() - t0
    print(f"\nTotal elapsed: {elapsed:.2f}s")
    print("Done.")


if __name__ == "__main__":
    main()
