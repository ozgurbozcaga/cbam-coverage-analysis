# CBAM Coverage Analysis

**World Bank Group Research Project**
Examining whether EU CBAM accelerated global carbon pricing adoption and industry sector coverage.

**Repository:** https://github.com/ozgurbozcaga/cbam-coverage-analysis (private)
**Last updated:** March 2026

---

## Project Overview

This project produces two outputs:
1. An analytical section for the World Bank State & Trends of Carbon Pricing publication
2. A World Bank Group Working Paper

The research examines whether the EU Carbon Border Adjustment Mechanism (CBAM) — through its announcement in 2019, formal legislative proposal in 2021, and entry into force in 2023 — accelerated the global adoption of carbon pricing instruments and increased the likelihood that new instruments were designed to cover CBAM-relevant sectors, particularly industry. The analysis combines World Bank State & Trends data with bilateral trade data from BACI to construct country-level CBAM trade exposure variables.

---

## Core Research Question

Did CBAM accelerate the global adoption of carbon pricing instruments, and did it specifically increase the likelihood that new instruments were designed to cover CBAM-relevant sectors? Is this relationship stronger in jurisdictions more economically exposed to CBAM through their export dependence on EU and UK markets?

---

## Repository Structure
```
cbam-coverage-analysis/
├── src/
│   ├── build_cbam_exposure.py       # BACI trade exposure construction
│   ├── build_cpi_panel.py           # CPI analytical panel construction
│   ├── analyse_h1.py                # Descriptive trend analysis (planned)
│   ├── analyse_h2_twfe.py           # TWFE baseline regressions (planned)
│   ├── analyse_h2_cs.R              # Callaway & Sant'Anna DiD in R (planned)
│   ├── analyse_h3_h4.py             # Exposure interactions and heterogeneity (planned)
│   └── analyse_h2a_hazard.R         # Cox proportional hazard model (planned)
├── data/
│   ├── lookup/                      # CPI list, prices, sectoral coverage (symlinked)
│   └── output/                      # Generated datasets
│       ├── cbam_exposure_annual.csv
│       ├── cbam_exposure_avg.csv
│       ├── cpi_panel.csv
│       └── cpi_panel_data_dictionary.csv
├── notebooks/                       # Jupyter notebooks for exploration
├── requirements.txt
└── README.md
```

---

## Data Assets

### Source Files (managed via OneDrive symlinks, not committed to git)

**cpi_list.xlsx** — Master registry of 162 CPIs: unique identifier, CPI name, instrument type (ETS/carbon tax/undefined), jurisdiction, country/region, scope (national/subnational/regional), start/end dates, policy pipeline dates, current status.

**cpi_price_usd.xlsx** — Annual carbon prices in nominal USD for 96 CPIs, 1990–2026. Converted from local currency using April 1 USD/LCU exchange rates. Income group and World Bank region attached for these 96 CPIs.

**sectoral_coverage_sectors.xlsx** — Sectoral coverage panel for 96 CPIs across 16 State & Trends report editions (CPW16 through S&T26), 2016–2026. Nine sector tokens: Power, Industry, Road transport, Aviation, Shipping, Buildings, Waste, Agriculture, Forestry.

**BACI HS12 (CEPII, 2012–2024)** — Bilateral trade flows at HS6 product level. Used to construct CBAM trade exposure variable. ~100MB per year, 11 files, accessed via OneDrive symlink.

### Constructed Output Files (committed to git)

**cbam_exposure_annual.csv** — Country-level CBAM trade exposure, 196 non-EU/non-EFTA exporting countries, 2014–2024. Time-varying importer perimeter: EU27 only 2014–2019, EU27+UK from 2020. Key values: Montenegro 22.18%, Ukraine 7.55%, Turkey 3.08%.

**cbam_exposure_avg.csv** — Static 2014–2024 average exposure per country. For comparability with Bruegel (2026).

**cpi_panel.csv** — Primary analytical panel: 1,771 rows, 161 CPIs × 11 years (2016–2026). All validation checks pass. See data dictionary for full variable definitions.

**cpi_panel_data_dictionary.csv** — Variable definitions, sources, dtypes, and notes for all columns in cpi_panel.csv.

---

## Key Data Notes

- ETS_US_VA removed from coverage file (incorrect entry)
- ETS_AU1 absent from coverage file — abolished before 2016, correct by design
- Tax_NG retained without sectoral coverage detail — expected implementation in 2026, coverage not yet confirmed
- ETS_EU_IS, ETS_EU_LI, ETS_EU_NO (Iceland, Liechtenstein, Norway EU ETS participation) retained with `is_eea_ets_extension` flag — CBAM exposure null by design since these are EFTA-exempt exporters
- Income group populated for all 161 CPIs (manual additions for 65 CPIs not in price file)
- Prices are nominal USD — deflation using World Bank GDP deflators applied as robustness check in H2c

---

## Key Empirical Pattern

The growth in industry sector coverage across CPIs over 2016–2026 is almost entirely an **extensive margin phenomenon**. Of 52 CPIs that first covered industry after 2016:
- 51 entered the sectoral coverage database already covering industry from inception
- 1 (ETS_CN, China's national ETS) represents a genuine within-instrument sector expansion: active from 2021 covering Power, added Industry in 2023

The primary story is not "existing CPIs expanded to cover industry in response to CBAM" but rather "post-CBAM, more instruments were adopted that included industry coverage from the start, particularly in jurisdictions with greater trade exposure to EU markets."

Industry coverage absolute count: 36 CPIs in 2016 → 79 in 2026, the largest absolute gain of any sector over the period.

---

## Hypotheses

**H1 — Industry sector experienced the largest absolute increase in the number of CPIs covering it over 2016–2026**
Descriptive hypothesis. Documents the stylised fact motivating the paper. Analysis covers absolute counts by sector and year, share of active CPIs covering each sector, decomposition into extensive vs intensive margin, and trajectory around 2019 CBAM policy discussions.

**H2a — Jurisdictions were more likely to advance CPIs through the policy pipeline following CBAM milestones**
Uses time-varying status variable (0=none, 1=consideration, 2=development, 3=implemented) as outcome. TWFE design with CBAM milestones as common shock and trade exposure as continuous intensity variable. Complementary Cox proportional hazard model as robustness.

**H2b — New CPIs adopted post-CBAM were more likely to be designed with industry coverage from inception**
Conditional on a new instrument entering the tracking database, tests whether it was more likely to cover industry if it entered post-CBAM milestone and in a high-exposure jurisdiction. TWFE primary specification; Callaway & Sant'Anna staggered adoption DiD where treatment is first entry with industry coverage and CBAM exposure predicts cohort timing.

**H2c — Carbon price levels grew faster post-CBAM in higher-exposure jurisdictions**
Tests whether CBAM strengthened the price intensive margin of existing instruments. Outcome: log nominal price in USD (log real price with GDP deflator as preferred robustness). Estimated separately for ETS and carbon tax subsamples.

**H3 — Post-CBAM industry coverage acceleration is stronger for CPIs in higher-CBAM-exposure jurisdictions**
Core novel contribution. CBAM trade exposure merged from BACI at country level onto all CPIs. TWFE with exposure × post-milestone interaction as primary. C&S continuous treatment specification testing whether higher-exposure countries belong to earlier adoption cohorts. Standard errors clustered at country level throughout.

**H4 — The H3 effect is concentrated among high- and upper-middle-income jurisdictions**
Subsample estimation by World Bank income group. WGI governance quality tested as moderator within high-income subsample. Note: income group is available for all 161 CPIs for H2a/H2b/H3; price subsample (96 CPIs) used for H2c income group analysis.

---

## Treatment Timing — Four Milestones

All specifications estimated separately for:
- **2019** — European Green Deal CBAM announcement (July 2019)
- **2021** — European Commission formal legislative proposal (July 2021)
- **2023** — CBAM entry into force, transitional period (October 2023)
- **Joint event study** — all three milestones with separate post-indicators

Note: S&T19 publication covers developments to approximately mid-2019. Instruments first appearing in S&T19 may predate the July 2019 announcement. Language uses "period of CBAM policy discussions beginning in 2019" rather than direct attribution to the announcement event.

---

## Econometric Strategy

### Primary — TWFE (following Bruegel identification logic)
```
Y_it = α_i + λ_t + β(Post_t × CBAM_exposure_i) + γX_it + ε_it
```

CBAM milestones as common shock, trade exposure as continuous intensity variable generating cross-sectional variation. CPI and year fixed effects. Standard errors clustered at country level.

### Complementary — Callaway & Sant'Anna Staggered Adoption DiD

Applied to H2b and H3. Treatment defined as first entry into coverage database with industry coverage. Cohort groups defined by year of first entry (2016–2026). Never-treated control group: CPIs that never cover industry. CBAM exposure tests whether higher-exposure country CPIs belong to earlier cohorts. Implemented in R using `did` package. Bias quantification: difference between TWFE and C&S aggregated ATTs reported explicitly.

### Complementary — Cox Proportional Hazard (H2a robustness)

First movement into each pipeline stage as duration outcome. CBAM exposure as time-invariant covariate. Handles left-censoring more explicitly than TWFE.

### What the Data Can and Cannot Identify

**Can identify:**
- Whether higher-exposure jurisdictions advanced CPIs through the pipeline faster post-2019
- Whether new post-CBAM instruments in high-exposure jurisdictions were more likely to cover industry
- Whether carbon prices grew faster post-CBAM in higher-exposure jurisdictions
- Income group heterogeneity in all of the above

**Cannot identify cleanly:**
- Whether CBAM caused existing instruments to expand sectoral coverage (within-instrument variation almost entirely absent)
- Whether S&T18→S&T19 jump is directly caused by the July 2019 announcement (publication timing ambiguity)
- Causal mechanisms — cannot rule out confounding through other EU trade policy channels

---

## Robustness Checks

| Code | Description |
|------|-------------|
| RC1 | National instruments only — excludes subnational CPIs |
| RC2 | Country-level aggregation — collapses to country-year panel |
| RC3 | S&T editions only — excludes interim CPW/CPD editions |
| RC4 | Excludes EU candidate countries (must align with EU ETS regardless of CBAM) |
| RC5 | Excludes Chinese subnational pilot ETSs |
| RC6 | Static vs time-varying CBAM exposure |
| RC7 | Real price deflation using World Bank GDP deflator |

---

## Panel Validation Summary

All 18 validation checks pass on cpi_panel.csv:

| Check | Result |
|-------|--------|
| Sector counts vs source Excel (Power, Industry) | PASS |
| Staggered adoption: 73 never / 36 always / 52 switchers | PASS |
| Only ETS_CN is genuine within-instrument expansion | PASS |
| Status hierarchy — no conflicts (max flags per row = 1) | PASS |
| Treatment timing indicators (post_2019, post_2021, post_2023) | PASS |
| Price data — no negatives, no values above 200 USD | PASS |
| Income group complete for all 161 CPIs | PASS |
| CBAM exposure nulls all EU/EFTA as expected | PASS |
| Ontario abolished CPI correctly handled | PASS |

---

## Setup
```bash
# Clone repository
git clone https://github.com/ozgurbozcaga/cbam-coverage-analysis.git
cd cbam-coverage-analysis

# Create virtual environment (Python 3.12)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Symlink data files (adjust paths to your OneDrive location)
ln -s "/mnt/c/Users/ozgur/OneDrive/Documents/WBG/CBAM_coverage/BACI_HS12" data/baci_hs12_src
ln -s "/mnt/c/Users/ozgur/OneDrive/Documents/WBG/CBAM_coverage/cpi_list.xlsx" data/lookup/cpi_list.xlsx
# ... (see setup instructions for full list)

# Build CBAM exposure variable (~90 seconds)
python3 src/build_cbam_exposure.py

# Build CPI analytical panel (~1 second)
python3 src/build_cpi_panel.py
```

---

## Relationship to Existing Literature

The closest paper is Bahí, Fuchs & Reverdy (Bruegel, WP 05/2026), which uses an earlier vintage of the same World Bank State & Trends data source. Their analysis uses a binary CPI adoption indicator, a static CBAM exposure measure, and a standard TWFE estimator without a sectoral dimension. This project introduces sectoral coverage as the primary outcome variable, uses a time-varying CBAM exposure measure with a corrected importer perimeter, and applies the Callaway & Sant'Anna estimator to test for bias in the TWFE approach. Comparisons with their estimates should be made cautiously given differences in data vintage, sample composition, and specification choices.

---

## Environment

- Python 3.12, DuckDB 1.5.1, pandas, linearmodels
- R with `did`, `survival` packages
- WSL2 Ubuntu on ThinkPad T14
- VS Code with Claude Code extension
- Data managed via OneDrive symlinks to WSL2 filesystem

---

*This project is under active development. Analysis scripts are being built iteratively. Contact: ozgurbozcaga@gmail.com*
