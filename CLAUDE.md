# CBAM Coverage Analysis

## What this project does
Empirical analysis of EU CBAM spillovers on global carbon pricing adoption
and instrument design. World Bank research project producing a working paper.

## Project structure
- `src/` — Python scripts. Run in order: build_* scripts first, then analyse_*
  scripts, then generate_paper.py
- `data/lookup/` — Input data (xlsx, csv). Do not modify these files.
- `data/baci_hs12_src/` — BACI HS12 bilateral trade data. Files named
  BACI_HS12_Y{year}_V202601.csv for years 2014-2024. Large files, do not commit.
- `data/output/` — Generated outputs organized by layer (layer_a through layer_e)
- `paper/` — LaTeX paper. Compile with: cd paper && bash compile.sh
- `.venv/` — Python virtual environment (Python 3.11+)

## Pipeline execution order
1. `python src/build_cbam_exposure.py` — builds exposure from BACI trade data
2. `python src/build_cpi_panel.py` — builds CPI panel from lookup files + exposure
3. `python src/build_controls.py` — builds WDI/WGI control variables
4. `python src/analyse_layer_a_bruegel_benchmark.py`
5. `python src/analyse_layer_b_pipeline.py`
6. `python src/analyse_layer_c_sectoral.py`
7. `python src/analyse_layer_d_price.py`
8. `python src/analyse_layer_e_robustness.py`
9. `python src/generate_paper.py` — assembles paper/cbam_spillovers.tex

## Key analytical conventions
- CBAM exposure = share of total exports to the CBAM zone in CBAM-covered
  HS6 products. Measured as a proportion (0-1), not percentage points.
- CBAM zone importer perimeter:
  * 2014-2019: EU27 + EEA-EFTA (NOR, ISL, LIE) + GBR (pre-Brexit)
  * 2020-2024: EU27 + EEA-EFTA only (GBR left EU 31 Jan 2020)
  * Switzerland (CHE) excluded — separate ETS linking agreement
- EEA-EFTA and CHE excluded from exporter sample (CBAM-exempt as exporters)
- Post indicators use strict inequality (year > milestone, not >=):
  * post_2019: Green Deal announced July 2019; April 1 data reference date
    means 2019 observation predates the announcement
  * post_2021: CBAM regulation proposed July 2021
  * post_2023: CBAM entered into force October 2023
- Standard errors clustered at country level in all regressions
- Price models use log(price), not log(price + 1)
- Layer C entrant regressions use country-clustered SEs, not HC1
- Cox hazard models: disclose penalizer parameter; report robust=True

## Key packages
- linearmodels (PanelOLS for TWFE)
- statsmodels (OLS, GLM, Logit)
- lifelines (CoxPHFitter, KaplanMeierFitter)
- duckdb (BACI trade data queries)
- pandas, numpy, matplotlib

## Git conventions
- Commit messages: "fix:", "feat:", "paper:", "rebuild:" prefixes
- Do not commit data/baci_hs12_src/ or data/output/ (check .gitignore)
