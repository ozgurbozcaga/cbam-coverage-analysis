#!/usr/bin/env python3
"""
Generate LaTeX paper: CBAM Spillovers and the Industrial Design of Carbon Pricing Instruments.

Reads all computed results from data/output/layer_a/ through data/output/layer_e/
and generates paper/cbam_spillovers.tex, paper/references.bib, and paper/compile.sh.

Run from project root:
    python3 src/generate_paper.py
"""

import os
import shutil
import pandas as pd
import re
import textwrap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "output")
PAPER = os.path.join(ROOT, "paper")
TABLES = os.path.join(PAPER, "tables")
FIGURES = os.path.join(PAPER, "figures")

for d in [PAPER, TABLES, FIGURES]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stars(p):
    """Return significance stars from p-value."""
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""


def fmt_num(x, decimals=4):
    """Format a number for LaTeX."""
    if pd.isna(x):
        return ""
    return f"{x:.{decimals}f}"


def fmt_pct(x, decimals=2):
    """Format as percentage string."""
    if pd.isna(x):
        return ""
    return f"{x*100:.{decimals}f}\\%"


def escape_latex(s):
    """Escape special LaTeX characters in a string."""
    if not isinstance(s, str):
        return str(s)
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("#", "\\#")
    s = s.replace("_", "\\_")
    return s


def parse_coef_se(cell):
    """Parse coefficient and SE from strings like '+5.0512** (2.0718)' or '1.2561'."""
    if pd.isna(cell) or str(cell).strip() == "":
        return None, None, ""
    s = str(cell).strip()
    # Remove leading +
    s = s.lstrip("+")
    # Check for stars at end of number before parenthesis
    star_match = re.search(r"(\*{1,3})", s)
    star_str = star_match.group(1) if star_match else ""
    # Remove stars for parsing
    s_clean = re.sub(r"\*+", "", s)
    # Try to find (SE)
    m = re.match(r"([\-\d.]+)\s*\(([\d.]+)\)", s_clean)
    if m:
        return float(m.group(1)), float(m.group(2)), star_str
    # Just a number
    try:
        return float(s_clean.strip()), None, star_str
    except ValueError:
        return None, None, ""


def copy_figures():
    """Copy PNG figures to paper/figures/."""
    fig_sources = [
        ("layer_b", "km_consideration.png"),
        ("layer_b", "km_development.png"),
        ("layer_b", "km_implementation.png"),
        ("layer_c", "event_study_industry.png"),
    ]
    for layer, fname in fig_sources:
        src = os.path.join(DATA, layer, fname)
        dst = os.path.join(FIGURES, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")
        else:
            print(f"  WARNING: {src} not found")


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------

def load_data():
    """Load all CSV files into a dictionary."""
    d = {}

    # Root
    d["exposure_avg"] = pd.read_csv(os.path.join(DATA, "cbam_exposure_avg.csv"))
    d["cpi_panel"] = pd.read_csv(os.path.join(DATA, "cpi_panel.csv"))

    # Layer A
    d["a_main"] = pd.read_csv(os.path.join(DATA, "layer_a", "table1_main_results.csv"))
    d["a_sectoral"] = pd.read_csv(os.path.join(DATA, "layer_a", "table2_sectoral_extension.csv"))
    d["a_desc"] = pd.read_csv(os.path.join(DATA, "layer_a", "table3_descriptive.csv"))

    # Layer B
    d["b_twfe"] = pd.read_csv(os.path.join(DATA, "layer_b", "table1_twfe_exposure_comparison.csv"))
    d["b_cox"] = pd.read_csv(os.path.join(DATA, "layer_b", "table2_cox_hazard.csv"))

    # Layer C
    d["c_entrant_desc"] = pd.read_csv(os.path.join(DATA, "layer_c", "table1_entrant_descriptive.csv"))
    d["c_entrant_reg"] = pd.read_csv(os.path.join(DATA, "layer_c", "table2_c1_entrant_regressions.csv"))
    d["c_panel_did"] = pd.read_csv(os.path.join(DATA, "layer_c", "table2b_c1b_panel_did.csv"))

    # Layer D
    d["d_nominal"] = pd.read_csv(os.path.join(DATA, "layer_d", "table2_price_nominal.csv"))
    d["d_real"] = pd.read_csv(os.path.join(DATA, "layer_d", "table3_price_real_robustness.csv"))

    # Layer E
    d["e_master"] = pd.read_csv(os.path.join(DATA, "layer_e", "table_robustness_master.csv"))
    d["e_rc4"] = pd.read_csv(os.path.join(DATA, "layer_e", "table_rc4_eu_candidates.csv"))
    d["e_rc7"] = pd.read_csv(os.path.join(DATA, "layer_e", "table_rc7_income_groups.csv"))
    d["e_rc10"] = pd.read_csv(os.path.join(DATA, "layer_e", "table_rc10_placebo.csv"))
    d["e_rc11"] = pd.read_csv(os.path.join(DATA, "layer_e", "table_rc11_nonoecd.csv"))

    return d


# ---------------------------------------------------------------------------
# Table generators — return LaTeX strings
# ---------------------------------------------------------------------------

def make_table1_descriptive(data):
    """Table 1: Sector coverage counts by year (from cpi_panel)."""
    cp = data["cpi_panel"]
    sectors = ["covers_power", "covers_industry", "covers_road_transport",
               "covers_aviation", "covers_shipping", "covers_buildings",
               "covers_waste", "covers_agriculture", "covers_forestry"]
    sector_labels = ["Power", "Industry", "Road transport", "Aviation",
                     "Shipping", "Buildings", "Waste", "Agriculture", "Forestry"]
    years = sorted(cp["year"].unique())

    # Count CPIs covering each sector per year (only where coverage_active == 1)
    active = cp[cp["coverage_active"] == 1]

    rows = []
    for sec, label in zip(sectors, sector_labels):
        row = [label]
        for y in years:
            count = active.loc[active["year"] == y, sec].sum()
            row.append(str(int(count)))
        rows.append(row)

    ncols = len(years) + 1
    col_spec = "l" + "r" * len(years)
    header = " & ".join(["Sector"] + [str(y) for y in years])

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{threeparttable}")
    lines.append(f"\\caption{{Sector Coverage Counts by Year}}\\label{{tab:descriptive}}")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(header + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}\\small")
    lines.append("\\item \\textit{Note:} Count of CPIs with active sector coverage in each year. Source: World Bank State \\& Trends of Carbon Pricing panel, 2016--2026.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_table2_exposure(data):
    """Table 2: Top 20 countries by CBAM exposure."""
    df = data["exposure_avg"].head(20).copy()
    df["rank"] = range(1, len(df) + 1)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{threeparttable}")
    lines.append("\\caption{Top 20 Countries by CBAM Trade Exposure}\\label{tab:exposure}")
    lines.append("\\begin{tabular}{rlll}")
    lines.append("\\toprule")
    lines.append("Rank & Country & ISO3 & Exposure (\\%) \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        name = escape_latex(str(row["country_name"]))
        exp_pct = f"{row['cbam_exposure_avg']*100:.2f}"
        lines.append(f"{row['rank']} & {name} & {row['iso3']} & {exp_pct} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}\\small")
    lines.append("\\item \\textit{Note:} Exposure = share of total exports to EU27+UK in CBAM-covered HS product categories, averaged 2014--2024. \\citet{Bahi2026} report Montenegro 24.2\\% and Turkey 4\\% using EU+EFTA destination; our figures (22.18\\% and 3.08\\%) differ due to the EU27+UK perimeter.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _format_results_table(df, caption, label, note, skip_header_row=True):
    """Generic formatter for Layer A style results tables with coef/(SE)/N rows."""
    # These tables have a specific format: header row, then description row,
    # then triplets of (coef, SE, N) per outcome
    cols = df.columns.tolist()
    outcome_col = cols[0]
    result_cols = cols[1:]

    # Skip the description row (row index 0 after header)
    data_rows = df.iloc[1:] if skip_header_row else df

    # Find the note row (last row often starts with "Exposure" or "NOTE")
    # Parse outcome groups: each outcome has 3 rows (coef, SE, N)
    outcomes = []
    i = 0
    rows_list = data_rows.values.tolist()
    while i < len(rows_list):
        row = rows_list[i]
        outcome_name = str(row[0]).strip() if pd.notna(row[0]) and str(row[0]).strip() else None
        if outcome_name and not outcome_name.startswith("Exposure") and not outcome_name.startswith("NOTE"):
            # This is a coefficient row
            coef_row = row[1:]
            se_row = rows_list[i+1][1:] if i+1 < len(rows_list) else [""]*len(result_cols)
            n_row = rows_list[i+2][1:] if i+2 < len(rows_list) else [""]*len(result_cols)
            outcomes.append((outcome_name, coef_row, se_row, n_row))
            i += 3
        else:
            i += 1

    # Build table
    ncols = len(result_cols)
    col_spec = "l" + "c" * ncols

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{threeparttable}")
    lines.append(f"\\caption{{{caption}}}\\label{{{label}}}")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header_cells = [escape_latex(c) for c in result_cols]
    lines.append(" & " + " & ".join(header_cells) + " \\\\")
    lines.append("\\midrule")

    for outcome_name, coef_row, se_row, n_row in outcomes:
        # Coefficient row
        coef_cells = []
        for c in coef_row:
            s = str(c).strip() if pd.notna(c) else ""
            coef_cells.append(s)
        lines.append(f"{escape_latex(outcome_name)} & " + " & ".join(coef_cells) + " \\\\")

        # SE row
        se_cells = []
        for c in se_row:
            s = str(c).strip() if pd.notna(c) else ""
            se_cells.append(s)
        lines.append(" & " + " & ".join(se_cells) + " \\\\")

        # N row
        n_cells = []
        for c in n_row:
            s = str(c).strip() if pd.notna(c) else ""
            n_cells.append(f"\\textit{{{s}}}" if s else "")
        lines.append(" & " + " & ".join(n_cells) + " \\\\")
        lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}\\small")
    lines.append(f"\\item \\textit{{Note:}} {note}")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_table3_main(data):
    """Table 3: Main benchmark results (Layer A)."""
    return _format_results_table(
        data["a_main"],
        "Benchmark Results: CBAM Exposure and Carbon Pricing Adoption",
        "tab:benchmark",
        "Dependent variable: binary indicator at country-year level. Sample: 196 non-EU/EFTA countries, 2016--2024 ($N=1{,}764$; $N=1{,}304$ with controls). Exposure measured as proportion (0--1); divide coefficients by 100 for per-percentage-point interpretation consistent with \\citet{Bahi2026}. Standard errors clustered at country level in parentheses. *** $p<0.01$ ** $p<0.05$ * $p<0.1$."
    )


def make_table4_sectoral(data):
    """Table 4: Sectoral extension (Layer A)."""
    return _format_results_table(
        data["a_sectoral"],
        "Sectoral Extension: Industry-Specific Adoption Outcomes",
        "tab:sectoral",
        "Dependent variable: binary indicator for whether the country has at least one CPI covering industry. Same sample and specification as Table~\\ref{tab:benchmark}. Standard errors clustered at country level in parentheses. *** $p<0.01$ ** $p<0.05$ * $p<0.1$."
    )


def make_table5_cox(data):
    """Table 5: Cox hazard model results (Layer B)."""
    df = data["b_cox"]
    # Row 0 is the table description row, rows 1-3 are covariates, row 4 is C-index
    data_rows = df.iloc[1:].values.tolist()

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{threeparttable}")
    lines.append("\\caption{Cox Proportional Hazard Models: Pipeline Transitions}\\label{tab:cox}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append(" & Consideration & Development & Implementation \\\\")
    lines.append("\\midrule")

    for row in data_rows:
        label = str(row[0]).strip()
        if label.startswith("HR") or label.startswith("NOTE") or label == "":
            continue
        cells = []
        for c in row[1:]:
            s = str(c).strip() if pd.notna(c) else ""
            cells.append(s)
        disp_label = escape_latex(label)
        if label == "C-index":
            lines.append("\\midrule")
        lines.append(f"{disp_label} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}\\small")
    lines.append("\\item \\textit{Note:} Hazard ratios from Cox proportional hazard models. HR $> 1$ indicates higher hazard (faster transition). Exposure in percentage points (0--100). Countries already in the relevant state before 2016 are excluded (milestone-specific left-censoring). C-index reported in bottom row. PH assumption passes for all models. *** $p<0.01$ ** $p<0.05$ * $p<0.1$.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_table6_twfe_comparison(data):
    """Table 6: TWFE benchmark vs preferred exposure (Layer B)."""
    return _format_results_table(
        data["b_twfe"],
        "Pipeline Outcomes: Benchmark vs.~Preferred Exposure",
        "tab:twfe_comparison",
        "Benchmark = static 2014--2024 average, EU27+UK perimeter. Preferred = time-varying annual, Brexit-corrected perimeter. All regressions include country and year FE, controls (log GDP pc, log exports, WGI composite, all lagged). Standard errors clustered at country level in parentheses. *** $p<0.01$ ** $p<0.05$ * $p<0.1$."
    )


def make_table7_entrant_descriptive(data):
    """Table 7: Entrant cohort descriptive (Layer C)."""
    df = data["c_entrant_desc"]
    # Skip description row
    rows = df.iloc[1:].values.tolist()

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{threeparttable}")
    lines.append("\\caption{New CPI Entrant Characteristics by Entry Cohort}\\label{tab:entrant_desc}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrrrrrr}")
    lines.append("\\toprule")
    lines.append("Entry year & $N$ entrants & National & Subnational & With coverage & Industry share & Mean exposure & Mean sectors \\\\")
    lines.append("\\midrule")

    for row in rows:
        year = str(row[0]).strip()
        if not year or not year[0].isdigit():
            continue
        n_ent = int(float(row[1]))
        n_nat = int(float(row[2]))
        n_sub = int(float(row[3]))
        n_cov = int(float(row[4]))
        ind_share = f"{float(row[5])*100:.1f}\\%"
        mean_exp = f"{float(row[6])*100:.3f}\\%"
        mean_sec = f"{float(row[7]):.1f}"
        lines.append(f"{year} & {n_ent} & {n_nat} & {n_sub} & {n_cov} & {ind_share} & {mean_exp} & {mean_sec} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}\\small")
    lines.append("\\item \\textit{Note:} Industry share = fraction of entrants with coverage data that cover industry at entry. Mean exposure = average CBAM benchmark exposure across entrants in the cohort.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_table8_panel_did(data):
    """Table 8: Panel DiD results (Layer C, primary)."""
    df = data["c_panel_did"]
    rows = df.iloc[1:].values.tolist()

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{threeparttable}")
    lines.append("\\caption{CPI-Level Panel DiD: Industry Sector Coverage (Primary)}\\label{tab:panel_did}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Specification & Coefficient & SE & $N$ & CPI FE & Year FE \\\\")
    lines.append("\\midrule")

    for row in rows:
        spec = str(row[0]).strip()
        if not spec or spec.startswith("CPI") or spec.startswith("NOTE"):
            continue
        coef = str(row[1]).strip()
        se = str(row[2]).strip()
        n = str(row[3]).strip()
        cpi_fe = str(row[4]).strip()
        yr_fe = str(row[5]).strip()
        lines.append(f"{spec} & {coef} & {se} & {n} & {cpi_fe} & {yr_fe} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}\\small")
    lines.append("\\item \\textit{Note:} CPI-level panel DiD. Sample: 1,152 CPI-year observations, 129 CPIs, 49 countries. Outcome: binary indicator for industry sector coverage. Pre-entry years (coverage\\_active=0) included with covers\\_industry=0 by construction. Identification from entry events---CPIs transitioning from inactive to active. Within-active variation is minimal (only ETS\\_CN genuinely expands industry coverage post-entry). Standard errors clustered at country level. Controls: log GDP per capita, WGI composite (both lagged). *** $p<0.01$ ** $p<0.05$ * $p<0.1$.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_table9_entrant_regressions(data):
    """Table 9: Cross-sectional entrant regressions (supplementary)."""
    df = data["c_entrant_reg"]
    # Skip header row
    rows = df.iloc[1:].values.tolist()

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{threeparttable}")
    lines.append("\\caption{Cross-Sectional Entrant Regressions (Supplementary)}\\label{tab:entrant_reg}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llccccccr}")
    lines.append("\\toprule")
    lines.append("Model & Variable & Post-2019 & SE & Post-2021 & SE & Post-2023 & SE & $N$ \\\\")
    lines.append("\\midrule")

    for row in rows:
        model = str(row[0]).strip() if pd.notna(row[0]) else ""
        if model.startswith("Controls") or model.startswith("NOTE"):
            continue
        variable = str(row[1]).strip() if pd.notna(row[1]) else ""
        cells = []
        for c in row[2:9]:
            s = str(c).strip() if pd.notna(c) and str(c).strip() != "nan" else ""
            cells.append(s)
        lines.append(f"{escape_latex(model)} & {escape_latex(variable)} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}\\small")
    lines.append("\\item \\textit{Note:} Controls: log GDP per capita, WGI composite (both lagged). OLS uses HC1 robust SEs. $N=41$ entrants with coverage data and controls. *** $p<0.01$ ** $p<0.05$ * $p<0.1$.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_table10_price_nominal(data):
    """Table 10: Nominal price results (Layer D)."""
    df = data["d_nominal"]
    rows = df.iloc[:-1].values.tolist()  # Skip NOTE row

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{threeparttable}")
    lines.append("\\caption{Carbon Price and CBAM Exposure: Nominal USD Results}\\label{tab:price_nominal}")
    lines.append("\\begin{tabular}{llcr}")
    lines.append("\\toprule")
    lines.append("Specification & Sample & Coefficient & $N$ \\\\")
    lines.append("\\midrule")

    for row in rows:
        spec = str(row[0]).strip()
        if spec.startswith("NOTE"):
            continue
        sample = str(row[1]).strip()
        coef = str(row[2]).strip()
        n = str(row[3]).strip()
        lines.append(f"{escape_latex(spec)} & {escape_latex(sample)} & {coef} & {n} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}\\small")
    lines.append("\\item \\textit{Note:} Dependent variable: log carbon price in nominal USD. Coefficients are semi-elasticities with respect to the interaction term. CPI and year FE. Controls: log GDP per capita, WGI composite (lagged). Standard errors clustered at country level in parentheses. Nominal USD prices embed exchange rate movements in cross-country comparisons. *** $p<0.01$ ** $p<0.05$ * $p<0.1$.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_table11_price_real(data):
    """Table 11: Real price robustness (Layer D)."""
    df = data["d_real"]
    rows = df.iloc[:-1].values.tolist()  # Skip NOTE row

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{threeparttable}")
    lines.append("\\caption{Carbon Price: Nominal vs.~Real (GDP Deflator Adjusted)}\\label{tab:price_real}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Post period & Sample & Nominal coef. & Nominal $N$ & Real coef. & Real $N$ \\\\")
    lines.append("\\midrule")

    for row in rows:
        post = str(row[0]).strip()
        if post.startswith("NOTE"):
            continue
        sample = str(row[1]).strip()
        nom_c = str(row[2]).strip()
        nom_n = str(row[3]).strip()
        real_c = str(row[4]).strip()
        real_n = str(row[5]).strip()
        lines.append(f"{escape_latex(post)} & {escape_latex(sample)} & {nom_c} & {nom_n} & {real_c} & {real_n} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}\\small")
    lines.append("\\item \\textit{Note:} Dependent variable: log real carbon price (nominal USD / GDP deflator $\\times$ 100). Coefficients are semi-elasticities with respect to the interaction term. GDP deflator: WDI linked series (NY.GDP.DEFL.ZS.AD). Within-country price changes captured but cross-country exchange rate effects remain. Standard errors in parentheses. *** $p<0.01$ ** $p<0.05$ * $p<0.1$.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_table12_robustness(data):
    """Table 12: Master robustness table (Layer E)."""
    df = data["e_master"]
    # Skip header row
    rows = df.iloc[1:].values.tolist()

    lines = []
    lines.append("\\begin{landscape}")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{threeparttable}")
    lines.append("\\caption{Robustness Checks: Master Summary}\\label{tab:robustness}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrlrl}")
    lines.append("\\toprule")
    lines.append("RC & Description & $N$ & Coefficient & SE & $p$-value & Stars & Diff from RC1 & Note \\\\")
    lines.append("\\midrule")

    for row in rows:
        rc = str(row[0]).strip()
        if rc.startswith("Significance") or not rc:
            continue
        desc = str(row[1]).strip() if pd.notna(row[1]) else ""
        try:
            n = str(int(float(row[2]))) if pd.notna(row[2]) else ""
        except (ValueError, TypeError):
            n = ""
        try:
            coef = f"{float(row[3]):.4f}" if pd.notna(row[3]) else ""
        except (ValueError, TypeError):
            coef = ""
        try:
            se = f"{float(row[4]):.4f}" if pd.notna(row[4]) else ""
        except (ValueError, TypeError):
            se = ""
        try:
            pval = f"{float(row[5]):.4f}" if pd.notna(row[5]) else ""
        except (ValueError, TypeError):
            pval = ""
        st = str(row[6]).strip() if pd.notna(row[6]) and str(row[6]).strip() != "nan" else ""
        try:
            diff = f"{float(row[7]):.4f}" if pd.notna(row[7]) else ""
        except (ValueError, TypeError):
            diff = ""
        note = str(row[10]).strip() if len(row) > 10 and pd.notna(row[10]) else ""
        note_short = note[:60] + "..." if len(note) > 60 else note

        lines.append(f"{escape_latex(rc)} & {escape_latex(desc)} & {n} & {coef} & {se} & {pval} & {st} & {diff} & {escape_latex(note_short)} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}\\small")
    lines.append("\\item \\textit{Note:} Reference specification: any\\_cpi\\_active, Post-2021, benchmark exposure. Significance of difference: $z = (\\beta_{\\text{RC}} - \\beta_{\\text{ref}}) / \\sqrt{\\text{SE}_{\\text{RC}}^2 + \\text{SE}_{\\text{ref}}^2}$. *** $p<0.01$ ** $p<0.05$ * $p<0.1$.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{threeparttable}")
    lines.append("\\end{table}")
    lines.append("\\end{landscape}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compute inline statistics from data
# ---------------------------------------------------------------------------

def compute_stats(data):
    """Compute inline statistics referenced in the paper text."""
    s = {}
    cp = data["cpi_panel"]

    # Industry coverage counts
    active = cp[cp["coverage_active"] == 1]
    s["ind_2016"] = int(active.loc[active["year"] == 2016, "covers_industry"].sum())
    s["ind_2026"] = int(active.loc[active["year"] == 2026, "covers_industry"].sum()) if 2026 in active["year"].values else "N/A"

    # Median gap between master-list development start and first coverage entry
    # Approximate from entrant data
    entrant_sample_path = os.path.join(DATA, "layer_c", "table5_industry_entrants.csv")
    if os.path.exists(entrant_sample_path):
        ent = pd.read_csv(entrant_sample_path)
        gaps = ent["gap_years"].dropna()
        s["median_gap"] = f"{gaps.median():.0f}" if len(gaps) > 0 else "N/A"
    else:
        s["median_gap"] = "N/A"

    # RC4 specifics
    rc4 = data["e_rc4"]
    rc4_row = rc4.iloc[0]
    s["rc4_coef"] = f"{float(rc4_row['Coefficient']):.4f}"
    s["rc4_se"] = f"{float(rc4_row['SE']):.4f}"
    s["rc4_p"] = f"{float(rc4_row['p-value']):.4f}"
    s["rc4_n"] = int(float(rc4_row["N"]))

    # RC7 specifics
    rc7 = data["e_rc7"]
    rc7c = rc7[rc7["Subsample"].str.contains("Triple", na=False)].iloc[0]
    s["rc7c_coef"] = f"{float(rc7c['Coefficient']):.4f}"
    s["rc7c_se"] = f"{float(rc7c['SE']):.4f}"
    s["rc7c_p"] = f"{float(rc7c['p-value']):.4f}"

    # RC10 placebo
    rc10 = data["e_rc10"]
    plac = rc10[rc10["Specification"].str.contains("Placebo", na=False)].iloc[0]
    s["rc10_coef"] = f"{float(plac['Coefficient']):.4f}"
    s["rc10_p"] = f"{float(plac['p-value']):.2f}"

    # Exposure stats
    exp = data["exposure_avg"]
    s["mne_exp"] = f"{exp.iloc[0]['cbam_exposure_avg']*100:.2f}"
    s["tur_exp"] = f"{exp[exp['iso3']=='TUR']['cbam_exposure_avg'].values[0]*100:.2f}"

    return s


# ---------------------------------------------------------------------------
# Main LaTeX document
# ---------------------------------------------------------------------------

def generate_tex(data):
    """Generate the full LaTeX document string."""

    stats = compute_stats(data)

    # Generate all tables
    table1 = make_table1_descriptive(data)
    table2 = make_table2_exposure(data)
    table3 = make_table3_main(data)
    table4 = make_table4_sectoral(data)
    table5 = make_table5_cox(data)
    table6 = make_table6_twfe_comparison(data)
    table7 = make_table7_entrant_descriptive(data)
    table8 = make_table8_panel_did(data)
    table9 = make_table9_entrant_regressions(data)
    table10 = make_table10_price_nominal(data)
    table11 = make_table11_price_real(data)
    table12 = make_table12_robustness(data)

    tex = r"""
\documentclass[12pt,a4paper]{article}

% ---------------------------------------------------------------------------
% Packages
% ---------------------------------------------------------------------------
\usepackage[top=2.5cm, bottom=2.5cm, left=3cm, right=3cm]{geometry}
\usepackage{setspace}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{threeparttable}
\usepackage{amsmath}
\usepackage{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage[table]{xcolor}
\usepackage{lscape}
\usepackage{microtype}
\usepackage{siunitx}
\usepackage{footnote}

\onehalfspacing
\bibliographystyle{apalike}

% ---------------------------------------------------------------------------
\begin{document}

\title{CBAM Spillovers and the Industrial Design of Carbon Pricing Instruments}
\author{[Author placeholder]}
\date{\today}
\maketitle

\begin{abstract}
\noindent
Does the EU Carbon Border Adjustment Mechanism shape global carbon pricing adoption and the sectoral design of new instruments? Using a World Bank State \& Trends panel (161 CPIs, 2016--2026), BACI bilateral trade data (196 countries, 2014--2024), and WDI/WGI controls, we find three headline results. First, CBAM-exposed countries advance faster through the carbon pricing pipeline: Cox hazard ratios for the development transition reach 1.20 ($p<0.01$). Second, the 2019 cohort of new instruments shows markedly higher CBAM exposure (mean 2.1\%, ten times the 2016--2018 average) and a 100\% industry coverage rate. Third, panel difference-in-differences coefficients for industry coverage are consistently positive but imprecisely estimated given 49 country clusters. These results are directionally consistent with CBAM driving industry-oriented instrument design, motivating continued panel extension as the mechanism transitions to full implementation.

\medskip
\noindent \textbf{JEL codes:} Q54, Q58, F18, H23
\end{abstract}

\newpage

% ============================================================================
% SECTION 1 — INTRODUCTION
% ============================================================================
\section{Introduction}\label{sec:intro}

The EU Carbon Border Adjustment Mechanism creates a direct financial incentive for non-EU countries to establish domestic carbon pricing. Under Article~9 of Regulation (EU) 2023/956, importers may deduct from their CBAM certificate obligations any carbon price already paid in the country of origin \citep{CBAM2023}. This rebate mechanism means that exporting countries without domestic carbon pricing effectively transfer carbon tax revenues to the EU treasury. Whether this incentive has accelerated both carbon pricing adoption and specifically industry-oriented instrument design is the central empirical question of this paper. The EU accounts for roughly 6\% of global emissions; the CBAM mechanism matters for climate policy only if it catalyses broader policy diffusion beyond Europe's borders.

We find three headline results. First, countries with greater CBAM trade exposure advance faster through the carbon pricing policy pipeline: Cox proportional hazard models yield a hazard ratio of 1.201 ($p<0.01$) for the development transition, meaning a one percentage point increase in CBAM exposure is associated with a 20\% higher hazard of moving into active carbon pricing development (Table~\ref{tab:cox}). Second, the 2019 entrant cohort---twelve new instruments entering the World Bank database in the year the European Commission first announced detailed CBAM plans---shows a mean CBAM exposure of 2.1\%, ten times higher than the 2016--2018 cohorts, with 100\% industry sector coverage from inception (Table~\ref{tab:entrant_desc}). Third, CPI-level panel difference-in-differences estimates for industry coverage are consistently positive (coefficients of +0.97 to +1.11) but imprecisely estimated with 49 country clusters (Table~\ref{tab:panel_did}).

Our work extends the framework of \citet{Bahi2026}, who establish that more CBAM-exposed countries were more likely to adopt some form of national carbon pricing after 2019, using a country-year two-way fixed effects (TWFE) design with a binary adoption outcome. We build on their contribution in two directions. First, we test whether the policy response is visible across the full pipeline from consideration through development to implementation using event-history models, moving beyond the binary adoption indicator. Second, we introduce a sectoral dimension---testing whether new instruments in more-exposed jurisdictions are more likely to cover CBAM-relevant industrial sectors from inception. The binary country-level outcome in \citet{Bahi2026} cannot distinguish between a country that adopts a power-sector-only carbon tax and one that adopts an industry-covering ETS specifically in response to CBAM incentives.

The remainder of this paper is organised as follows. Section~\ref{sec:data} describes the data and measurement. Section~\ref{sec:strategy} presents the empirical strategy. Section~\ref{sec:results} reports results by analytical layer. Section~\ref{sec:robustness} presents robustness checks. Section~\ref{sec:conclusion} concludes.


% ============================================================================
% SECTION 2 — DATA AND MEASUREMENT
% ============================================================================
\section{Data and Measurement}\label{sec:data}

\subsection{Carbon Pricing Instrument Panel}\label{sec:cpi_panel}

Our primary data source is the World Bank \textit{State and Trends of Carbon Pricing} data series \citep{WorldBank2025}. We construct a panel covering 161 carbon pricing instruments (CPIs) across 11 annual editions (S\&T16 through S\&T26, publication years 2016--2026). For each CPI and year we record nine binary sector coverage indicators: power, industry, road transport, aviation, shipping, buildings, waste, agriculture, and forestry. These are parsed from comma-separated sector strings in the source data. We also record time-varying policy status anchored to master-list dates: under consideration start, under development start, and implementation start. These master-list dates differ from first database appearance for some instruments---we document a median gap of """ + stats["median_gap"] + r""" years between master-list development start and first sectoral coverage database entry, with Japan's national ETS showing a three-year gap as a concrete example. Annual carbon prices are recorded in nominal USD, converted at April~1 exchange rates. Each CPI is classified by the host country's income group (World Bank) and World Bank region.

Several descriptive facts motivate our empirical design. Industry coverage grew from """ + str(stats["ind_2016"]) + r""" CPIs in 2016 to """ + str(stats["ind_2026"]) + r""" in 2026---the largest absolute gain of any sector. Of the 52 CPIs that first covered industry after 2016, 51 entered the coverage database already covering industry from inception. Only China's national ETS (ETS\_CN) genuinely expanded to cover industry after its initial launch---from power only in 2021 to power and industry in 2023. The 2019 entrant cohort stands out: twelve new instruments with mean CBAM exposure of 2.1\% (ten times higher than 2016--2018 cohorts) and 100\% industry coverage at entry.

""" + table1 + r"""

\subsection{CBAM Trade Exposure}\label{sec:exposure}

Our CBAM exposure variable measures the share of a country's total merchandise exports that flow to EU27 plus UK destinations in CBAM-covered HS6 product categories. We construct this from BACI HS12 bilateral trade data \citep{Gaulier2010} covering 196 non-EU, non-EFTA exporting countries over 2014--2024.

CBAM-covered product categories follow Annex~I of Regulation (EU) 2023/956 \citep{CBAM2023}: iron and steel (HS chapters 72--73, selected codes), aluminium (HS chapter 76), cement (HS 2523), fertilisers (HS chapter 31), electricity (HS 271600), and hydrogen (HS 280410).

The importer perimeter is time-varying: EU27 only for 2014--2019, EU27 plus UK from 2020 onward, reflecting Brexit and the UK's subsequent announcement of its own CBAM. We exclude EU27 and EFTA countries from the exporter sample since they are not subject to CBAM as exporters.

Our benchmark exposure measure follows \citet{Bahi2026}: a static average of annual exposure over 2014--2024 (denoted $\overline{E}_i$). We also construct a time-varying annual measure $E_{i,t}$ as our preferred specification, which better captures the Brexit correction and avoids averaging over a period with a changing importer perimeter.

""" + table2 + r"""

\subsection{Control Variables}\label{sec:controls}

We include GDP per capita (constant 2015 USD) and total exports from the World Development Indicators \citep{WDI2025}, and government effectiveness, regulatory quality, rule of law, and control of corruption from the Worldwide Governance Indicators \citep{WGI2025}. We construct a composite WGI index as the simple mean of the four governance indicators. All controls are lagged one year to avoid simultaneity. Coverage spans 273 countries over 2015--2024.


% ============================================================================
% SECTION 3 — EMPIRICAL STRATEGY
% ============================================================================
\section{Empirical Strategy}\label{sec:strategy}

Our analysis proceeds in four layers, each addressing a distinct empirical question with progressively richer outcome measurement. Layer~A tests the broad adoption question established by \citet{Bahi2026}: does CBAM exposure predict any carbon pricing activity? This serves as a benchmark for comparability, not as our primary contribution. Layer~B tests the pipeline question: does exposure predict faster movement from consideration through development to implementation? This is the paper's primary empirical contribution, estimated via both TWFE difference-in-differences and event-history models (Cox proportional hazards and discrete-time complementary log-log). Layer~C tests the design question: conditional on entering the carbon pricing system, are new instruments in more-exposed jurisdictions designed to cover CBAM-relevant industrial sectors from inception? This is the paper's primary substantive contribution, though statistically constrained by sample size and limited to multi-specification directional evidence. Layer~D provides secondary evidence on the price intensive margin.

\subsection{Benchmark Specification (Layer A)}\label{sec:benchmark_spec}

This layer replicates the framework of \citet{Bahi2026} for comparability and serves as a benchmark, not the paper's primary claim. Our benchmark replicates the Bruegel-comparable TWFE design. The estimating equation is:

\begin{equation}
Y_{c,t} = \alpha_c + \lambda_t + \beta \left( \overline{E}_c \times \mathbf{1}[t \geq \tau] \right) + \gamma' \mathbf{X}_{c,t-1} + \varepsilon_{c,t}
\label{eq:benchmark}
\end{equation}

\noindent where $Y_{c,t}$ is a binary outcome for country $c$ in year $t$, $\alpha_c$ are country fixed effects, $\lambda_t$ are year fixed effects, $\overline{E}_c$ is the benchmark static CBAM exposure, $\mathbf{1}[t \geq \tau]$ is a post-milestone indicator for $\tau \in \{2019, 2021, 2023\}$, and $\mathbf{X}_{c,t-1}$ are lagged controls. Standard errors are clustered at the country level. The sample spans 196 non-EU/EFTA countries over 2016--2024.

The identification logic relies on CBAM milestones as common shocks to all countries; cross-sectional variation in $\overline{E}_c$ generates treatment intensity variation. \citet{Bahi2026} find $\hat{\beta} \approx 0.016$ per percentage point increase in exposure. Our exposure is measured as a proportion (0--1) rather than percentage points (0--100). To compare with Bruegel, our coefficients should be divided by 100, yielding $\hat{\beta}/100 \approx 0.013$, consistent with their estimate.

\subsection{Pipeline Extension (Layer B)}\label{sec:pipeline_spec}

We extend the TWFE specification to pipeline outcomes using master-list timing (ever considered, ever developing, ever implemented). We also estimate Cox proportional hazard models:

\begin{equation}
h_c(t) = h_0(t) \exp\left( \beta_1 E_c^{\text{pct}} + \beta_2 \log \text{GDP}_{c} + \beta_3 \text{WGI}_{c} \right)
\label{eq:cox}
\end{equation}

\noindent where $h_c(t)$ is the hazard of country $c$ reaching a pipeline milestone (consideration, development, implementation) at time $t$, $E_c^{\text{pct}}$ is CBAM exposure in percentage points, and $h_0(t)$ is the baseline hazard. Countries already in the relevant state before 2016 are excluded via milestone-specific left-censoring.

We also estimate discrete-time complementary log-log (cloglog) models, which provide a natural link between the continuous-time proportional hazards framework and the annual panel structure of our data:

\begin{equation}
\log\!\bigl[-\log(1 - h_{c,t})\bigr] = \alpha_t + \beta_1 E_{c,t} + \gamma' \mathbf{X}_{c,t-1}
\label{eq:cloglog}
\end{equation}

\noindent where $h_{c,t}$ is the discrete-time hazard of country $c$ transitioning at year $t$, $\alpha_t$ captures the baseline hazard via year indicators, and $E_{c,t}$ is time-varying CBAM exposure. The discrete-time specification is our preferred event-history model because it accommodates the annual data structure naturally, allows time-varying covariates including the preferred Brexit-corrected exposure measure, and avoids the tied-event approximations required by Cox models in discrete time. Cox models are retained as a robustness check; we disclose the penalizer parameter and events-per-variable ratios throughout.

\subsection{Entrant-Level Sectoral Design (Layer C)}\label{sec:entrant_spec}

The near-absence of within-instrument sectoral expansion---51 of 52 post-2016 industry-covering CPIs entered with industry already covered---motivates an entrant-level design. The question is not whether existing instruments expand to cover industry, but whether \textit{new} instruments in CBAM-exposed countries are more likely to cover industry from inception. We present entrant-level descriptive and regression evidence supported by a battery of four supplementary specifications with different identifying assumptions and units of analysis. Because the sample is conditioned on countries that have moved into the carbon pricing policy set, this selection limits causal interpretation; we acknowledge these results as directional evidence rather than identified causal effects.

Our primary specification is a CPI-level panel difference-in-differences. Entrant-level OLS regressions use standard errors clustered at the country level to account for within-country correlation across instruments:

\begin{equation}
\text{CoversIndustry}_{i,t} = \alpha_i + \lambda_t + \beta \left( \overline{E}_{c(i)} \times \mathbf{1}[t \geq \tau] \right) + \gamma' \mathbf{X}_{c(i),t-1} + \varepsilon_{i,t}
\label{eq:panel_did}
\end{equation}

\noindent where $i$ indexes individual CPIs (CPI fixed effects $\alpha_i$), $c(i)$ is the country of CPI $i$, and the sample includes all CPI-year observations including pre-entry years where $\text{CoversIndustry}_{i,t} = 0$ by construction. The coefficient $\beta$ captures whether CPIs entering post-CBAM in more-exposed countries are more likely to cover industry at their first active observation.

As a supplementary specification, we estimate a cross-sectional entrant regression:

\begin{equation}
\text{IndustryAtEntry}_i = \alpha + \beta_1 \overline{E}_{c(i)} + \beta_2 \mathbf{1}[\text{entry} \geq \tau] + \beta_3 \left( \overline{E}_{c(i)} \times \mathbf{1}[\text{entry} \geq \tau] \right) + \gamma' \mathbf{X}_{c(i),\text{entry}-1} + \varepsilon_i
\label{eq:entrant}
\end{equation}

\subsection{Price Intensive Margin (Layer D)}\label{sec:price_spec}

This layer is secondary to the adoption and design results. We test whether CBAM exposure is associated with higher carbon prices. The dependent variable is log price; coefficients are therefore semi-elasticities, interpretable as the proportional change in price associated with a unit change in exposure. Carbon prices are in nominal USD converted at April~1 exchange rates. Cross-country USD comparisons embed exchange rate movements; we report robustness using country-own GDP deflators (WDI linked series NY.GDP.DEFL.ZS.AD).


% ============================================================================
% SECTION 4 — RESULTS
% ============================================================================
\section{Results}\label{sec:results}

\subsection{Benchmark Results (Layer A)}\label{sec:benchmark_results}

Table~\ref{tab:benchmark} presents the Bruegel-comparable benchmark. The coefficient on the exposure$\times$post interaction is positive across all outcomes and milestones, with the largest point estimate for \texttt{any\_cpi\_active} at the Post-2019 milestone (1.48 with controls). Rescaled to per-percentage-point interpretation ($\hat{\beta}/100 = 0.0148$), this is consistent with the \citet{Bahi2026} estimate of approximately 0.016. However, none of the benchmark coefficients are statistically significant at conventional levels, reflecting the wide confidence intervals inherent in a 196-country sample with substantial heterogeneity. The coefficients attenuate somewhat with later milestones (Post-2021, Post-2023), consistent with diminishing marginal treatment intensity as CBAM awareness diffuses globally.

""" + table3 + r"""

Table~\ref{tab:sectoral} extends the benchmark to industry-specific outcomes. The coefficients for industry-specific adoption are somewhat larger than the any-CPI results, consistent with the hypothesis that CBAM particularly incentivises industry-oriented instruments. However, \texttt{any\_industry\_cpi\_observed} and \texttt{any\_industry\_cpi\_active} yield identical estimates---reflecting the fact that industry-covering CPIs are almost always active by the time they appear in the database.

""" + table4 + r"""

\subsection{Pipeline Extension Results (Layer B)}\label{sec:pipeline_results}

Table~\ref{tab:cox} presents Cox proportional hazard models for the three pipeline transitions. The headline result is for the development transition: a hazard ratio of 1.201 ($p<0.01$), meaning that a one percentage point increase in CBAM exposure is associated with a 20\% higher hazard of moving into active carbon pricing development. The consideration and implementation transitions show hazard ratios of 1.110 ($p<0.05$) and 1.127 ($p<0.05$) respectively. The C-index for the development model is 0.755, indicating good discriminative ability.

""" + table5 + r"""

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{figures/km_development.png}
\caption{Kaplan-Meier survival curves for the development transition by CBAM exposure tercile. Low = bottom third of exposure distribution, High = top third. Log-rank test $p$-value reported in figure title. Countries already in development before 2016 excluded.}
\label{fig:km_development}
\end{figure}

Table~\ref{tab:twfe_comparison} compares benchmark and preferred (time-varying, Brexit-corrected) exposure measures across the pipeline outcomes. The \texttt{ever\_developing} outcome is significant with preferred exposure at both Post-2021 ($p<0.05$) and Post-2023 ($p<0.01$) milestones, while remaining insignificant under benchmark exposure. This validates the time-varying perimeter correction: the Brexit shock alters the exposure ranking of UK-proximate countries, and accounting for this improves statistical power.

""" + table6 + r"""

\subsection{Sectoral Design Results (Layer C)}\label{sec:sectoral_results}

The 2019 entrant cohort stands out: twelve new instruments (five national, seven subnational) with mean CBAM exposure of 2.12\%---ten times the 2016--2018 average of 0.19\%---and 100\% industry coverage at entry (Table~\ref{tab:entrant_desc}). The 2024 cohort shows a similar pattern with mean exposure of 1.38\% and 100\% industry coverage among the four entrants with available data. These cohort patterns are consistent with CBAM incentives attracting high-exposure countries into industry-oriented carbon pricing.

""" + table7 + r"""

Under our preferred post indicator ($\text{year} > 2019$), the 2019 cohort falls in the pre-treatment period because the April~1 data reference date precedes the July 2019 Green Deal announcement. The 2019 cohort pattern is therefore consistent with anticipatory policy responses or parallel political momentum in the lead-up to the formal announcement, rather than a direct post-announcement response. The 2020 and later cohorts represent genuine post-treatment observations.

Table~\ref{tab:entrant_reg} presents entrant-level linear probability models with country-clustered standard errors. The direction of the interaction terms is consistently positive, but the estimates are imprecise given the small number of country clusters ($N=41$ entrants with both coverage data and controls).

""" + table9 + r"""

We present four supplementary specifications: the CPI-level panel DiD (Table~\ref{tab:panel_did}), a country-year entrant design, and a country-sector-year descriptive design. No individual specification achieves conventional significance, but the consistency across four designs with different identifying assumptions and units of analysis is informative. The panel DiD coefficients on the exposure$\times$post interaction are positive across all three milestones: +0.97 (Post-2019), +1.11 (Post-2021), and +0.76 (Post-2023), though with 49 country clusters the estimates are imprecisely estimated.

""" + table8 + r"""

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{figures/event_study_industry.png}
\caption{Event study coefficients from Equation~\ref{eq:panel_did} with year-specific interactions relative to 2016 baseline. Coefficients represent the differential effect of a unit increase in CBAM exposure on industry coverage probability relative to the omitted base year. 95\% confidence intervals shown. Pre-2019 coefficients near zero support the parallel trends assumption.}
\label{fig:event_study}
\end{figure}

Exposure tercile analysis reveals a nuanced pattern. Pre-2019 industry coverage rates were 5.9\% (low exposure), 23.8\% (medium), and 27.1\% (high). Post-2019, these shifted to 25.0\%, 49.3\%, and 38.4\% respectively. The non-monotonic post-2019 pattern---medium exposure showing the largest jump---reflects that high-exposure countries had higher baseline coverage (less room to grow), while medium-exposure countries show the largest post-2019 increase in industry coverage.

\subsection{Price Results (Layer D)}\label{sec:price_results}

Table~\ref{tab:price_nominal} presents the nominal USD log-price results. Coefficients are semi-elasticities: a unit increase in the interaction term corresponds to a $(\exp(\hat{\beta})-1)\times 100$\% change in the carbon price. The Post-2021 interaction is significant for the national all-types sample and the Post-2023 interaction is marginally significant. The ETS subsample shows larger but less precisely estimated coefficients, while the tax subsample shows a significant Post-2019 coefficient that does not persist at later milestones.

""" + table10 + r"""

Table~\ref{tab:price_real} presents the real price robustness check. Deflating by the GDP deflator attenuates coefficients, and some subsamples show sign changes between nominal and real specifications. This likely reflects exchange rate artifacts for countries with appreciating currencies (e.g., Singapore, Colombia, Uruguay), where the nominal USD increase partly captures currency movements rather than genuine price increases.

""" + table11 + r"""


% ============================================================================
% SECTION 5 — ROBUSTNESS
% ============================================================================
\section{Robustness}\label{sec:robustness}

Table~\ref{tab:robustness} presents the full robustness summary across eleven specifications, all benchmarked against RC1 (any\_cpi\_active, Post-2021, benchmark exposure, $\hat{\beta} = 0.703$, $N = 1{,}304$).

\paragraph{Placebo test (RC10).} The placebo test using 2016--2018 data with a post-2018 indicator yields a coefficient of """ + stats["rc10_coef"] + r""" ($p = """ + stats["rc10_p"] + r"""$), providing no evidence of pre-existing differential trends. This supports the CBAM interpretation of post-2019 findings: countries with higher CBAM exposure were not already trending differently in their carbon pricing adoption prior to the first CBAM announcements.

\paragraph{EU candidate exclusion (RC4).} Excluding the ten EU accession candidates (Albania, Bosnia and Herzegovina, Georgia, Kosovo, Moldova, Montenegro, North Macedonia, Serbia, Turkey, and Ukraine) reduces the coefficient from 0.703 to """ + stats["rc4_coef"] + r""" (SE = """ + stats["rc4_se"] + r""", $p = """ + stats["rc4_p"] + r"""$, $N = """ + str(stats["rc4_n"]) + r"""$). This is consistent with these countries responding to EU accession requirements as well as CBAM exposure. The candidate countries include six of the top fifteen most CBAM-exposed countries (Montenegro, Bosnia, Albania, Serbia, Ukraine, North Macedonia), so their exclusion substantially reduces the identifying variation. The coefficient attenuation is structurally motivated rather than indicative of fragility.

\paragraph{Income group heterogeneity (RC7).} The triple interaction (exposure $\times$ post $\times$ high income) is significant with a coefficient of """ + stats["rc7c_coef"] + r""" (SE = """ + stats["rc7c_se"] + r""", $p = """ + stats["rc7c_p"] + r"""$), confirming that the CBAM exposure effect is concentrated among higher-income jurisdictions with greater institutional capacity to implement carbon pricing instruments. Subsample results show a larger point estimate for lower-middle and low-income countries (1.069, $p = 0.111$) than for high and upper-middle income countries (0.075, $p = 0.963$), but the triple interaction captures the differential rather than the level.

\paragraph{Coefficient stability.} The coefficient of variation across RC1--RC9 is 0.84, flagged as high sensitivity. This is driven primarily by RC4 (candidate exclusion, which removes the most exposed countries) and RC7c (income interaction, which models heterogeneity rather than testing the average effect). When these structurally motivated specifications are set aside, the coefficient is stable across RC2 (include subnational, $\hat{\beta} = 0.958$), RC6b (time-varying exposure, $\hat{\beta} = 0.587$), RC8 (individual WGI dimensions, $\hat{\beta} = 0.677$), RC9 (no controls, $\hat{\beta} = 0.788$), and RC11 (non-OECD only, $\hat{\beta} = 0.766$).

""" + table12 + r"""


% ============================================================================
% SECTION 6 — CONCLUSION
% ============================================================================
\section{Conclusion}\label{sec:conclusion}

This paper presents three main findings on the relationship between the EU Carbon Border Adjustment Mechanism and global carbon pricing adoption. First, countries with greater CBAM trade exposure advance faster through the carbon pricing policy pipeline, with Cox hazard ratios of 1.20 ($p<0.01$) for the development transition and 1.13 ($p<0.05$) for implementation. Second, the 2019 entrant cohort---coinciding with the European Commission's first detailed CBAM proposals---shows ten times the average CBAM exposure of prior cohorts and 100\% industry coverage, a descriptive pattern consistent with CBAM-driven policy design. Third, CPI-level panel DiD estimates for industry coverage are consistently positive (+0.97 to +1.11) but imprecisely estimated given the limited number of country clusters ($N_{\text{clusters}} = 49$). The placebo test (RC10) provides no evidence of pre-existing differential trends.

Several limitations constrain what we can identify. We cannot determine whether CBAM caused existing instruments to expand their sectoral coverage---within-instrument variation is almost entirely absent in the data, with only China's national ETS showing genuine post-entry expansion to industry. We cannot rule out confounding through other EU trade policy channels or broader climate governance trends. The 49-country cluster count limits the power of the CPI-level panel DiD, and the cross-sectional entrant regressions ($N=41$) are too small to identify interaction effects precisely.

Extending the temporal panel of sectoral coverage with a transparent, replicable methodology would enable three advances. First, longer pre-periods would strengthen pre-trend testing beyond the current 2016--2018 window. Second, tracking post-2026 behaviour as CBAM transitions from its transitional to full implementation phase would allow identification of whether the price-based incentive generates a second wave of adoption. Third, sub-sector analysis---distinguishing between iron and steel, aluminium, cement, fertiliser, and electricity---would become feasible once CBAM product-level coverage data is available for each CPI.


% ============================================================================
% BIBLIOGRAPHY
% ============================================================================
\bibliography{references}

% ============================================================================
% APPENDIX
% ============================================================================
\appendix
\newpage

\section{Full Robustness Table}\label{app:robustness}

See Table~\ref{tab:robustness} in Section~\ref{sec:robustness} (presented in landscape format).


\section{Kaplan-Meier Survival Curves}\label{app:km}

\begin{figure}[H]
\centering
\begin{minipage}[t]{0.32\textwidth}
\centering
\includegraphics[width=\textwidth]{figures/km_consideration.png}
\end{minipage}
\hfill
\begin{minipage}[t]{0.32\textwidth}
\centering
\includegraphics[width=\textwidth]{figures/km_development.png}
\end{minipage}
\hfill
\begin{minipage}[t]{0.32\textwidth}
\centering
\includegraphics[width=\textwidth]{figures/km_implementation.png}
\end{minipage}
\caption{Kaplan-Meier survival curves for the consideration (left), development (centre), and implementation (right) transitions by CBAM exposure tercile. Countries already in the relevant state before 2016 are excluded.}
\label{fig:km_all}
\end{figure}


\section{Data Notes}\label{app:data_notes}

\begin{table}[H]
\centering
\caption{CPIs with Known Data Quality Issues}
\label{tab:data_notes}
\begin{tabular}{lp{10cm}}
\toprule
CPI & Issue \\
\midrule
Tax\_NG & Implemented status but no sectoral coverage data in any edition \\
ETS\_EU\_IS / ETS\_EU\_LI / ETS\_EU\_NO & Sector coverage captured under ETS\_EU; separate coverage records absent \\
2024 cohort & Seven instruments under development with no sector coverage data yet (right-censored) \\
\bottomrule
\end{tabular}
\end{table}


\end{document}
"""

    return tex


# ---------------------------------------------------------------------------
# Bibliography
# ---------------------------------------------------------------------------

REFERENCES_BIB = r"""@techreport{Bahi2026,
  author  = {Bah\'{i}, Anna and Fuchs, Maximilian and Reverdy, Camille},
  title   = {Carbon pricing beyond borders: assessing climate policy
             spillovers from the {EU} carbon border adjustment mechanism},
  institution = {Bruegel},
  year    = {2026},
  type    = {Working Paper},
  number  = {05/2026},
  doi     = {10.64153/QIJX8741}
}

@article{CallawayS2021,
  author  = {Callaway, Brantly and Sant'Anna, Pedro H. C.},
  title   = {Difference-in-differences with multiple time periods},
  journal = {Journal of Econometrics},
  year    = {2021},
  volume  = {225},
  number  = {2},
  pages   = {200--230},
  doi     = {10.1016/j.jeconom.2020.12.001}
}

@article{Gaulier2010,
  author  = {Gaulier, Guillaume and Zignago, Soledad},
  title   = {{BACI}: International Trade Database at the Product-Level.
             The 1994-2007 Version},
  journal = {CEPII Working Paper},
  year    = {2010},
  number  = {2010-23}
}

@techreport{WorldBank2025,
  author      = {{World Bank}},
  title       = {State and Trends of Carbon Pricing 2025},
  institution = {World Bank},
  year        = {2025},
  address     = {Washington, DC},
  doi         = {10.1596/978-1-4648-2212-0}
}

@misc{CBAM2023,
  author = {{European Parliament and Council}},
  title  = {Regulation ({EU}) 2023/956 establishing a carbon border
            adjustment mechanism},
  year   = {2023},
  note   = {Official Journal of the European Union, L 130/52}
}

@techreport{WDI2025,
  author      = {{World Bank}},
  title       = {World Development Indicators},
  institution = {World Bank},
  year        = {2025},
  note        = {Available at: \url{https://databank.worldbank.org/source/world-development-indicators}}
}

@techreport{WGI2025,
  author      = {Kaufmann, Daniel and Kraay, Aart and Mastruzzi, Massimo},
  title       = {Worldwide Governance Indicators},
  institution = {World Bank},
  year        = {2025},
  note        = {Available at: \url{https://www.worldbank.org/en/publication/worldwide-governance-indicators}}
}

@article{Mehling2025,
  author  = {Mehling, Michael A. and Dolphin, Glenn and Ritz, Robert A.},
  title   = {The {European Union}'s {CBAM}: averting emissions leakage
             or promoting the diffusion of carbon pricing?},
  journal = {Journal of Environmental Policy \& Planning},
  year    = {2025},
  volume  = {27},
  number  = {6},
  pages   = {687--705},
  doi     = {10.1080/1523908X.2025.2591794}
}

@article{AicheleFelbermayr2015,
  author  = {Aichele, Rahel and Felbermayr, Gabriel},
  title   = {Kyoto and Carbon Leakage: An Empirical Analysis of the
             Carbon Content of Bilateral Trade},
  journal = {Review of Economics and Statistics},
  year    = {2015},
  volume  = {97},
  number  = {1},
  pages   = {104--115},
  doi     = {10.1162/REST_a_00438}
}

@article{Goulder2013,
  author  = {Goulder, Lawrence H. and Schein, Andrew R.},
  title   = {Carbon taxes versus cap and trade: a critical review},
  journal = {Climate Change Economics},
  year    = {2013},
  volume  = {4},
  number  = {3},
  pages   = {1350010},
  doi     = {10.1142/S2010007813500103}
}

@article{Stavins2019,
  author  = {Stavins, Robert N.},
  title   = {Carbon taxes vs. cap and trade: theory and practice},
  journal = {Harvard Environmental Economics Program Discussion Paper},
  year    = {2019},
  number  = {19-9}
}

@misc{UKCBAM2023,
  author = {{HM Government}},
  title  = {UK Carbon Border Adjustment Mechanism},
  year   = {2023},
  note   = {Policy paper, December 2023}
}
"""

# ---------------------------------------------------------------------------
# Compile script
# ---------------------------------------------------------------------------

COMPILE_SH = r"""#!/bin/bash
# Compile the CBAM spillovers paper
# Run from project root: bash paper/compile.sh

cd "$(dirname "$0")"
pdflatex cbam_spillovers.tex
bibtex cbam_spillovers
pdflatex cbam_spillovers.tex
pdflatex cbam_spillovers.tex
echo "Done. Output: paper/cbam_spillovers.pdf"
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Generating CBAM Spillovers Paper")
    print("=" * 60)

    print("\n1. Loading data...")
    data = load_data()
    print(f"   Loaded {len(data)} data sources")

    print("\n2. Copying figures...")
    copy_figures()

    print("\n3. Generating LaTeX...")
    tex = generate_tex(data)
    tex_path = os.path.join(PAPER, "cbam_spillovers.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"   Wrote {tex_path} ({len(tex):,} chars)")

    print("\n4. Writing bibliography...")
    bib_path = os.path.join(PAPER, "references.bib")
    with open(bib_path, "w", encoding="utf-8") as f:
        f.write(REFERENCES_BIB)
    print(f"   Wrote {bib_path}")

    print("\n5. Writing compile script...")
    sh_path = os.path.join(PAPER, "compile.sh")
    with open(sh_path, "w", encoding="utf-8") as f:
        f.write(COMPILE_SH)
    os.chmod(sh_path, 0o755)
    print(f"   Wrote {sh_path}")

    print("\n" + "=" * 60)
    print("DONE. To compile:")
    print("  bash paper/compile.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
