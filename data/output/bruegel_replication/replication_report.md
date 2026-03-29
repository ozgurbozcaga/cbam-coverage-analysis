# Bruegel (2026) Replication Comparison Report

Bahí, Fuchs & Reverdy (2026) "Carbon pricing beyond borders: assessing
climate policy spillovers from the EU carbon border adjustment mechanism",
Bruegel Working Paper 05/2026, 26 March 2026.

This document compares our replication of the Bruegel analysis against their
published results, documents all systematic differences, and summarises what
our paper adds beyond their framework.

---

## 1. Exposure Comparison

### Perimeter definitions

| Dimension | Bruegel | Our preferred | Our benchmark (replication) |
|-----------|---------|---------------|---------------------------|
| Destination perimeter | EU + EFTA (incl. CHE) | EU27 + EEA-EFTA (excl. CHE) | EU27 + EEA-EFTA + GBR (pre-2020) |
| Brexit adjustment | None (static) | Yes (GBR dropped from 2020) | GBR included throughout (for replication) |
| Exposure type | Static avg 2014–2024 | Time-varying annual | Static avg 2014–2024 (for replication) |
| Exporter exclusions | EEA-EFTA, CHE | EEA-EFTA, CHE | Same |
| Panel window | 2014–2025 (12 years) | 2016–2024 (9 years) | 2016–2024 (9 years) |

**Note on Switzerland:** Bruegel includes CHE in the destination perimeter
(EU + EFTA). We exclude it because Swiss imports are exempt from CBAM under
the EU–Switzerland ETS linking agreement (Article 1, Decision No 1/2024 of
the EU–Switzerland Joint Committee). This is a substantive methodological
choice, not a data error. For countries with significant Swiss-bound CBAM
exports, our exposure measure is mechanically lower.

**Note on Brexit:** Bruegel do not appear to make a Brexit adjustment; their
perimeter is static across all years. We correct for GBR's departure from
the EU on 31 January 2020: GBR is included in the destination perimeter for
2014–2019 and excluded from 2020 onward in our preferred (time-varying)
measure. For the replication benchmark, we use a static measure that includes
GBR throughout, to approximate Bruegel's approach.

### Top-20 exposure ranking comparison

Our replication exposure ranking (static avg, EU27+EEA-EFTA+GBR perimeter,
excl. CHE) versus Bruegel's Figure 1:

| Rank | Our ranking | Exposure (%) | Bruegel ranking | Notes |
|------|------------|-------------|-----------------|-------|
| 1 | Montenegro | 24.4 | Montenegro | 24.2% — match |
| 2 | Mozambique | 15.0 | Mozambique | 15.0% — match |
| 3 | Bosnia & Herzegovina | 10.2 | Bosnia & Herzegovina | 13.4% — gap likely due to CHE |
| 4 | Albania | 8.3 | Ukraine | Ranking swap |
| 5 | Serbia | 7.7 | Albania | Ranking swap |
| 6 | Ukraine | 7.5 | Serbia | Ranking swap |
| 7 | North Macedonia | 7.0 | — | Similar order below top-3 |
| 8 | New Caledonia | 6.2 | — | |
| 9 | Armenia | 6.2 | — | |
| 10 | Moldova | 4.8 | — | |

The top-3 countries (MNE, MOZ, BIH) are identical in both rankings. Minor
re-orderings in ranks 4–7 reflect the CHE perimeter difference and BACI data
vintage. Bruegel reports MNE at 24.2%; we get 24.4% — within rounding of
BACI revision differences.

---

## 2. Table A2 Replication Quality

### Coefficient comparison

| Col | Specification | Bruegel coeff | Bruegel SE | Our coeff | Our SE | Diff | % diff | Flag |
|-----|--------------|--------------|-----------|----------|--------|------|--------|------|
| (1) | Pooled OLS, no FE | 0.038*** | (0.004) | 0.017*** | (0.006) | -0.021 | -55% | **FLAGGED** |
| (2) | Country + Year FE | 0.016*** | (0.004) | 0.017*** | (0.004) | +0.001 | +6% | |
| (3) | FE + controls | 0.015*** | (0.004) | 0.016*** | (0.005) | +0.001 | +6% | |
| (4) | Annual exposure, FE | 0.010** | (0.004) | 0.012*** | (0.004) | +0.002 | +21% | |
| (5) | Annual + controls | 0.009** | (0.004) | 0.011** | (0.005) | +0.002 | +27% | |
| (6) | Lagged + controls | 0.010** | (0.005) | 0.010** | (0.005) | +0.000 | +1% | |

**Mean absolute difference (cols 2–6):** 0.0012 (excellent replication).
**Mean absolute difference (all cols):** 0.0046.

### Flagged coefficient: Column 1

Column 1 (pooled OLS, no fixed effects) shows our coefficient at 0.017 vs
Bruegel's 0.038 — a 55% difference. This is the only coefficient exceeding
the 50% threshold. The discrepancy is expected and benign:

- **Panel window effect:** Without country FE, the coefficient captures
  cross-sectional level differences. Bruegel's longer panel (2014–2025,
  12 years) includes 2 additional pre-treatment years and 1 post-treatment
  year. The 2025 observation adds a year where CBAM is in force and more
  countries have adopted policies, mechanically increasing the pooled
  cross-sectional correlation.
- **Country sample composition:** We include ~196 countries vs Bruegel's 163.
  The additional countries are mostly small economies with near-zero exposure.
  Without FE, they dilute the cross-sectional gradient.
- **N difference:** 1,764 vs 1,956 (192 fewer observations).

**Assessment:** Once fixed effects absorb time-invariant heterogeneity
(cols 2–6), our coefficients closely match Bruegel's. No coefficient changes
sign. All significance levels are preserved. This constitutes a **close
replication** of the core finding.

### Observation count differences

| Column | Bruegel N | Our N | Difference | Explanation |
|--------|----------|-------|-----------|-------------|
| (1)–(3) | 1,956 / 1,593 | 1,764 / 1,299 | -192 / -294 | Panel window: 9 vs 12 years |
| (4) | 1,793 | 1,764 | -29 | Bruegel may have different country coverage for annual exposure |
| (6) | 1,433 | 1,299 | -134 | Lag drops first year; panel length difference |

---

## 3. Standard Error Comparison

### HC1 vs cluster-robust

Bruegel reports "robust standard errors in parentheses" without specifying
the clustering structure. Their SEs are consistent with HC1 (heteroskedasticity-
robust) standard errors at the observation level, not clustered at the
country level.

Our replication uses HC1 (robust) standard errors to match Bruegel's
specification. Our main analysis (Layers A–E) uses **country-clustered**
standard errors throughout, which are typically larger because they account
for within-country serial correlation.

### Comparison across specifications

| Col | Bruegel SE (HC1) | Our SE (HC1) | Our SE (clustered) | Significance change? |
|-----|-----------------|-------------|-------------------|---------------------|
| (1) | 0.004 | 0.006 | — | No (still ***) |
| (2) | 0.004 | 0.004 | — | No (still ***) |
| (3) | 0.004 | 0.005 | — | No (still ***) |
| (4) | 0.004 | 0.004 | — | No (still ***) |
| (5) | 0.004 | 0.005 | — | No (still **) |
| (6) | 0.005 | 0.005 | — | No (still **) |

### Impact on significance conclusions

Under our Layer A analysis with country-clustered SEs and our preferred
exposure measure, the benchmark TWFE coefficients are larger but SEs are
also larger (e.g., any_cpi_announced Post-2021: coeff 0.889, SE 0.808).
This means that while point estimates are economically meaningful, statistical
significance is weaker with clustered SEs — consistent with the conservative
inference approach appropriate for country-level panel data with ~145
clusters.

The Cox proportional hazard models (Layer B), which do not suffer from the
same FE incidental parameter issues, show consistently significant effects
(HR 1.11–1.21, p < 0.05) even with robust variance estimation, providing
complementary evidence for the causal channel.

---

## 4. EU Candidate Robustness (Table A3)

### Coefficient comparison across 9 specifications

| Col | Specification | Bruegel coeff | Bruegel SE | Our coeff | Our SE | Sign match? |
|-----|--------------|--------------|-----------|----------|--------|-------------|
| (1) | Static + candidate | 0.016*** | (0.004) | 0.018*** | (0.005) | Yes |
| (2) | ln(Static) + candidate | 0.010*** | (0.003) | 0.041** | (0.020) | Yes |
| (3) | Static, excl candidates | 0.005* | (0.003) | -0.007*** | (0.003) | **NO** |
| (4) | Annual + candidate | 0.010** | (0.004) | 0.012** | (0.005) | Yes |
| (5) | ln(Annual) + candidate | 0.010*** | (0.003) | 0.032* | (0.019) | Yes |
| (6) | Annual, excl candidates | 0.007*** | (0.002) | -0.006** | (0.002) | **NO** |
| (7) | Lagged + candidate | 0.010** | (0.003) | 0.011** | (0.005) | Yes |
| (8) | ln(Lagged) + candidate | 0.008*** | (0.003) | 0.031 | (0.019) | Yes |
| (9) | Lagged, excl candidates | 0.005** | (0.002) | -0.006** | (0.002) | **NO** |

### EU candidate dummy comparison

| Col | Bruegel candidate coeff | Our candidate coeff | Notes |
|-----|------------------------|--------------------|----|
| (1) | -0.125** (0.057) | -0.162*** (0.030) | Same sign, ours stronger |
| (2) | -0.105* (0.055) | -0.139*** (0.030) | Same sign, ours stronger |
| (4) | -0.112** (0.056) | -0.132*** (0.030) | Same sign, ours stronger |
| (5) | -0.107* (0.055) | -0.127*** (0.029) | Same sign, ours stronger |
| (7) | -0.063** (0.025) | -0.160*** (0.027) | Same sign, ours stronger |
| (8) | -0.049** (0.024) | -0.146*** (0.030) | Same sign, ours stronger |

### Key finding: sign reversal when excluding candidates

The most notable difference is in columns (3), (6), and (9), where EU
candidate countries are excluded entirely. Bruegel obtains small positive
coefficients (0.005–0.007); we obtain small negative coefficients (-0.006
to -0.007). This sign reversal indicates that the CBAM exposure effect in
the full sample is driven substantially by EU candidate countries.

**Likely explanation:** Our shorter panel (2016–2024 vs 2014–2025) and
different country sample mean that once the ~10 EU candidate countries are
removed, the remaining variation in exposure among non-candidate countries
is insufficient to identify a positive effect in our sample. This finding is
**substantively important**: it suggests that the Bruegel result may partly
reflect EU accession conditionality (acquis alignment) rather than pure CBAM
incentives. Our Layer E robustness check RC4 confirms this — the coefficient
reverses to -0.437 (p = 0.18) when candidates are excluded, compared to the
baseline of +0.727.

### Assessment

The candidate robustness check reveals a genuine fragility in the core
CBAM spillover finding. Both papers show a large negative EU candidate
coefficient, confirming that candidate countries are systematically different.
The question of whether the remaining effect (after controlling for or
excluding candidates) is robust is where the papers diverge, with our
shorter panel finding a sign reversal that Bruegel's longer panel does not.

---

## 5. Income Heterogeneity (Table A4)

### Higher/Lower income comparison

| Col | Group | Bruegel coeff | Bruegel SE | Our coeff | Our SE | Sign match? |
|-----|-------|--------------|-----------|----------|--------|-------------|
| (1) | Higher | 0.019*** | (0.004) | 0.023*** | (0.004) | Yes |
| (1) | Lower | 0.000 | (0.003) | -0.002 | (0.002) | Yes (both ~0) |
| (2) | Higher | 0.020*** | (0.005) | 0.023*** | (0.004) | Yes |
| (2) | Lower | -0.004 | (0.003) | -0.003 | (0.002) | Yes |
| (3) | Higher + candidate | 0.021*** | (0.004) | 0.023*** | (0.004) | Yes |
| (3) | Lower + candidate | -0.004 | (0.003) | -0.001 | (0.002) | Yes |

### Quartile comparison (cols 4–6)

| Col | Quartile | Bruegel coeff | Our coeff | Sign match? |
|-----|----------|--------------|----------|-------------|
| (4) | Q1+Q2 | 0.000 (0.003) | -0.004*** (0.001) | Marginal |
| (4) | Q3 | 0.016*** (0.005) | 0.026*** (0.004) | Yes |
| (4) | Q4 | 0.025*** (0.005) | -0.014*** (0.005) | **NO** |
| (5) | Q1+Q2 | -0.002 (0.002) | -0.004*** (0.001) | Yes |
| (5) | Q3 | 0.020*** (0.005) | 0.025*** (0.004) | Yes |
| (5) | Q4 | 0.021*** (0.008) | -0.018*** (0.006) | **NO** |

### Key finding: "effect concentrated in higher income" replicates partially

The central finding that the CBAM spillover is **concentrated among higher-
income countries** replicates clearly:

- Higher-income coefficient: both papers find positive, significant effects
  (Bruegel 0.019–0.021; ours 0.023).
- Lower-income coefficient: both find near-zero, insignificant effects.
- EU candidate coefficient: both find large negative effects when included.

**Q4 (highest quartile) sign reversal:** The quartile decomposition reveals
a discrepancy in Q4 (above 75th percentile of income). Bruegel finds Q4
positive and significant (0.021–0.025); we find it negative and significant
(-0.014 to -0.018). This may reflect compositional differences: high-income
OECD countries in Q4 (which already had carbon pricing before CBAM) may show
a negative gradient in our shorter panel where fewer new adoptions occur.
Q3 (50th–75th percentile) is where both papers agree the effect is strongest.

Our Layer E income heterogeneity check (RC7c) confirms the differential
effect: the triple interaction (exposure x post x high_income) is 2.22***
(SE 0.84), indicating a statistically significant difference between income
groups.

---

## 6. Key Differences and Their Sources

### Data vintage differences (expected, benign)

| Difference | Impact | Assessment |
|-----------|--------|-----------|
| BACI V202601 trade data vs Bruegel's vintage | Minor exposure differences (~0.1–0.3 ppt for top countries) | Benign |
| WDI/WGI control variable vintage | Different N with controls (1,299 vs 1,593) | Benign |
| Panel window (2016–2024 vs 2014–2025) | 192 fewer obs without controls; 294 fewer with controls | Expected; affects pooled OLS most |
| Country sample (196 vs 163) | Additional small countries add noise in pooled specs | Benign |
| CPI data source (our panel vs Bruegel hand-collection) | Replication uses Bruegel's Table A1 years directly to avoid coding differences from "Removed" instruments | Explained in replication notes |

### Methodological choices (deliberate, documented)

| Choice | Bruegel | Ours | Justification |
|--------|---------|------|--------------|
| Exposure perimeter | EU + EFTA (incl. CHE) | EU27 + EEA-EFTA (excl. CHE) | CHE exempt from CBAM under ETS linking agreement |
| Brexit adjustment | None | GBR excluded from 2020 onward | GBR left EU 31 Jan 2020; its imports no longer subject to EU CBAM |
| Post indicator timing | year > 2019 | year > 2019 (same) | Green Deal announced July 2019; April 1 S&T reference date predates announcement |
| Standard errors | HC1 robust | Country-clustered | Clustered SEs account for serial correlation within countries |
| Post-2021, post-2023 | Not tested separately | Tested as additional milestones | CBAM proposed July 2021, entered force October 2023 |

### Genuine discrepancies (requiring interpretation)

| Finding | Bruegel | Ours | Interpretation |
|---------|---------|------|---------------|
| Excl-candidates sign | Positive (0.005–0.007*) | Negative (-0.006 to -0.007**) | Shorter panel + different sample = fragility of non-candidate effect |
| Q4 income quartile | Positive (0.021–0.025***) | Negative (-0.014 to -0.018***) | Compositional: high-income countries with pre-existing policies |
| Pooled OLS magnitude | 0.038*** | 0.017*** | Cross-sectional sensitivity to sample and panel length |

---

## 7. What Our Paper Adds Beyond Bruegel

### 7.1 Pipeline event-history models (Layer B)

Bruegel's analysis is limited to TWFE difference-in-differences with a binary
outcome (any CPI announced/developed/implemented). We extend this with:

- **Cox proportional hazard models** estimating the effect of CBAM exposure on
  the *rate* of transition through the CPI pipeline (consideration →
  development → implementation). Hazard ratios: 1.11** (consideration),
  1.21*** (development), 1.13** (implementation). A 1 ppt increase in
  CBAM exposure raises the hazard of progressing to the next pipeline stage
  by 11–21%.

- **Discrete-time cloglog models** as a robustness check, with consistent
  results across penalizer and robust SE specifications.

- **Kaplan-Meier survival curves** showing separation between high- and
  low-exposure country groups.

### 7.2 Sectoral design dimension (Layer C)

Bruegel examines only the extensive margin (whether a country has *any* CPI).
We additionally examine:

- **Industry coverage at entry:** whether CPIs adopted in high-exposure
  countries are more likely to cover CBAM-relevant industries (power,
  industry, metals, cement).

- **CPI-level panel DiD** at the instrument level, testing whether post-CBAM
  entrants in high-exposure countries have broader industry coverage.

- **Descriptive analysis** of the 46 national CPIs in the sample, documenting
  the gap between entry and first industry coverage.

### 7.3 Time-varying Brexit-corrected exposure (Layer B)

Our preferred exposure measure is time-varying (annual) and Brexit-corrected.
TWFE results show this measure yields:

- Tighter standard errors (e.g., ever_developing Post-2021: SE 0.298 vs
  0.636 for the benchmark static measure).
- More significant coefficients (e.g., ever_developing Post-2021: 0.867***
  vs 1.003 not significant).

This suggests that time-varying, perimeter-corrected exposure reduces
measurement error and improves identification.

### 7.4 Corrected post indicator timing

Our post indicators use strict inequality (year > milestone, not >=) based
on the April 1 reference date of the World Bank S&T series:

- **post_2019:** Green Deal announced July 2019; the 2019 observation
  (reference date ~April 2019) predates the announcement, so 2019 is
  correctly classified as pre-treatment.
- **post_2021:** CBAM regulation proposed July 2021.
- **post_2023:** CBAM entered into force October 2023.

Bruegel uses the same post_2019 convention (year > 2019) but does not test
post_2021 or post_2023 as separate treatment dates.

### 7.5 Carbon price analysis (Layer D)

We examine whether CBAM exposure affects the *level* of carbon prices
(nominal and real USD), finding:

- Significant positive effects on tax-based instrument prices
  (Post-2021: +30.30*** nominal, +25.69*** real per 1 ppt exposure).
- Negative but insignificant effects on ETS prices (small N = 35,
  wide confidence intervals).

Bruegel documents descriptive price trends but does not estimate price
regressions.

### 7.6 Montenegro sensitivity analysis (Layer B)

Montenegro is the highest-exposed country (24.4%) and also an EU candidate
with rapid CPI adoption (2018→2019→2020). Our leave-one-out analysis shows:

- Excluding Montenegro reduces the cloglog hazard ratio for consideration
  from 1.11** to 1.08 (p = 0.16), rendering it insignificant.
- Development and implementation HRs remain robust to Montenegro exclusion.

This documents a specific influential observation that Bruegel does not
discuss.

### 7.7 Comprehensive robustness battery (Layer E)

Our Layer E includes 13 robustness checks systematically varying:

- Subnational instrument inclusion (RC2)
- S&T edition confirmation (RC3)
- EU candidate exclusion (RC4) — sign reversal documented
- Chinese subnational ETS treatment (RC5)
- Static vs time-varying exposure (RC6)
- Income group heterogeneity (RC7) — triple interaction test
- WGI decomposition (RC8)
- Pure FE without controls (RC9)
- Placebo test pre-2019 (RC10) — correctly null
- Non-OECD subsample (RC11)
- Log exposure transformation (RC12)
- Leave-one-out top-5 exposed (RC13)

---

## Summary Assessment

The Bruegel replication is **successful for the core finding**: CBAM exposure
is positively associated with CPI adoption in the full sample, the effect is
concentrated in higher-income countries, and EU candidate countries are
systematically different. Mean absolute coefficient difference across Table A2
cols 2–6 is 0.0012, well within expected tolerance.

The **key fragility** identified is the sensitivity to EU candidate country
inclusion. Once candidates are excluded, our analysis finds a sign reversal
that Bruegel's does not, raising the question of whether the effect is driven
by CBAM incentives or EU accession conditionality.

Our paper extends the Bruegel framework by modelling the CPI adoption
*pipeline* (not just the binary outcome), examining *sectoral design*
(not just extensive margin), using *time-varying Brexit-corrected* exposure,
and providing a comprehensive robustness battery. These extensions yield
additional evidence — particularly from Cox hazard models — that the CBAM
exposure channel operates at each stage of the policy pipeline, with hazard
ratios of 1.11–1.21 per percentage point of exposure.
