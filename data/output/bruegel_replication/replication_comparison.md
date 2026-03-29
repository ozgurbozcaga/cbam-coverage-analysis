# Bruegel (2026) Replication Comparison

Comparing our replication against Bruegel's published results.

## Key differences from Bruegel

| Dimension | Bruegel | Our replication |
|-----------|--------|-----------------|
| Panel window | 2014–2025 (12 years) | 2016–2024 (9 years) |
| Countries | 163 | ~196 |
| Max N (no controls) | 1,956 | 1,764 |
| Max N (with controls) | 1,593 | 1,299 |
| Exposure perimeter | EU + EFTA (static, no Brexit adj) | Same |
| CPI data | Hand-collected from WB S&T series | Our cpi_panel.csv |
| Post indicator | year > 2019 | year > 2019 (same) |

## Table A2 — Main results

| Column | Bruegel coeff | Our coeff | Difference | Bruegel N | Our N |
|--------|-------------|----------|-----------|----------|------|
| col1 | 0.0380 (0.0040) | 0.0169*** (0.0060) | -0.0211 | 1,956 | 1,764 |
| col2 | 0.0160 (0.0040) | 0.0169*** (0.0044) | +0.0009 | 1,956 | 1,764 |
| col3 | 0.0150 (0.0040) | 0.0159*** (0.0047) | +0.0009 | 1,593 | 1,299 |
| col4 | 0.0100 (0.0040) | 0.0121*** (0.0040) | +0.0021 | 1,793 | 1,764 |
| col5 | 0.0090 (0.0040) | 0.0114** (0.0046) | +0.0024 | 1,593 | 1,299 |
| col6 | 0.0100 (0.0050) | 0.0101** (0.0046) | +0.0001 | 1,433 | 1,299 |

## DV construction

We use Bruegel's own Table A1 announcement/development/implementation years to construct the dependent variable. This avoids coding differences from 'removed' instruments in our CPI panel (e.g., BIH, GAB, GEO, MDA, MKD have instruments listed as 'Removed' in our CPI data that Bruegel excludes). Using our CPI panel directly produced coefficients ~50% smaller due to inflated baseline adoption rates.

## Likely sources of remaining divergence

1. **Panel window**: Bruegel uses 2014–2025 (12 years × 163 countries = 1,956 obs). We use 2016–2024 (9 years). This reduces our N and eliminates 2 pre-treatment years (2014–2015) and 1 post-treatment year (2025). The shorter pre-treatment window particularly affects the pooled OLS (Col 1) and event study pre-trends.

2. **Country sample**: We include ~196 countries vs Bruegel's 163. The additional countries are mostly small economies with near-zero CBAM exposure. They add noise but little signal to the FE specifications, which is why Cols 2–6 match well.

3. **Pooled OLS (Col 1)**: The large gap in Col 1 (0.017 vs 0.038) reflects cross-sectional differences. Without FE, the coefficient captures level differences across countries, which are sensitive to sample composition. The FE specifications (Cols 2–6) remove these level differences and match Bruegel closely.

4. **Exposure measure**: Both use EU+EFTA static perimeter. Minor differences from BACI data vintage (V202601) or HS6 product codes.

## Assessment

Mean absolute coefficient difference across Table A2: 0.0046
Overall: **Close replication** — coefficients within reasonable tolerance given panel window differences.
