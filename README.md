# CBAM Coverage Analysis
World Bank research project: EU CBAM spillovers on carbon pricing sectoral coverage.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data
BACI trade data accessed via OneDrive symlinks — not committed to git.
CPI lookup files in data/lookup/ are committed.

## Structure
- src/        Python scripts
- notebooks/  Jupyter notebooks
- data/lookup/   CPI list, prices, sectoral coverage
- data/output/   Generated datasets
