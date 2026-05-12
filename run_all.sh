#!/usr/bin/env bash
set -e

python src/00_download_data.py
python src/01_prepare_returns.py
python src/02_descriptive_analysis.py
python src/03_tests.py
python src/04_garch_models.py
python src/05_diagnostics.py
python src/06_generate_summary_outputs.py
