#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python code/00_download_data.py
python code/01_prepare_returns.py
python code/02_descriptive_analysis.py
python code/03_tests.py
python code/04_garch_models.py
python code/05_diagnostics.py
python code/06_generate_summary_outputs.py
