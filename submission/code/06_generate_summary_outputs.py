"""
06_generate_summary_outputs.py
汇总所有关键结果，验证输出文件完整性。
"""
from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parent.parent
REQUIRED = {
    "data": [
        "data/raw/hs300_daily_raw.csv",
        "data/processed/hs300_daily_clean.csv",
        "data/processed/hs300_returns.csv",
        "data/processed/hs300_garch_conditional_volatility.csv",
        "data/processed/hs300_garch_standardized_residuals.csv",
    ],
    "figures": [
        "figures/fig1_hs300_close.png",
        "figures/fig2_hs300_returns.png",
        "figures/fig3_hs300_squared_returns.png",
        "figures/fig4_return_histogram.png",
        "figures/fig5_garch_conditional_volatility.png",
        "figures/fig6_standardized_residuals.png",
        "figures/fig7_squared_standardized_residuals.png",
        "figures/fig8_acf_standardized_residuals.png",
        "figures/fig9_acf_squared_standardized_residuals.png",
    ],
    "tables": [
        "tables/table1_descriptive_stats.csv",
        "tables/table1_descriptive_stats.md",
        "tables/table2_tests.csv",
        "tables/table2_tests.md",
        "tables/table3_garch_main.csv",
        "tables/table3_garch_main.md",
        "tables/table4_model_comparison.csv",
        "tables/table4_model_comparison.md",
        "tables/table5_residual_diagnostics.csv",
        "tables/table5_residual_diagnostics.md",
        "tables/table6_extended_asymmetric_models.csv",
        "tables/table6_extended_asymmetric_models.md",
    ],
    "paper": [
        "assignment_final.tex",
        "assignment_final.pdf",
        "ai_usage_statement.md",
    ],
}


def main():
    all_ok = True
    for category, files in REQUIRED.items():
        missing = [f for f in files if not (PROJECT / f).exists()]
        if missing:
            print(f"[WARN] Missing {category}: {missing}")
            all_ok = False
        else:
            print(f"[OK]   {category}: {len(files)} files present")

    if all_ok:
        print("\n" + "=" * 60)
        print("All required output files present. Project is complete.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Some files are missing — run earlier scripts first.")
        print("=" * 60)

    # Print summary counts
    for category, files in REQUIRED.items():
        print(f"  {category}: {len(files)} expected files")


if __name__ == "__main__":
    main()
