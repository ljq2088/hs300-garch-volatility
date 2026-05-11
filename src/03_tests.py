"""
03_tests.py
时间序列检验：ADF, Ljung-Box (ret & ret^2), ARCH-LM。
"""
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from tabulate import tabulate

PROJECT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT / "data" / "processed"
TABLES = PROJECT / "tables"


def load_returns():
    df = pd.read_csv(DATA_PROCESSED / "hs300_returns.csv", parse_dates=["date"])
    return df["ret"].values


def run_tests(ret):
    results = []
    ret = ret.copy()
    ret2 = ret ** 2

    # 1. ADF
    adf_res = adfuller(ret, autolag="AIC")
    results.append({
        "test_name": "ADF",
        "lag": adf_res[2],
        "statistic": adf_res[0],
        "p_value": adf_res[1],
        "interpretation": "stationary" if adf_res[1] < 0.05 else "non-stationary",
    })

    # 2. Ljung-Box on ret
    for lag in [10, 20]:
        lb = acorr_ljungbox(ret, lags=lag, return_df=True)
        row = lb.iloc[-1]
        results.append({
            "test_name": "Ljung-Box(ret)",
            "lag": lag,
            "statistic": row["lb_stat"],
            "p_value": row["lb_pvalue"],
            "interpretation": "autocorrelation present" if row["lb_pvalue"] < 0.05 else "no significant autocorr",
        })

    # 3. Ljung-Box on ret^2
    for lag in [10, 20]:
        lb = acorr_ljungbox(ret2, lags=lag, return_df=True)
        row = lb.iloc[-1]
        results.append({
            "test_name": "Ljung-Box(ret^2)",
            "lag": lag,
            "statistic": row["lb_stat"],
            "p_value": row["lb_pvalue"],
            "interpretation": "volatility clustering" if row["lb_pvalue"] < 0.05 else "no volatility clustering evidence",
        })

    # 4. ARCH-LM
    for lag in [5, 10, 12]:
        lm_stat, lm_pval, f_stat, f_pval = het_arch(ret, nlags=lag)
        results.append({
            "test_name": "ARCH-LM",
            "lag": lag,
            "statistic": lm_stat,
            "p_value": lm_pval,
            "interpretation": "ARCH effect" if lm_pval < 0.05 else "no ARCH effect",
        })

    return results


def main():
    ret = load_returns()
    results = run_tests(ret)

    tbl = pd.DataFrame(results)
    TABLES.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(TABLES / "table2_tests.csv", index=False)

    md = tabulate(tbl, headers="keys", tablefmt="github", showindex=False,
                  floatfmt=".4f")
    (TABLES / "table2_tests.md").write_text(
        "Table 2: Time series tests\n\n" + md + "\n", encoding="utf-8"
    )
    print("[INFO] table2_tests saved.")
    print(md)


if __name__ == "__main__":
    main()
