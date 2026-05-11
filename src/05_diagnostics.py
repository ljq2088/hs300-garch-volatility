"""
05_diagnostics.py
GARCH(1,1)主模型残差诊断。
诊断对象：标准化残差 z_t = ε_t / σ_t
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from tabulate import tabulate

PROJECT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT / "data" / "processed"
FIGURES = PROJECT / "figures"
TABLES = PROJECT / "tables"

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_data():
    df = pd.read_csv(DATA_PROCESSED / "hs300_returns.csv", parse_dates=["date"])
    vol = pd.read_csv(DATA_PROCESSED / "hs300_garch_conditional_volatility.csv",
                      parse_dates=["date"])
    return df, vol


def main():
    df, vol = load_data()
    dates = df["date"].values
    ret = df["ret"].values
    sigma = vol["conditional_volatility"].values

    # 标准化残差
    z = ret / sigma
    z2 = z ** 2

    FIGURES.mkdir(parents=True, exist_ok=True)

    # fig6: 标准化残差
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, z, linewidth=0.3, color="black")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("Standardized Residuals $z_t$ — GARCH(1,1)")
    ax.set_xlabel("Date")
    ax.set_ylabel("$z_t$")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_standardized_residuals.png", dpi=150)
    plt.close(fig)

    # fig7: 平方标准化残差
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, z2, linewidth=0.3, color="black")
    ax.set_title("Squared Standardized Residuals $z_t^2$ — GARCH(1,1)")
    ax.set_xlabel("Date")
    ax.set_ylabel("$z_t^2$")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_squared_standardized_residuals.png", dpi=150)
    plt.close(fig)

    # fig8: ACF of z
    fig = plot_acf(z, lags=30, title="ACF of Standardized Residuals $z_t$")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_acf_standardized_residuals.png", dpi=150)
    plt.close(fig)

    # fig9: ACF of z^2
    fig = plot_acf(z2, lags=30, title="ACF of Squared Standardized Residuals $z_t^2$")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig9_acf_squared_standardized_residuals.png", dpi=150)
    plt.close(fig)

    print("[INFO] 4 diagnostic figures saved.")

    # 诊断检验
    results = []

    for lag in [10, 20]:
        lb = acorr_ljungbox(z, lags=lag, return_df=True)
        row = lb.iloc[-1]
        results.append({
            "test": f"Ljung-Box(z_t) lag={lag}",
            "statistic": row["lb_stat"],
            "p_value": row["lb_pvalue"],
            "interpretation": "still autocorrelated" if row["lb_pvalue"] < 0.05
            else "no significant autocorrelation (mean spec OK)",
        })

    for lag in [10, 20]:
        lb = acorr_ljungbox(z2, lags=lag, return_df=True)
        row = lb.iloc[-1]
        results.append({
            "test": f"Ljung-Box(z_t^2) lag={lag}",
            "statistic": row["lb_stat"],
            "p_value": row["lb_pvalue"],
            "interpretation": "remaining ARCH effect" if row["lb_pvalue"] < 0.05
            else "GARCH captured most heteroskedasticity",
        })

    for lag in [5, 10, 12]:
        lm_stat, lm_pval, _, _ = het_arch(z, nlags=lag)
        results.append({
            "test": f"ARCH-LM(z_t) lag={lag}",
            "statistic": lm_stat,
            "p_value": lm_pval,
            "interpretation": "remaining ARCH" if lm_pval < 0.05
            else "heteroskedasticity well captured",
        })

    tbl = pd.DataFrame(results)
    TABLES.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(TABLES / "table5_residual_diagnostics.csv", index=False)
    md = tabulate(tbl, headers="keys", tablefmt="github", showindex=False,
                  floatfmt=".4f")
    (TABLES / "table5_residual_diagnostics.md").write_text(
        "Table 5: Residual diagnostics — GARCH(1,1) standardized residuals\n\n" + md + "\n",
        encoding="utf-8",
    )
    print("[INFO] table5_residual_diagnostics saved.")
    print(md)


if __name__ == "__main__":
    main()
