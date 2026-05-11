"""
02_descriptive_analysis.py
描述性统计与图表生成。
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm
from tabulate import tabulate

PROJECT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT / "data" / "processed"
FIGURES = PROJECT / "figures"
TABLES = PROJECT / "tables"

# 中文字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def load_returns():
    path = DATA_PROCESSED / "hs300_returns.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，请先运行 01_prepare_returns.py")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def make_figures(df):
    FIGURES.mkdir(parents=True, exist_ok=True)
    ret = df["ret"].values
    close = df["close"].values
    dates = df["date"].values

    # fig1: 收盘价
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, close, linewidth=0.6, color="black")
    ax.set_title("HS300 Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_hs300_close.png", dpi=150)
    plt.close(fig)

    # fig2: 收益率
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, ret, linewidth=0.3, color="black")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("HS300 Daily Log Returns (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_hs300_returns.png", dpi=150)
    plt.close(fig)

    # fig3: 平方收益率
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, ret**2, linewidth=0.3, color="black")
    ax.set_title("HS300 Squared Daily Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Squared return")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_hs300_squared_returns.png", dpi=150)
    plt.close(fig)

    # fig4: 收益率直方图 + 正态参考线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ret, bins=60, density=True, alpha=0.6, color="gray", edgecolor="white")
    x_grid = np.linspace(ret.min(), ret.max(), 500)
    ax.plot(x_grid, norm.pdf(x_grid, ret.mean(), ret.std()),
            color="black", linewidth=1.2, label="Normal reference")
    ax.set_title("HS300 Return Histogram")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_return_histogram.png", dpi=150)
    plt.close(fig)

    print("[INFO] 4 figures saved to figures/")


def make_table(df):
    TABLES.mkdir(parents=True, exist_ok=True)
    ret = df["ret"].values
    stats = {
        "statistic": ["count", "mean", "std", "min", "max", "skewness", "kurtosis"],
        "value": [
            len(ret),
            np.mean(ret),
            np.std(ret, ddof=1),
            np.min(ret),
            np.max(ret),
            skew(ret, bias=False),
            kurtosis(ret, bias=False),
        ],
    }
    tbl = pd.DataFrame(stats)

    # CSV
    tbl.to_csv(TABLES / "table1_descriptive_stats.csv", index=False)

    # Markdown
    md = tabulate(tbl, headers="keys", tablefmt="github", showindex=False,
                  floatfmt=".4f")
    (TABLES / "table1_descriptive_stats.md").write_text(
        "Table 1: Descriptive statistics of HS300 daily log returns\n\n" + md + "\n",
        encoding="utf-8",
    )
    print("[INFO] table1_descriptive_stats saved.")

    # 打印关键解释
    print(f"\n--- Descriptive Summary ---")
    print(f"  mean  = {stats['value'][1]:.4f}%")
    print(f"  std   = {stats['value'][2]:.4f}%")
    print(f"  min   = {stats['value'][3]:.4f}%")
    print(f"  max   = {stats['value'][4]:.4f}%")
    print(f"  skew  = {stats['value'][5]:.4f}")
    print(f"  kurt  = {stats['value'][6]:.4f}  (excess vs normal=0)")


def main():
    df = load_returns()
    make_figures(df)
    make_table(df)


if __name__ == "__main__":
    main()
