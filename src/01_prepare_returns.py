"""
01_prepare_returns.py
计算沪深300日对数收益率。
公式: ret_t = 100 * [log(P_t) - log(P_{t-1})]
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def compute_returns():
    clean_path = DATA_PROCESSED / "hs300_daily_clean.csv"
    if not clean_path.exists():
        print(f"[ERROR] 找不到清洗数据: {clean_path}")
        print("请先运行: python src/00_download_data.py")
        return

    df = pd.read_csv(clean_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 计算对数收盘价和百分比日对数收益率
    df["log_close"] = np.log(df["close"])
    df["ret"] = 100.0 * df["log_close"].diff()

    # 删除 ret 缺失的第一行
    df = df.dropna(subset=["ret"]).reset_index(drop=True)

    # 保存
    returns_path = DATA_PROCESSED / "hs300_returns.csv"
    df[["date", "close", "log_close", "ret"]].to_csv(returns_path, index=False)
    print(f"[INFO] 收益率数据已保存: {returns_path}")

    # 样本信息
    n = len(df)
    missing = df["ret"].isna().sum()
    print(f"[INFO] 起始日期: {df['date'].min().date()}")
    print(f"[INFO] 结束日期: {df['date'].max().date()}")
    print(f"[INFO] 样本长度: {n}")
    print(f"[INFO] 缺失值数量: {missing}")


if __name__ == "__main__":
    compute_returns()
