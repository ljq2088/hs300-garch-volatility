"""
00_download_data.py
下载沪深300指数日度历史行情数据并标准化。
数据源：AkShare (东方财富)
"""
import sys
from pathlib import Path
import pandas as pd

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def download_hs300_daily():
    """尝试通过 akshare 下载沪深300日度数据，失败时给出明确报错。"""
    try:
        import akshare as ak
        print("[INFO] AkShare imported successfully.")
    except ImportError:
        print("[ERROR] akshare 未安装。请运行: pip install akshare")
        sys.exit(1)

    # 尝试接口1: index_zh_a_hist (东方财富)
    df = None
    for attempt, func_desc in enumerate(
        [
            ("index_zh_a_hist", lambda: ak.index_zh_a_hist(
                symbol="000300", period="daily",
                start_date="20150101", end_date="20251231"
            )),
            ("stock_zh_index_daily(sh000300)", lambda: ak.stock_zh_index_daily(
                symbol="sh000300"
            )),
        ],
        start=1,
    ):
        try:
            print(f"[INFO] 尝试接口 {attempt}: {func_desc[0]}")
            df = func_desc[1]()
            if df is not None and not df.empty:
                print(f"[INFO] 接口 {attempt} 成功，获取 {len(df)} 行数据。")
                break
        except Exception as e:
            print(f"[WARN] 接口 {attempt} 失败: {e}")
            continue

    if df is None or df.empty:
        print("\n" + "=" * 60)
        print("[ERROR] 所有 AkShare 接口均失败。")
        print("请手动提供沪深300日度CSV文件，至少包含 date 和 close 两列，")
        print(f"保存至: {DATA_RAW / 'hs300_daily_raw.csv'}")
        print("=" * 60)
        sys.exit(1)

    # 标准化列名
    column_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns=column_map)

    # 确保必要列存在
    required_cols = ["date", "close"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] 数据缺少列: {col}。现有列: {list(df.columns)}")
            sys.exit(1)

    # 清理和标准化
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # 只保留需要的列（如果有的话）
    keep_cols = [c for c in required_cols + ["open", "high", "low", "volume", "amount"]
                 if c in df.columns]
    df = df[keep_cols]

    # 保存
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_RAW / "hs300_daily_raw.csv"
    clean_path = DATA_PROCESSED / "hs300_daily_clean.csv"
    df.to_csv(raw_path, index=False)
    df.to_csv(clean_path, index=False)

    print(f"[INFO] 原始数据已保存: {raw_path}")
    print(f"[INFO] 清理数据已保存: {clean_path}")
    print(f"[INFO] 数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"[INFO] 样本量: {len(df)}")
    print(f"[INFO] 缺失值: close={df['close'].isna().sum()}, total={df.isna().sum().sum()}")


if __name__ == "__main__":
    download_hs300_daily()
