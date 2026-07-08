"""
用成交量當流動性 proxy，評估「漲停鎖死」的名目獲利有多少比例現實中真的買得到。

核心問題：鎖死漲停代表當天想買的人遠多於想賣的人，成交量往往很小；
你的委買單能不能成交、能成交多少，跟當天的流動性直接相關，但我們沒有
委買委賣簿的深度資料，只能用「成交量」這個間接指標去估。這裡用兩種方式量化：

1. 相對量比 volume_ratio = 當天成交量 / 過去 N 個交易日平均成交量
   （不管你想買多少，純粹看這檔股票「今天」跟「平常」比起來，
   到底有沒有實際的雙向成交在發生 -- 量比越低代表越可能是單邊掛死、沒有真成交）

2. 參與率模型（capacity-constrained fill model）
   假設每筆交易投入固定金額 ASSUMED_TRADE_CAPITAL，換算成想買的股數；
   再假設你最多只能吃下當天成交量的 PARTICIPATION_CAP（預設 5%，
   這是流動性文獻常見的「不影響市場」參與率上限，不是精確委買賣資料），
   藉此把「名目報酬率」轉成「打了流動性折扣之後，實際能賺到多少錢」。

ASSUMED_TRADE_CAPITAL / PARTICIPATION_CAP / BASELINE_WINDOW 都是模型假設，
可以自行調整做敏感度測試。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_XLSX = r"C:\TejPro\TejPro\DataExport\20260708230422DataExport.xlsx"
LIMIT_UP_PCT = 0.10
EPSILON = 0.001
BASELINE_WINDOW = 20              # 過去幾個交易日算「正常」成交量基準
MIN_BASELINE_DAYS = 5
ASSUMED_TRADE_CAPITAL = 100_000   # 每筆交易假設投入的資金（元）
PARTICIPATION_CAP = 0.05          # 假設最多能吃下當天成交量的比例

OUT_DIR = Path(__file__).parent
TRADES_CSV = OUT_DIR / "market_liquidity_trades.csv"
QUARTILE_PNG = OUT_DIR / "liquidity_quartile_return.png"
CAP_CURVE_PNG = OUT_DIR / "liquidity_capped_cumulative_twd.png"
SCATTER_PNG = OUT_DIR / "liquidity_vs_return_scatter.png"

COLUMN_MAP = {
    "代號": "code",
    "名稱": "name",
    "年月日": "date",
    "開盤價(元)": "open",
    "收盤價(元)": "close",
    "最高價(元)": "high",
    "最低價(元)": "low",
    "成交量(千股)": "volume_k",
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet")
    df = df.rename(columns=COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    g = df.groupby("code", sort=False)
    df["prev_close"] = g["close"].shift(1)
    df["next_open"] = g["open"].shift(-1)
    df["limit_up_price"] = df["prev_close"] * (1 + LIMIT_UP_PCT)

    # 過去 N 天（不含當天）的平均成交量，當作「正常流動性」基準
    df["volume_k_prior"] = g["volume_k"].shift(1)
    df["baseline_volume_k"] = df.groupby("code")["volume_k_prior"].transform(
        lambda s: s.rolling(BASELINE_WINDOW, min_periods=MIN_BASELINE_DAYS).mean()
    )
    df["volume_ratio"] = df["volume_k"] / df["baseline_volume_k"]
    df.loc[df["baseline_volume_k"].fillna(0) <= 0, "volume_ratio"] = np.nan
    return df


def find_trades(df: pd.DataFrame):
    touched = df["high"] >= df["limit_up_price"] * (1 - EPSILON)
    events = df[touched].copy()
    events["locked"] = events["close"] >= events["limit_up_price"] * (1 - EPSILON)

    locked_events = events[events["locked"]].copy()
    pending = locked_events[locked_events["next_open"].isna()].copy()
    locked_events = locked_events[locked_events["next_open"].notna()].copy()
    locked_events["type"] = "locked"
    locked_events["entry_price"] = locked_events["close"]
    locked_events["exit_price"] = locked_events["next_open"]

    not_locked_events = events[~events["locked"]].copy()
    not_locked_events["type"] = "not_locked"
    not_locked_events["entry_price"] = not_locked_events["high"]
    not_locked_events["exit_price"] = not_locked_events["close"]

    trades = pd.concat([locked_events, not_locked_events], ignore_index=True)
    trades["pnl"] = trades["exit_price"] - trades["entry_price"]
    trades["pnl_pct"] = trades["pnl"] / trades["entry_price"]
    trades = trades.sort_values(["date", "code"]).reset_index(drop=True)

    cols = [
        "code", "name", "date", "type", "entry_price", "exit_price", "pnl", "pnl_pct",
        "volume_k", "baseline_volume_k", "volume_ratio",
    ]
    return trades[cols], pending


def add_liquidity_model(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["volume_shares"] = trades["volume_k"] * 1000
    trades["desired_shares"] = ASSUMED_TRADE_CAPITAL / trades["entry_price"]
    trades["realistic_shares"] = np.minimum(
        trades["desired_shares"], PARTICIPATION_CAP * trades["volume_shares"]
    )
    trades["fill_ratio"] = (trades["realistic_shares"] / trades["desired_shares"]).clip(upper=1.0)
    trades["naive_pnl_twd"] = trades["desired_shares"] * trades["pnl"]
    trades["realistic_pnl_twd"] = trades["realistic_shares"] * trades["pnl"]
    return trades


def summarize_liquidity(trades: pd.DataFrame):
    print("=== 流動性分析 ===")
    print(f"總交易數：{len(trades)}\n")

    by_type = trades.groupby("type").agg(
        件數=("pnl_pct", "size"),
        平均相對量比=("volume_ratio", "mean"),
        中位數相對量比=("volume_ratio", "median"),
        平均可成交比例=("fill_ratio", "mean"),
    )
    print("依鎖死/未鎖死分類的流動性指標：")
    print(by_type)
    print()

    naive_total = trades["naive_pnl_twd"].sum()
    realistic_total = trades["realistic_pnl_twd"].sum()
    print(f"名目總損益（假設每筆都能全額投入 {ASSUMED_TRADE_CAPITAL:,.0f} 元）：{naive_total:,.0f} 元")
    print(f"流動性折算後總損益（每筆最多吃下當天成交量的 {PARTICIPATION_CAP:.0%}）：{realistic_total:,.0f} 元")
    print(f"折算後剩餘比例：{realistic_total / naive_total:.1%}\n")

    locked = trades[trades["type"] == "locked"].dropna(subset=["volume_ratio"]).copy()

    rho, p = stats.spearmanr(locked["volume_ratio"], locked["pnl_pct"])
    print(f"[鎖死組] 相對量比 vs 報酬率 Spearman 相關：rho={rho:.3f}, p={p:.4g}")
    rho2, p2 = stats.spearmanr(locked["fill_ratio"], locked["pnl_pct"])
    print(f"[鎖死組] 可成交比例 vs 報酬率 Spearman 相關：rho={rho2:.3f}, p={p2:.4g}\n")

    locked["liquidity_quartile"] = pd.qcut(
        locked["volume_ratio"], 4,
        labels=["Q1 最不流動", "Q2", "Q3", "Q4 最流動"],
        duplicates="drop",
    )
    quartile_stats = locked.groupby("liquidity_quartile", observed=True).agg(
        件數=("pnl_pct", "size"),
        勝率=("pnl_pct", lambda s: (s > 0).mean()),
        平均報酬率=("pnl_pct", "mean"),
        平均可成交比例=("fill_ratio", "mean"),
    )
    print("[鎖死組] 依相對量比分四分位：")
    print(quartile_stats)
    print()

    return locked, quartile_stats


def plot_quartile_bar(quartile_stats: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    data = quartile_stats.reset_index()
    sns.barplot(data=data, x="liquidity_quartile", y="平均報酬率")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.title("鎖死漲停：依「相對成交量」四分位的平均報酬率")
    plt.xlabel("相對量比四分位（Q1=最不流動）")
    plt.ylabel("平均報酬率")
    plt.tight_layout()
    plt.savefig(QUARTILE_PNG)
    print(f"已存圖：{QUARTILE_PNG}")


def plot_capacity_curves(trades: pd.DataFrame):
    ordered = trades.sort_values("date").reset_index(drop=True)
    ordered["名目（假設全額成交）"] = ordered["naive_pnl_twd"].cumsum()
    ordered["流動性折算後"] = ordered["realistic_pnl_twd"].cumsum()
    long = ordered.melt(
        id_vars=["date"],
        value_vars=["名目（假設全額成交）", "流動性折算後"],
        var_name="series", value_name="cum_twd",
    )

    plt.figure(figsize=(11, 5))
    sns.lineplot(data=long, x="date", y="cum_twd", hue="series")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.title(f"累積損益（元）：名目 vs 流動性折算後（每筆假設投入 {ASSUMED_TRADE_CAPITAL:,.0f} 元）")
    plt.xlabel("日期")
    plt.ylabel("累積損益（元）")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CAP_CURVE_PNG)
    print(f"已存圖：{CAP_CURVE_PNG}")


def plot_scatter(locked: pd.DataFrame):
    valid = locked[(locked["volume_ratio"] > 0) & np.isfinite(locked["volume_ratio"])].copy()
    # 用手動 log10 轉換 + 線性座標軸，避免 matplotlib log-scale 的科學記號 (10^-1)
    # 在某些字型組合下無法正確顯示負號 glyph 的問題
    valid["log10_volume_ratio"] = np.log10(valid["volume_ratio"])

    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=valid, x="log10_volume_ratio", y="pnl_pct", alpha=0.4, s=20)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.title("鎖死漲停：相對量比 vs 報酬率")
    plt.xlabel("log10(相對量比)　— 0 代表當天成交量=過去20日均量，負值代表比平常更冷清")
    plt.ylabel("報酬率")
    plt.tight_layout()
    plt.savefig(SCATTER_PNG)
    print(f"已存圖：{SCATTER_PNG}")


def main():
    df = load_data(DATA_XLSX)
    trades, pending = find_trades(df)
    trades = add_liquidity_model(trades)

    trades.to_csv(TRADES_CSV, index=False, encoding="utf-8-sig")
    print(f"已存交易明細（含流動性欄位）：{TRADES_CSV}\n")

    locked, quartile_stats = summarize_liquidity(trades)
    plot_quartile_bar(quartile_stats)
    plot_capacity_curves(trades)
    plot_scatter(locked)


if __name__ == "__main__":
    main()
