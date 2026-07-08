"""
全台股（今年）漲停策略回測

同一套策略邏輯（見 limit_up_backtest.py），套用到 TEJ 匯出的全市場日資料：
- 當天觸及漲停 -> 假設能買到漲停價
  - 鎖死到收盤：隔天開盤賣出，pnl = 隔天開盤價 - 當天收盤(漲停)價
  - 沒鎖死：當天收盤賣出，pnl = 當天收盤價 - 當天最高(漲停)價

因為橫跨上千檔不同價位的股票，統計與畫圖都以 pnl_pct（報酬率）為主，
pnl（價差）只保留在明細表中做參考。
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

DATA_XLSX = r"C:\TejPro\TejPro\DataExport\20260708215428DataExport.xlsx"
LIMIT_UP_PCT = 0.10
EPSILON = 0.001

OUT_DIR = Path(__file__).parent
TRADES_CSV = OUT_DIR / "market_trades.csv"
DIST_PNG = OUT_DIR / "market_pnl_distribution.png"
CUM_PNG = OUT_DIR / "market_cumulative_pnl_pct.png"
IC_CSV = OUT_DIR / "market_daily_ic.csv"
IC_PNG = OUT_DIR / "market_daily_ic.png"

COLUMN_MAP = {
    "代號": "code",
    "名稱": "name",
    "年月日": "date",
    "開盤價(元)": "open",
    "收盤價(元)": "close",
    "最高價(元)": "high",
    "最低價(元)": "low",
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet")
    df = df.rename(columns=COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df


def find_trades(df: pd.DataFrame):
    df = df.copy()
    g = df.groupby("code", sort=False)
    df["prev_close"] = g["close"].shift(1)
    df["next_open"] = g["open"].shift(-1)
    df["limit_up_price"] = df["prev_close"] * (1 + LIMIT_UP_PCT)

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

    cols = ["code", "name", "date", "type", "entry_price", "exit_price", "pnl", "pnl_pct"]
    return trades[cols], pending[["code", "name", "date"]]


def summarize(trades: pd.DataFrame, pending: pd.DataFrame):
    n = len(trades)
    wins = trades[trades["pnl_pct"] > 0]
    win_rate = len(wins) / n

    print(f"資料期間：{trades['date'].min().date()} ~ {trades['date'].max().date()}")
    print(f"觸及漲停事件（已平倉）：{n} 筆　（另有 {len(pending)} 筆鎖死漲停但尚無隔日資料，未計入統計）")
    print(f"涉及股票數：{trades['code'].nunique()} 檔\n")

    print(f"整體勝率：{win_rate:.1%}")
    print(f"平均每筆報酬率：{trades['pnl_pct'].mean():.2%}")
    print(f"報酬率總和（等權重逐筆加總）：{trades['pnl_pct'].sum():.2%}")
    print(f"最大單筆獲利：{trades['pnl_pct'].max():.2%}")
    print(f"最大單筆虧損：{trades['pnl_pct'].min():.2%}\n")

    by_type = trades.groupby("type").agg(
        件數=("pnl_pct", "size"),
        勝率=("pnl_pct", lambda s: (s > 0).mean()),
        平均報酬率=("pnl_pct", "mean"),
        報酬率總和=("pnl_pct", "sum"),
    )
    print("依鎖死/未鎖死分類：")
    print(by_type)
    print()

    top5 = trades.nlargest(5, "pnl_pct")[["date", "code", "name", "type", "pnl_pct"]]
    bottom5 = trades.nsmallest(5, "pnl_pct")[["date", "code", "name", "type", "pnl_pct"]]
    print("報酬率最高 5 筆：")
    print(top5.to_string(index=False))
    print("\n報酬率最低 5 筆：")
    print(bottom5.to_string(index=False))


def statistical_tests(trades: pd.DataFrame):
    print("=== 統計檢定 ===")

    locked_pnl = trades.loc[trades["type"] == "locked", "pnl_pct"]
    not_locked_pnl = trades.loc[trades["type"] == "not_locked", "pnl_pct"]

    print("\n[單樣本 t 檢定] 平均報酬率是否顯著不等於 0：")
    for label, series in [
        ("整體", trades["pnl_pct"]),
        ("僅鎖死", locked_pnl),
        ("僅未鎖死", not_locked_pnl),
    ]:
        t_stat, p_value = stats.ttest_1samp(series, popmean=0)
        print(f"  {label}：mean={series.mean():.4%}, n={len(series)}, t={t_stat:.3f}, p={p_value:.4g}")

    print("\n[兩樣本 t 檢定 (Welch)] 鎖死 vs 未鎖死的平均報酬率是否有顯著差異：")
    t_stat2, p_value2 = stats.ttest_ind(locked_pnl, not_locked_pnl, equal_var=False)
    print(f"  t={t_stat2:.3f}, p={p_value2:.4g}")

    print("\n[線性迴歸] 累積報酬率曲線 vs 交易序號（檢驗趨勢是否顯著、線性配適程度 R²）：")
    for label, sub in [
        ("整體", trades),
        ("僅鎖死", trades[trades["type"] == "locked"]),
        ("僅未鎖死", trades[trades["type"] == "not_locked"]),
    ]:
        ordered = sub.sort_values("date")
        cum = ordered["pnl_pct"].cumsum().to_numpy()
        x = np.arange(len(cum))
        result = stats.linregress(x, cum)
        print(
            f"  {label}：slope={result.slope:.4f}／筆, R²={result.rvalue ** 2:.4f}, "
            f"p={result.pvalue:.4g}"
        )
    print()


def ic_ir_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """
    訊號：當天是否鎖死漲停（1=鎖死, 0=未鎖死）
    結果：該筆交易的實際報酬率 pnl_pct
    每個交易日做一次橫斷面 Spearman 相關係數（= 當天的 IC），
    再把每天的 IC 收集成時間序列，算 IR = mean(IC) / std(IC)。
    同一天至少要有鎖死跟未鎖死兩種事件同時出現，相關係數才算得出來。
    """
    print("=== IC / IR 分析（訊號：是否鎖死 vs 實際報酬率）===")

    df = trades.copy()
    df["signal"] = (df["type"] == "locked").astype(int)

    records = []
    for date, group in df.groupby("date"):
        if group["signal"].nunique() < 2 or len(group) < 3:
            continue
        ic, _ = stats.spearmanr(group["signal"], group["pnl_pct"])
        if pd.isna(ic):
            continue
        records.append({"date": date, "ic": ic, "n": len(group)})

    ic_df = pd.DataFrame(records)
    if ic_df.empty:
        print("可用天數不足（同一天需同時出現鎖死與未鎖死事件），無法計算 IC 時間序列。\n")
        return ic_df

    mean_ic = ic_df["ic"].mean()
    std_ic = ic_df["ic"].std(ddof=1)
    ir = mean_ic / std_ic if std_ic > 0 else float("nan")
    t_stat, p_value = stats.ttest_1samp(ic_df["ic"], popmean=0)

    print(f"可計算 IC 的交易日數：{len(ic_df)} / {df['date'].nunique()}")
    print(f"平均 IC（Spearman rank correlation）：{mean_ic:.4f}")
    print(f"IC 標準差：{std_ic:.4f}")
    print(f"IR（mean IC / std IC）：{ir:.4f}")
    print(f"IC 序列 t 檢定（平均 IC 是否顯著不為 0）：t={t_stat:.3f}, p={p_value:.4g}\n")

    ic_df.to_csv(IC_CSV, index=False, encoding="utf-8-sig")
    print(f"已存每日 IC：{IC_CSV}")
    return ic_df


def plot_ic(ic_df: pd.DataFrame):
    if ic_df.empty:
        return
    mean_ic = ic_df["ic"].mean()

    plt.figure(figsize=(11, 5))
    sns.lineplot(data=ic_df, x="date", y="ic", marker="o", markersize=4)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axhline(mean_ic, color="red", linestyle="--", linewidth=1, label=f"平均 IC = {mean_ic:.3f}")
    plt.legend()
    plt.title("每日 IC（鎖死訊號 vs 實際報酬率）")
    plt.xlabel("日期")
    plt.ylabel("IC（Spearman）")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(IC_PNG)
    print(f"已存圖：{IC_PNG}")


def plot_distribution(trades: pd.DataFrame):
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=trades, x="pnl_pct", hue="type", element="step",
        stat="density", common_norm=False, kde=True, bins=40,
    )
    plt.title("漲停事件報酬率分布（鎖死 vs 未鎖死）")
    plt.xlabel("單筆報酬率")
    plt.ylabel("密度")
    plt.tight_layout()
    plt.savefig(DIST_PNG)
    print(f"\n已存圖：{DIST_PNG}")


def plot_cumulative(trades: pd.DataFrame):
    # 每個系列各自獨立排序、各自累加 -- 不要把不同類型的交易混在同一條累加線裡，
    # 否則「未鎖死」的點只是疊在「整體」曲線上，看起來會像跟著整體一起漲。
    series_list = []
    for label, sub in [
        ("整體", trades),
        ("僅鎖死", trades[trades["type"] == "locked"]),
        ("僅未鎖死", trades[trades["type"] == "not_locked"]),
    ]:
        ordered = sub.sort_values("date").reset_index(drop=True)
        ordered["cum_pnl_pct"] = ordered["pnl_pct"].cumsum()
        ordered["series"] = label
        series_list.append(ordered[["date", "series", "cum_pnl_pct"]])
    combined = pd.concat(series_list, ignore_index=True)

    plt.figure(figsize=(11, 5))
    sns.lineplot(data=combined, x="date", y="cum_pnl_pct", hue="series", marker="o", markersize=4)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.title("全台股漲停策略 累積報酬率曲線（各類型各自累加）")
    plt.xlabel("日期")
    plt.ylabel("累積報酬率")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CUM_PNG)
    print(f"已存圖：{CUM_PNG}")


def main():
    df = load_data(DATA_XLSX)
    trades, pending = find_trades(df)

    if trades.empty:
        print("沒有偵測到任何漲停事件。")
        return

    trades.to_csv(TRADES_CSV, index=False, encoding="utf-8-sig")
    print(f"已存交易明細：{TRADES_CSV}\n")

    summarize(trades, pending)
    statistical_tests(trades)
    ic_df = ic_ir_analysis(trades)
    plot_distribution(trades)
    plot_cumulative(trades)
    plot_ic(ic_df)


if __name__ == "__main__":
    main()
