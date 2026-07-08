"""
聯發科（2454.TW）漲停策略回測

策略：
- 當天觸及漲停 -> 假設能買到漲停價
  - 若鎖死到收盤：隔天開盤賣出，損益 = 隔天開盤價 - 當天收盤(漲停)價
  - 若沒鎖死：當天收盤賣出，損益 = 當天收盤價 - 當天最高(漲停)價
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

TICKER = "2454.TW"
START_DATE = "2026-01-01"  # 可調整成更早日期以擴大回測範圍
LIMIT_UP_PCT = 0.10        # 台股個股漲停幅度
EPSILON = 0.001            # 容忍跳動點位取整造成的誤差

OUT_DIR = Path(__file__).parent
TRADES_CSV = OUT_DIR / "trades.csv"
CHART_PNG = OUT_DIR / "cumulative_pnl.png"


def fetch_data(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} starting {start}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def find_trades(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prev_close"] = df["Close"].shift(1)
    df["limit_up_price"] = df["prev_close"] * (1 + LIMIT_UP_PCT)

    trades = []
    n = len(df)
    for i in range(1, n):  # i=0 has no prev_close
        row = df.iloc[i]
        limit_price = row["limit_up_price"]
        if pd.isna(limit_price):
            continue

        touched = row["High"] >= limit_price * (1 - EPSILON)
        if not touched:
            continue

        locked = row["Close"] >= limit_price * (1 - EPSILON)
        date = df.index[i]

        if locked:
            if i + 1 >= n:
                trades.append({
                    "date": date,
                    "type": "locked",
                    "entry_price": row["Close"],
                    "exit_price": None,
                    "pnl": None,
                    "pnl_pct": None,
                    "note": "尚未平倉（無隔日資料）",
                })
                continue
            next_row = df.iloc[i + 1]
            entry_price = row["Close"]
            exit_price = next_row["Open"]
            pnl = exit_price - entry_price
            trades.append({
                "date": date,
                "type": "locked",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl / entry_price,
                "note": "",
            })
        else:
            entry_price = row["High"]
            exit_price = row["Close"]
            pnl = exit_price - entry_price
            trades.append({
                "date": date,
                "type": "not_locked",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl / entry_price,
                "note": "",
            })

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    closed = trades.dropna(subset=["pnl"])
    if closed.empty:
        print("沒有已平倉的交易可供統計。")
        return closed

    wins = closed[closed["pnl"] > 0]
    win_rate = len(wins) / len(closed)
    total_pnl = closed["pnl"].sum()
    avg_pnl = closed["pnl"].mean()
    avg_pnl_pct = closed["pnl_pct"].mean()
    max_win = closed["pnl"].max()
    max_loss = closed["pnl"].min()

    print(f"標的：{TICKER}　回測起始：{START_DATE}")
    print(f"總交易次數：{len(closed)}（另有 {len(trades) - len(closed)} 筆尚未平倉）")
    print(f"勝率：{win_rate:.1%}")
    print(f"總損益：{total_pnl:.2f} 元")
    print(f"平均每筆損益：{avg_pnl:.2f} 元（{avg_pnl_pct:.2%}）")
    print(f"最大單筆獲利：{max_win:.2f} 元")
    print(f"最大單筆虧損：{max_loss:.2f} 元")
    print()
    print(closed.to_string(index=False))
    return closed


def plot_cumulative_pnl(closed: pd.DataFrame):
    if closed.empty:
        return
    cum = closed["pnl"].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(closed["date"], cum, marker="o")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.title(f"{TICKER} 漲停策略 累積損益曲線")
    plt.xlabel("日期")
    plt.ylabel("累積損益（元）")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHART_PNG)
    print(f"\n已存圖：{CHART_PNG}")


def main():
    df = fetch_data(TICKER, START_DATE)
    trades = find_trades(df)

    if trades.empty:
        print(f"{START_DATE} 至今，{TICKER} 沒有偵測到任何漲停事件。")
        sys.exit(0)

    trades.to_csv(TRADES_CSV, index=False, encoding="utf-8-sig")
    print(f"已存交易明細：{TRADES_CSV}\n")

    closed = summarize(trades)
    plot_cumulative_pnl(closed)


if __name__ == "__main__":
    main()
