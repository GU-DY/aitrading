from datetime import datetime
import pandas as pd
from src.backtester import Backtester           # 路径按你的实际位置调整
from src.main import run_hedge_fund                  # 你的智能体
                                                     # 如果路径不同请相应修改

def run_backtest(ticker: str,
                 start_date: str,
                 end_date: str,
                 initial_capital: float,
                 num_of_news: int = 5) -> dict:
    """
    运行回测并返回:
    {
        "table": [...],     # 列表(dict) → 前端表格 / 画线
        "plot" : "xxxx"     # Base64 PNG → <img src="data:image/png;base64,...">
    }
    """
    bt = Backtester(
        agent=run_hedge_fund,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        num_of_news=num_of_news
    )

    # 1. 跑回测
    bt.run_backtest()

    # 2. 获取 (DataFrame, base64 图)
    perf_df, img_b64 = bt.analyze_performance()

    # 3. DataFrame ⇒ 列表(dict)，日期列转字符串
    perf_df = perf_df.reset_index()
    if "date" not in perf_df.columns:         # 处理 index→date
        perf_df.rename(columns={"index": "date", "Date": "date"}, inplace=True)
    perf_df["date"] = perf_df["date"].apply(
        lambda x: x.strftime("%Y-%m-%d") if not isinstance(x, str) else x
    )
    table = perf_df.to_dict(orient="records")

    # 4. 打包返回
    return {
        "table": table,
        "plot": img_b64
    }