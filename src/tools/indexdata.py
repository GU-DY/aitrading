import pandas as pd
from WindPy import w
import re
def extract_stock_code(s):
    # 匹配字符串开头的连续数字
    match = re.match(r'^\d+', s)
    return match.group(0) if match else ''

def get_index_constituents(index_code: str, date: str) -> pd.DataFrame:
    """Query index constituents and weights via WindPy.

    Parameters
    ----------
    index_code : str
        Wind code for the index, e.g. "000300.SH".
    date : str
        Query date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``wind_code`` and ``weight`` filtered to only
        include stocks ending with ``.SZ`` or ``.SH``.
    """
    w.start()  # 必须执行且返回 ErrorCode=0 才有效
    result = w.wset(
        "indexconstituent",
        f"date={date};windcode={index_code};field=wind_code,i_weight",
    )
    if result.ErrorCode != 0:
        raise RuntimeError(f"Wind wset error: {result.ErrorCode}")

    df = pd.DataFrame({
        "wind_code": result.Data[0],
        "weight": result.Data[1],
    })
    df = df[df["wind_code"].str.endswith((".SZ", ".SH"))].reset_index(drop=True)
    return df


def predict_stock_return(stock_code: str, date: str) -> float:
    """Dummy single stock return prediction.

    The current implementation generates a deterministic pseudo-random
    return between ``-0.05`` and ``0.05`` based on the stock code and date.
    This serves as a placeholder for a real forecasting model.
    """
    import random

    random.seed(hash(stock_code + date) & 0xFFFFFFFF)
    return random.uniform(-0.05, 0.05)


def predict_index_return(index_code: str, date: str) -> float:
    """Predict index return by aggregating weighted stock predictions.

    ``predict_stock_return`` is used for each constituent. Replace it with a
    real forecasting model as needed.
    """
    constituents = get_index_constituents(index_code, date)
    if constituents.empty:
        return 0.0

    constituents["pred"] = constituents["wind_code"].apply(
        lambda x: predict_stock_return(x, date)
    )
    # Weight is in percentage; convert to fraction when summing
    return float((constituents["pred"] * constituents["weight"]).sum() / 100)


def predict_stock_return_agent(stock_code: str, start_date: str) -> float:
    """Predict single-stock return using the hedge fund agent workflow.

    This function mirrors the logic in ``hedge_fund_service.run_hedge_fund``
    but returns the predicted percentage return (e.g. ``5.0`` for ``5%``).
    """
    from datetime import datetime, timedelta
    import json
    import re

    from langchain_core.messages import HumanMessage
    from hedge_fund_service import app, _strip_code_block
    from src.agents.portfolio_manager import compute_predicted_return

    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    portfolio = {"cash": 100_000, "stock": 0}

    fs = app.invoke(
        {
            "messages": [HumanMessage(content="Make a trading decision based on the provided data.")],
            "data": {
                "ticker": extract_stock_code(stock_code),
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "num_of_news": 5,
            },
            "metadata": {"show_reasoning": False},
        }
    )

    final_raw = _strip_code_block(fs["messages"][-1].content.strip())
    try:
        js = json.loads(final_raw)
    except Exception:
        js = json.loads(re.sub(r"```(?:json)?|```", "", final_raw).strip())

    pct = js.get("predicted_return_pct", 0.0)
    if pct in (0, 0.0):
        pct = compute_predicted_return(js.get("agent_signals", []))
    elif abs(pct) < 1:
        pct = pct * 100

    return float(pct)


def predict_index_return_agent(index_code: str, date: str, start_date: str) -> float:
    """Predict index return by aggregating hedge-fund-agent forecasts.

    ``predict_stock_return_agent`` is invoked for each constituent.  The
    returned value is a decimal fraction (e.g. ``0.05`` represents ``5%``).
    """
    constituents = get_index_constituents(index_code, date)
    if constituents.empty:
        return 0.0

    constituents["pred"] = constituents["wind_code"].apply(
        lambda code: predict_stock_return_agent(code, start_date)
    )
    return float((constituents["pred"] * constituents["weight"]).sum() / 10000)