"""
financial_data_utils.py  （改进版）
---------------------------------

* 统一日志配置  
* safe_float 避免 None/空值/np.nan 转换错误  
* WindPy 自动启动、代码补后缀、兜底封装  
* 四大核心接口  
    1. 财务指标（agent 精简版）  
    2. 财报行项目（最新 & 上期）  
    3. 市场数据快照  
    4. 价格历史 + 常用技术指标  
* 内置 __main__ 测试：自动连接 Wind 并演示三个接口

依赖：  
    - Wind 终端 + WindPy  
    - 第三方包：akshare、pandas、numpy
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import akshare as ak
from WindPy import w  # pip install WindPy（随 Wind 终端安装）

# -----------------------------------------------------------------------------
# 🔧 日志配置
# -----------------------------------------------------------------------------
LOG_LVL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("financial_data")
logger.setLevel(LOG_LVL)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(_h)

# -----------------------------------------------------------------------------
# 🔧 通用工具
# -----------------------------------------------------------------------------
def safe_float(val: Any, default: float = 0.0) -> float:  # noqa: ANN401
    """安全转 float：None/空串/np.nan ➜ default。"""
    try:
        if val in (None, "", np.nan):
            return default
        return float(val)
    except Exception:
        return default


def _first_exist(src: Union[pd.Series, Dict[str, Any]], keys: list[str]) -> float:
    """返回 src 中首个存在且非空字段的数值（float）。"""
    for k in keys:
        if k in src and pd.notna(src[k]):
            return safe_float(src[k])
    return 0.0


# -----------------------------------------------------------------------------
# 🔧 Wind 工具函数
# -----------------------------------------------------------------------------
def _ensure_wind_started() -> None:
    if not w.isconnected():
        w.start()
        if not w.isconnected():
            raise RuntimeError("Unable to connect WindPy")


def _wind_code(symbol: str) -> str:
    if symbol.startswith(("6", "9")):
        return f"{symbol}.SH"
    if symbol.startswith(("0", "3")):
        return f"{symbol}.SZ"
    raise ValueError(f"Invalid A-share code: {symbol}")


def fallback_wind_wss(symbol: str, fields: str, options: str = "") -> Dict[str, Any]:
    """Wind wss → dict(field→value)；失败返回空 dict。"""
    _ensure_wind_started()
    data = w.wss(_wind_code(symbol), fields, options)
    if data.ErrorCode != 0:
        logger.warning("Wind wss error %s [fields=%s]", data.ErrorCode, fields)
        return {}
    return {
        f: safe_float(data.Data[idx][0]) if data.Data[idx] else None
        for idx, f in enumerate(fields.split(","))
    }

# -----------------------------------------------------------------------------
# 📊 财务指标 (agent 精简版)
# -----------------------------------------------------------------------------
def get_financial_metrics(
    symbol: str,
    cache_dir: str = "src/data/financial_metrics",
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """主要指标：ROE、PE 等；Akshare 主，Wind 兜底。返回 list 仅 1 项。"""
    logger.info("[metrics] %s", symbol)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol}_metrics.json")

    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as fh:
                return [json.load(fh)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("read-cache failed: %s", exc)

    # 实时行情
    try:
        spot = ak.stock_zh_a_spot_em()
        row = spot.loc[spot["代码"] == symbol].iloc[0]
        stock_rt = {
            "market_cap": safe_float(row["总市值"]),
            "float_market_cap": safe_float(row["流通市值"]),
            "pe_ratio": safe_float(row["市盈率-动态"]),
            "price_to_book": safe_float(row["市净率"]),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Akshare spot fail (%s) → Wind", exc)
        flds = "mkt_cap_ard,mkt_cap_float,pe_ttm,pb"
        d = fallback_wind_wss(symbol, flds)
        stock_rt = {
            "market_cap": safe_float(d.get("mkt_cap_ard")),
            "float_market_cap": safe_float(d.get("mkt_cap_float")),
            "pe_ratio": safe_float(d.get("pe_ttm")),
            "price_to_book": safe_float(d.get("pb")),
        }

    # 财务指标表
    try:
        year = datetime.now().year - 1
        ind = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=str(year))
        ind["日期"] = pd.to_datetime(ind["日期"])
        latest = ind.sort_values("日期", ascending=False).iloc[0]
        fin_ind = {
            "return_on_equity": safe_float(latest["净资产收益率(%)"]) / 100,
            "net_margin": safe_float(latest["销售净利率(%)"]) / 100,
            "operating_margin": safe_float(latest["营业利润率(%)"]) / 100,
            "revenue_growth": safe_float(latest["主营业务收入增长率(%)"]) / 100,
            "earnings_growth": safe_float(latest["净利润增长率(%)"]) / 100,
            "book_value_growth": safe_float(latest["净资产增长率(%)"]) / 100,
            "current_ratio": safe_float(latest["流动比率"]),
            "debt_to_equity": safe_float(latest["资产负债率(%)"]) / 100,
            "free_cash_flow_per_share": safe_float(latest["每股经营性现金流(元)"]),
            "earnings_per_share": safe_float(latest["加权每股收益(元)"]),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Akshare indicator fail (%s) → Wind", exc)
        flds = (
            "roe_ttm,net_profit_margin_ttm,op_income_margin_ttm,or_yoy,"
            "net_profit_yoy,bps_yoy,current,debtequity,opcfps,eps_ttm"
        )
        d = fallback_wind_wss(symbol, flds)
        fin_ind = {
            "return_on_equity": safe_float(d.get("roe_ttm")),
            "net_margin": safe_float(d.get("net_profit_margin_ttm")),
            "operating_margin": safe_float(d.get("op_income_margin_ttm")),
            "revenue_growth": safe_float(d.get("or_yoy")),
            "earnings_growth": safe_float(d.get("net_profit_yoy")),
            "book_value_growth": safe_float(d.get("bps_yoy")),
            "current_ratio": safe_float(d.get("current")),
            "debt_to_equity": safe_float(d.get("debtequity")),
            "free_cash_flow_per_share": safe_float(d.get("opcfps")),
            "earnings_per_share": safe_float(d.get("eps_ttm")),
        }

    # price_to_sales
    try:
        inc = ak.stock_financial_report_sina(stock=f"sh{symbol}", symbol="利润表")
        revenue_total = safe_float(inc.iloc[0]["营业总收入"]) if not inc.empty else 0
    except Exception:
        d = fallback_wind_wss(symbol, "tot_oper_rev")
        revenue_total = safe_float(d.get("tot_oper_rev"))

    metrics_all = {
        **stock_rt,
        **fin_ind,
        "price_to_sales": (
            stock_rt.get("market_cap", 0) / revenue_total if revenue_total else 0
        ),
    }
    agent_view = {
        k: metrics_all[k]
        for k in (
            "return_on_equity",
            "net_margin",
            "operating_margin",
            "revenue_growth",
            "earnings_growth",
            "book_value_growth",
            "current_ratio",
            "debt_to_equity",
            "free_cash_flow_per_share",
            "earnings_per_share",
            "pe_ratio",
            "price_to_book",
            "price_to_sales",
        )
    }

    if use_cache:
        try:
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(agent_view, fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("save-cache failed: %s", exc)

    return [agent_view]


# -----------------------------------------------------------------------------
# 📑 财报行项目（最新 & 上期）
# -----------------------------------------------------------------------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """去掉列名中的单位/括号等后缀并清理空格。"""
    pattern = r"[（）()].*?$"
    df.rename(
        columns=lambda x: re.sub(pattern, "", str(x)).strip(),
        inplace=True,
    )
    return df


def _wind_series(symbol: str, mapping: Dict[str, str]) -> pd.Series:
    d = fallback_wind_wss(symbol, ",".join(mapping.values()))
    if not d:
        return pd.Series(dtype=float)
    return pd.Series({alias: safe_float(d.get(fld)) for alias, fld in mapping.items()})


def get_financial_statements(symbol: str) -> List[Dict[str, Any]]:
    logger.info("[statements] %s", symbol)

    def _fetch(table: str, wind_map: Dict[str, str]) -> pd.DataFrame:
        try:
            df = ak.stock_financial_report_sina(stock=f"sh{symbol}", symbol=table)
            if df.empty:
                raise ValueError("empty")
            df = _normalize_cols(df)
            return df.iloc[:2]  # 最新两期
        except Exception:  # noqa: BLE001
            s = _wind_series(symbol, wind_map)
            return pd.DataFrame([s, s])

    bs = _fetch(
        "资产负债表",
        {"流动资产合计": "tot_cur_assets", "流动负债合计": "tot_cur_liab"},
    )
    inc = _fetch(
        "利润表",
        {
            "营业总收入": "tot_oper_rev",
            "营业利润": "oper_profit",
            "净利润": "net_profit",
        },
    )
    cf = _fetch(
        "现金流量表",
        {
            "固定资产折旧、油气资产折耗、生产性生物资产折旧": "depr_ttm",
            "购建固定资产、无形资产和其他长期资产支付的现金": "capex",
            "经营活动产生的现金流量净额": "ncfo",
        },
    )

    # 字段别名列表
    CAPEX_KEYS = [
        "购建固定资产、无形资产和其他长期资产支付的现金",
        "c_pay_acq_const_fiolta",
        "capex",
    ]
    DEPR_KEYS = [
        "固定资产折旧、油气资产折耗、生产性生物资产折旧",
        "depr_ttm",
    ]
    CFO_KEYS = [
        "经营活动产生的现金流量净额",
        "ncfo",
    ]

    def _build(b: pd.Series, i: pd.Series, c: pd.Series) -> Dict[str, float]:
        wc = safe_float(b["流动资产合计"]) - safe_float(b["流动负债合计"])
        capex = abs(_first_exist(c, CAPEX_KEYS))
        return {
            "net_income": safe_float(i["净利润"]),
            "operating_revenue": safe_float(i["营业总收入"]),
            "operating_profit": safe_float(i["营业利润"]),
            "working_capital": wc,
            "depreciation_and_amortization": _first_exist(c, DEPR_KEYS),
            "capital_expenditure": capex,
            "free_cash_flow": _first_exist(c, CFO_KEYS) - capex,
        }

    latest = _build(bs.iloc[0], inc.iloc[0], cf.iloc[0])
    prev = _build(bs.iloc[1], inc.iloc[1], cf.iloc[1])
    return [latest, prev]


# -----------------------------------------------------------------------------
# 📈 市场数据快照
# -----------------------------------------------------------------------------
def _calc_from_hist(code: str, trade_date: str, lookback: int = 260, avg_days: int = 20) -> Dict[str, float]:
    end = datetime.strptime(trade_date, "%Y%m%d").strftime("%Y-%m-%d")
    start = w.tdaysoffset(-lookback, end).Data[0][0].strftime("%Y-%m-%d")
    hist = w.wsd(code, "high,low,volume", start, end, "PriceAdj=F;Fill=Previous")
    if hist.ErrorCode != 0:
        raise RuntimeError(hist.ErrorCode)
    df = pd.DataFrame(hist.Data, index=hist.Fields).T
    return {
        "fifty_two_week_high": safe_float(df["HIGH"].max()),
        "fifty_two_week_low": safe_float(df["LOW"].min()),
        "average_volume": safe_float(df["VOLUME"].tail(avg_days).mean()),
    }


def get_market_data(
    symbol: str,
    trade_date: str | None = None,
    avg_days: int = 20,
) -> Dict[str, Any]:
    if trade_date is None:
        trade_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    _ensure_wind_started()
    wind_code = _wind_code(symbol)
    opts = f"unit=1;tradeDate={trade_date}"
    snap = w.wss(wind_code, "mkt_cap,volume", opts)
    base = {
        "trade_date": datetime.strptime(trade_date, "%Y%m%d").strftime("%Y-%m-%d"),
        "market_cap": safe_float(snap.Data[0][0] if snap.ErrorCode == 0 else None),
        "volume": safe_float(snap.Data[1][0] if snap.ErrorCode == 0 else None),
    }
    try:
        base.update(_calc_from_hist(wind_code, trade_date, 260, avg_days))
    except Exception as exc:  # noqa: BLE001
        logger.error("hist calc fail: %s", exc)
    return base


# -----------------------------------------------------------------------------
# 🕰️ 价格历史 + 技术指标
# -----------------------------------------------------------------------------
def get_price_history_from_wind(symbol: str, start: str, end: str) -> pd.DataFrame:
    _ensure_wind_started()
    res = w.wsd(_wind_code(symbol), "open,high,low,close,volume,amt", start, end, "PriceAdj=F")
    if res.ErrorCode != 0:
        raise RuntimeError(res.ErrorCode)
    df = pd.DataFrame({f: col for f, col in zip(res.Fields, res.Data)})
    df["date"] = pd.to_datetime(res.Times)
    df.rename(
        columns={
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOLUME": "volume",
            "AMT": "amount",
        },
        inplace=True,
    )
    return (
        df[["date", "open", "high", "low", "close", "volume", "amount"]]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )


def get_price_history(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    adjust: str = "qfq",
) -> pd.DataFrame:  # noqa: D401
    end_dt = (
        datetime.strptime(end_date, "%Y-%m-%d")
        if end_date
        else datetime.now() - timedelta(days=1)
    )
    start_dt = (
        datetime.strptime(start_date, "%Y-%m-%d")
        if start_date
        else end_dt - timedelta(days=365)
    )

    def _ak(start: datetime, end: datetime) -> pd.DataFrame:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust=adjust,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "change_amount",
                "换手率": "turnover",
            },
            inplace=True,
        )
        df["date"] = pd.to_datetime(df["date"])
        return df

    df = _ak(start_dt, end_dt)
    if df.empty:
        df = get_price_history_from_wind(
            symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
        )
    if df.empty:
        return df

    # 技术指标示例
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["momentum_1m"] = df["close"].pct_change(20)
    df["momentum_3m"] = df["close"].pct_change(60)
    df["momentum_6m"] = df["close"].pct_change(120)
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_momentum"] = df["volume"] / df["volume_ma20"]

    returns = df["close"].pct_change()
    df["historical_volatility"] = returns.rolling(20).std() * np.sqrt(252)

    vol120 = returns.rolling(120).std() * np.sqrt(252)
    vol_min = vol120.rolling(120).min()
    vol_max = vol120.rolling(120).max()
    df["volatility_regime"] = np.where(
        vol_max - vol_min > 0,
        (df["historical_volatility"] - vol_min) / (vol_max - vol_min),
        0,
    )
    vol_mean = df["historical_volatility"].rolling(120).mean()
    vol_std = df["historical_volatility"].rolling(120).std()
    df["volatility_z_score"] = (df["historical_volatility"] - vol_mean) / vol_std

    tr = pd.DataFrame(
        {
            "h-l": df["high"] - df["low"],
            "h-pc": abs(df["high"] - df["close"].shift(1)),
            "l-pc": abs(df["low"] - df["close"].shift(1)),
        }
    )
    df["atr"] = tr.max(axis=1).rolling(14).mean()
    df["atr_ratio"] = df["atr"] / df["close"]

    log_ret = np.log(df["close"] / df["close"].shift(1))

    def _hurst(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) < 30:
            return np.nan
        lags = range(2, min(11, len(s) // 4))
        tau = [np.mean(s.rolling(lag).std().dropna()) for lag in lags]
        if len([t for t in tau if t > 0]) < 3:
            return np.nan
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return poly[0] / 2

    df["hurst_exponent"] = log_ret.rolling(120, min_periods=60).apply(_hurst, raw=False)
    df["skewness"] = returns.rolling(20).skew()
    df["kurtosis"] = returns.rolling(20).kurt()

    return df


# -----------------------------------------------------------------------------
# 📑 将任意 prices 数组转 DataFrame (列名规范)
# -----------------------------------------------------------------------------
def prices_to_df(prices: List[Dict[str, Any]]) -> pd.DataFrame:  # noqa: D401
    df = pd.DataFrame(prices)
    cn2en = {
        "收盘": "close",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change_amount",
        "换手率": "turnover",
    }
    df.rename(columns={c: cn2en[c] for c in df.columns & cn2en.keys()}, inplace=True)
    for col in ("close", "open", "high", "low", "volume"):
        if col not in df.columns:
            df[col] = 0.0
    return df


# -----------------------------------------------------------------------------
# 🛠️ 简单测试脚本
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sym = "600519"  # 茅台示例

    print("\n=== Wind connection ===")
    _ensure_wind_started()
    print("Wind connected:", w.isconnected())

    print("\n=== Market snapshot ===")
    print(get_market_data(sym))

    print("\n=== Financial metrics ===")
    print(get_financial_metrics(sym, use_cache=False)[0])

    print("\n=== Financial statements (latest & prev) ===")
    latest, prev = get_financial_statements(sym)
    print("Latest:", latest)
    print("Previous:", prev)

    print("\n=== Price history last 5 rows ===")
    ph = get_price_history(
        sym,
        start_date=(datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
    )
    ph.tail().to_excel('C:/Users/13269032233/Desktop/pricehis.xlsx')
    print(ph.tail())
