import os
from typing import Dict, Any, List, Union
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import json
import numpy as np
from src.utils.logging_config import setup_logger

# 设置日志记录
logger = setup_logger('api')



import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import akshare as ak
from WindPy import w                              # pip install WindPy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------- Wind 工具函数 -------------------------------------------------- #
def ensure_wind_started() -> None:
    """确保 WindPy 已启动且连接正常"""
    try:
        if not w.isconnected():
            w.start()
            logger.info("WindPy 启动完成")
    except Exception as e:
        logger.error(f"启动 WindPy 失败: {e}")


def fallback_wind_wss(
    symbol: str,
    fields: str,
    options: str = ""
) -> Dict[str, Any]:
    """
    WindPy wss 兜底调用，将返回值按字段映射为 dict
    若请求失败返回 {}，调用方自行决定后续逻辑
    """
    ensure_wind_started()
    wind_symbol = f"{symbol}.SH" if symbol.startswith("6") else f"{symbol}.SZ"
    try:
        data = w.wss(wind_symbol, fields, options)
        if data.ErrorCode == 0:
            return {f: (data.Data[idx][0] if data.Data[idx] else None)
                    for idx, f in enumerate(fields.split(","))}
        logger.warning(f"Wind wss 错误码 {data.ErrorCode}，fields={fields}")
    except Exception as e:
        logger.error(f"Wind wss 异常: {e}，fields={fields}")
    return {}


# ---------- 主函数一：财务指标 (metrics) ------------------------------------ #
def get_financial_metrics(
    symbol: str,
    cache_dir: str = "src/data/financial_metrics",
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """拉取财务指标，Akshare 为主，Wind 为兜底"""
    logger.info(f"[metrics] Getting financial indicators for {symbol} ...")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol}_metrics.json")

    # ---------- STEP 0: 读取缓存 ----------
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            logger.info("Loaded financial metrics from cache")
            return [cached]
        except Exception as e:
            logger.warning(f"Failed to load cached metrics: {e}")

    # ---------- STEP 1: 实时行情 (市值/估值) ----------
    stock_rt: Dict[str, Any] = {}
    try:
        logger.info("Fetching real-time quotes via Akshare ...")
        realtime = ak.stock_zh_a_spot_em()
        row = realtime[realtime["代码"] == symbol]
        if not row.empty:
            s = row.iloc[0]
            stock_rt = {
                "market_cap": float(s.get("总市值", 0)),
                "float_market_cap": float(s.get("流通市值", 0)),
                "pe_ratio": float(s.get("市盈率-动态", 0)),
                "price_to_book": float(s.get("市净率", 0)),
            }
        else:
            raise ValueError("symbol not in Akshare spot")
    except Exception as e:
        logger.warning(f"Akshare 实时行情失败: {e} → Wind 兜底")
        fields = "mkt_cap_ard,mkt_cap_float,pe_ttm,pb"
        data = fallback_wind_wss(symbol, fields)
        stock_rt = {
            "market_cap": float(data.get("mkt_cap_ard", 0)),
            "float_market_cap": float(data.get("mkt_cap_float", 0)),
            "pe_ratio": float(data.get("pe_ttm", 0)),
            "price_to_book": float(data.get("pb", 0)),
        }

    # ---------- STEP 2: 财务指标表 ----------
    current_year = datetime.now().year
    try:
        logger.info("Fetching financial indicators via Akshare ...")
        df_ind = ak.stock_financial_analysis_indicator(
            symbol=symbol, start_year=str(current_year - 1)
        )
        if df_ind is None or df_ind.empty:
            raise ValueError("Akshare 指标为空")
        df_ind["日期"] = pd.to_datetime(df_ind["日期"])
        latest_financial = df_ind.sort_values("日期", ascending=False).iloc[0]
        fin_ind = {
            "return_on_equity": float(latest_financial["净资产收益率(%)"]) / 100.0,
            "net_margin": float(latest_financial["销售净利率(%)"]) / 100.0,
            "operating_margin": float(latest_financial["营业利润率(%)"]) / 100.0,
            "revenue_growth": float(latest_financial["主营业务收入增长率(%)"]) / 100.0,
            "earnings_growth": float(latest_financial["净利润增长率(%)"]) / 100.0,
            "book_value_growth": float(latest_financial["净资产增长率(%)"]) / 100.0,
            "current_ratio": float(latest_financial["流动比率"]),
            "debt_to_equity": float(latest_financial["资产负债率(%)"]) / 100.0,
            "free_cash_flow_per_share": float(latest_financial["每股经营性现金流(元)"]),
            "earnings_per_share": float(latest_financial["加权每股收益(元)"]),
        }
    except Exception as e:
        logger.warning(f"Akshare 指标失败: {e} → Wind 兜底")
        fields = (
            "roe_ttm,net_profit_margin_ttm,op_income_margin_ttm,"
            "or_yoy,net_profit_yoy,bps_yoy,current,debtequity,"
            "opcfps,eps_ttm"
        )
        data = fallback_wind_wss(symbol, fields)
        fin_ind = {
            "return_on_equity": float(data.get("roe_ttm", 0)),
            "net_margin": float(data.get("net_profit_margin_ttm", 0)),
            "operating_margin": float(data.get("op_income_margin_ttm", 0)),
            "revenue_growth": float(data.get("or_yoy", 0)),
            "earnings_growth": float(data.get("net_profit_yoy", 0)),
            "book_value_growth": float(data.get("bps_yoy", 0)),
            "current_ratio": float(data.get("current", 0)),
            "debt_to_equity": float(data.get("debtequity", 0)),
            "free_cash_flow_per_share": float(data.get("opcfps", 0)),
            "earnings_per_share": float(data.get("eps_ttm", 0)),
        }

    # ---------- STEP 3: 利润表（用于 price_to_sales） ----------
    try:
        logger.info("Fetching income statement via Akshare ...")
        inc = ak.stock_financial_report_sina(
            stock=f"sh{symbol}", symbol="利润表"
        )
        if inc.empty:
            raise ValueError("inc empty")
        latest_income = inc.iloc[0]
        revenue_total = float(latest_income["营业总收入"])
    except Exception as e:
        logger.warning(f"Akshare 利润表失败: {e} → Wind 兜底")
        data = fallback_wind_wss(symbol, "tot_oper_rev")
        revenue_total = float(data.get("tot_oper_rev", 0))

    # ---------- STEP 4: 指标整合 ----------
    metrics_all = {
        **stock_rt,
        **fin_ind,
        "price_to_sales": (
            stock_rt.get("market_cap", 0) / revenue_total
            if revenue_total
            else 0
        ),
        # 把原本完整指标放进这里，如需更多字段可继续添加
    }

    # ---------- STEP 5: Agent 关心的指标子集 ----------
    agent_metrics = {
        # 盈利
        "return_on_equity": metrics_all["return_on_equity"],
        "net_margin": metrics_all["net_margin"],
        "operating_margin": metrics_all["operating_margin"],
        # 增长
        "revenue_growth": metrics_all["revenue_growth"],
        "earnings_growth": metrics_all["earnings_growth"],
        "book_value_growth": metrics_all["book_value_growth"],
        # 健康
        "current_ratio": metrics_all["current_ratio"],
        "debt_to_equity": metrics_all["debt_to_equity"],
        "free_cash_flow_per_share": metrics_all["free_cash_flow_per_share"],
        "earnings_per_share": metrics_all["earnings_per_share"],
        # 估值
        "pe_ratio": metrics_all["pe_ratio"],
        "price_to_book": metrics_all["price_to_book"],
        "price_to_sales": metrics_all["price_to_sales"],
    }

    # ---------- STEP 6: 缓存 ----------
    if use_cache:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(agent_metrics, f, ensure_ascii=False, indent=2)
            logger.info("Saved financial metrics to cache")
        except Exception as e:
            logger.warning(f"Failed to save metrics cache: {e}")

    return [agent_metrics]


# ---------- 主函数二：财报行项目 (statements) ------------------------------- #
def get_financial_statements(symbol: str) -> List[Dict[str, Any]]:
    """获取最新两期关键行项目，Akshare 为主，Wind 兜底"""
    logger.info(f"[statements] Getting financial statements for {symbol} ...")

    # ---------- 工具：通用兜底 ----------
    def _wind_fields_to_series(fields_map: Dict[str, str]) -> pd.Series:
        """按给定 Wind 字段映射抓取一行 Series；抓不到返回空 Series"""
        fields = ",".join(fields_map.values())
        data = fallback_wind_wss(symbol, fields)
        if not data:
            return pd.Series(dtype=float)
        return pd.Series({
            alias: float(data.get(wind_fld, 0))
            for alias, wind_fld in fields_map.items()
        })

    # ---------- 资产负债表 ----------
    try:
        logger.info("Fetching balance sheet via Akshare ...")
        bs = ak.stock_financial_report_sina(
            stock=f"sh{symbol}", symbol="资产负债表"
        )
        if bs.empty:
            raise ValueError("balance sheet empty")
        latest_bs, prev_bs = bs.iloc[0], bs.iloc[1] if len(bs) > 1 else bs.iloc[0]
    except Exception as e:
        logger.warning(f"Akshare 资产负债表失败: {e} → Wind 兜底")
        latest_bs = _wind_fields_to_series({
            "流动资产合计": "tot_cur_assets",
            "流动负债合计": "tot_cur_liab",
        })
        prev_bs = latest_bs.copy()

    # ---------- 利润表 ----------
    try:
        logger.info("Fetching income statement via Akshare ...")
        inc = ak.stock_financial_report_sina(
            stock=f"sh{symbol}", symbol="利润表"
        )
        if inc.empty:
            raise ValueError("income empty")
        latest_inc, prev_inc = inc.iloc[0], inc.iloc[1] if len(inc) > 1 else inc.iloc[0]
    except Exception as e:
        logger.warning(f"Akshare 利润表失败: {e} → Wind 兜底")
        latest_inc = _wind_fields_to_series({
            "营业总收入": "tot_oper_rev",
            "营业利润": "oper_profit",
            "净利润": "net_profit",
        })
        prev_inc = latest_inc.copy()

    # ---------- 现金流量表 ----------
    try:
        logger.info("Fetching cash flow via Akshare ...")
        cf = ak.stock_financial_report_sina(
            stock=f"sh{symbol}", symbol="现金流量表"
        )
        if cf.empty:
            raise ValueError("cashflow empty")
        latest_cf, prev_cf = cf.iloc[0], cf.iloc[1] if len(cf) > 1 else cf.iloc[0]
    except Exception as e:
        logger.warning(f"Akshare 现金流量表失败: {e} → Wind 兜底")
        latest_cf = _wind_fields_to_series({
            "固定资产折旧、油气资产折耗、生产性生物资产折旧": "depr_ttm",
            "购建固定资产、无形资产和其他长期资产支付的现金": "capex",
            "经营活动产生的现金流量净额": "ncfo",
        })
        prev_cf = latest_cf.copy()

    # ---------- 构造两期行项目 ----------
    def _build_item(bs_row: pd.Series, inc_row: pd.Series, cf_row: pd.Series) -> Dict[str, float]:
        try:
            wc = float(bs_row.get("流动资产合计", 0)) - float(bs_row.get("流动负债合计", 0))
            capex_cash = abs(float(cf_row.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
            fcf = float(cf_row.get("经营活动产生的现金流量净额", 0)) - capex_cash
            return {
                "net_income": float(inc_row.get("净利润", 0)),
                "operating_revenue": float(inc_row.get("营业总收入", 0)),
                "operating_profit": float(inc_row.get("营业利润", 0)),
                "working_capital": wc,
                "depreciation_and_amortization": float(
                    cf_row.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)
                    or cf_row.get("depr_ttm", 0)
                ),
                "capital_expenditure": capex_cash,
                "free_cash_flow": fcf,
            }
        except Exception as ex:
            logger.error(f"组装行项目失败: {ex}")
            return {
                "net_income": 0,
                "operating_revenue": 0,
                "operating_profit": 0,
                "working_capital": 0,
                "depreciation_and_amortization": 0,
                "capital_expenditure": 0,
                "free_cash_flow": 0,
            }

    latest_item = _build_item(latest_bs, latest_inc, latest_cf)
    prev_item = _build_item(prev_bs, prev_inc, prev_cf)
    return [latest_item, prev_item]


#######旧
# def get_financial_metrics(symbol: str, cache_dir: str = "src/data/financial_metrics", use_cache: bool = True) -> Dict[str, Any]:
#     """获取财务指标数据"""
#     logger.info(f"Getting financial indicators for {symbol}...")
#     os.makedirs(cache_dir, exist_ok=True)
#     cache_file = os.path.join(cache_dir, f"{symbol}_metrics.json")
#     if use_cache and os.path.exists(cache_file):
#         try:
#             with open(cache_file, "r", encoding="utf-8") as f:
#                 cached = json.load(f)
#             logger.info("Loaded financial metrics from cache")
#             return [cached]
#         except Exception as e:
#             logger.warning(f"Failed to load cached metrics: {e}")
#     try:
#         # 获取实时行情数据（用于市值和估值比率）
#         logger.info("Fetching real-time quotes...")
#         realtime_data = ak.stock_zh_a_spot_em()
#         if realtime_data is None or realtime_data.empty:
#             logger.warning("No real-time quotes data available")
#             return [{}]

#         stock_data = realtime_data[realtime_data['代码'] == symbol]
#         if stock_data.empty:
#             logger.warning(f"No real-time quotes found for {symbol}")
#             return [{}]

#         stock_data = stock_data.iloc[0]
#         logger.info("✓ Real-time quotes fetched")

#         # 获取新浪财务指标
#         logger.info("Fetching Sina financial indicators...")
#         current_year = datetime.now().year
#         financial_data = ak.stock_financial_analysis_indicator(
#             symbol=symbol, start_year=str(current_year-1))
#         if financial_data is None or financial_data.empty:
#             logger.warning("No financial indicator data available")
#             return [{}]

#         # 按日期排序并获取最新的数据
#         financial_data['日期'] = pd.to_datetime(financial_data['日期'])
#         financial_data = financial_data.sort_values('日期', ascending=False)
#         latest_financial = financial_data.iloc[0] if not financial_data.empty else pd.Series(
#         )
#         logger.info(
#             f"✓ Financial indicators fetched ({len(financial_data)} records)")
#         logger.info(f"Latest data date: {latest_financial.get('日期')}")

#         # 获取利润表数据（用于计算 price_to_sales）
#         logger.info("Fetching income statement...")
#         try:
#             income_statement = ak.stock_financial_report_sina(
#                 stock=f"sh{symbol}", symbol="利润表")
#             if not income_statement.empty:
#                 latest_income = income_statement.iloc[0]
#                 logger.info("✓ Income statement fetched")
#             else:
#                 logger.warning("Failed to get income statement")
#                 logger.error("No income statement data found")
#                 latest_income = pd.Series()
#         except Exception as e:
#             logger.warning("Failed to get income statement")
#             logger.error(f"Error getting income statement: {e}")
#             latest_income = pd.Series()

#         # 构建完整指标数据
#         logger.info("Building indicators...")
#         try:
#             def convert_percentage(value: float) -> float:
#                 """将百分比值转换为小数"""
#                 try:
#                     return float(value) / 100.0 if value is not None else 0.0
#                 except:
#                     return 0.0

#             all_metrics = {
#                 # 市场数据
#                 "market_cap": float(stock_data.get("总市值", 0)),
#                 "float_market_cap": float(stock_data.get("流通市值", 0)),

#                 # 盈利数据
#                 "revenue": float(latest_income.get("营业总收入", 0)),
#                 "net_income": float(latest_income.get("净利润", 0)),
#                 "return_on_equity": convert_percentage(latest_financial.get("净资产收益率(%)", 0)),
#                 "net_margin": convert_percentage(latest_financial.get("销售净利率(%)", 0)),
#                 "operating_margin": convert_percentage(latest_financial.get("营业利润率(%)", 0)),

#                 # 增长指标
#                 "revenue_growth": convert_percentage(latest_financial.get("主营业务收入增长率(%)", 0)),
#                 "earnings_growth": convert_percentage(latest_financial.get("净利润增长率(%)", 0)),
#                 "book_value_growth": convert_percentage(latest_financial.get("净资产增长率(%)", 0)),

#                 # 财务健康指标
#                 "current_ratio": float(latest_financial.get("流动比率", 0)),
#                 "debt_to_equity": convert_percentage(latest_financial.get("资产负债率(%)", 0)),
#                 "free_cash_flow_per_share": float(latest_financial.get("每股经营性现金流(元)", 0)),
#                 "earnings_per_share": float(latest_financial.get("加权每股收益(元)", 0)),

#                 # 估值比率
#                 "pe_ratio": float(stock_data.get("市盈率-动态", 0)),
#                 "price_to_book": float(stock_data.get("市净率", 0)),
#                 "price_to_sales": float(stock_data.get("总市值", 0)) / float(latest_income.get("营业总收入", 1)) if float(latest_income.get("营业总收入", 0)) > 0 else 0,
#             }

#             # 只返回 agent 需要的指标
#             agent_metrics = {
#                 # 盈利能力指标
#                 "return_on_equity": all_metrics["return_on_equity"],
#                 "net_margin": all_metrics["net_margin"],
#                 "operating_margin": all_metrics["operating_margin"],

#                 # 增长指标
#                 "revenue_growth": all_metrics["revenue_growth"],
#                 "earnings_growth": all_metrics["earnings_growth"],
#                 "book_value_growth": all_metrics["book_value_growth"],

#                 # 财务健康指标
#                 "current_ratio": all_metrics["current_ratio"],
#                 "debt_to_equity": all_metrics["debt_to_equity"],
#                 "free_cash_flow_per_share": all_metrics["free_cash_flow_per_share"],
#                 "earnings_per_share": all_metrics["earnings_per_share"],

#                 # 估值比率
#                 "pe_ratio": all_metrics["pe_ratio"],
#                 "price_to_book": all_metrics["price_to_book"],
#                 "price_to_sales": all_metrics["price_to_sales"],
#             }

#             logger.info("✓ Indicators built successfully")

#             # 打印所有获取到的指标数据（用于调试）
#             logger.debug("\n获取到的完整指标数据：")
#             for key, value in all_metrics.items():
#                 logger.debug(f"{key}: {value}")

#             logger.debug("\n传递给 agent 的指标数据：")
#             for key, value in agent_metrics.items():
#                 logger.debug(f"{key}: {value}")

#             if use_cache:
#                 try:
#                     with open(cache_file, "w", encoding="utf-8") as f:
#                         json.dump(agent_metrics, f, ensure_ascii=False, indent=2)
#                     logger.info("Saved financial metrics to cache")
#                 except Exception as e:
#                     logger.warning(f"Failed to save metrics cache: {e}")

#             return [agent_metrics]

#         except Exception as e:
#             logger.error(f"Error building indicators: {e}")
#             return [{}]

#     except Exception as e:
#         logger.error(f"Error getting financial indicators: {e}")
#         return [{}]


# def get_financial_statements(symbol: str) -> Dict[str, Any]:
#     """获取财务报表数据"""
#     logger.info(f"Getting financial statements for {symbol}...")
#     try:
#         # 获取资产负债表数据
#         logger.info("Fetching balance sheet...")
#         try:
#             balance_sheet = ak.stock_financial_report_sina(
#                 stock=f"sh{symbol}", symbol="资产负债表")
#             if not balance_sheet.empty:
#                 latest_balance = balance_sheet.iloc[0]
#                 previous_balance = balance_sheet.iloc[1] if len(
#                     balance_sheet) > 1 else balance_sheet.iloc[0]
#                 logger.info("✓ Balance sheet fetched")
#             else:
#                 logger.warning("Failed to get balance sheet")
#                 logger.error("No balance sheet data found")
#                 latest_balance = pd.Series()
#                 previous_balance = pd.Series()
#         except Exception as e:
#             logger.warning("Failed to get balance sheet")
#             logger.error(f"Error getting balance sheet: {e}")
#             latest_balance = pd.Series()
#             previous_balance = pd.Series()

#         # 获取利润表数据
#         logger.info("Fetching income statement...")
#         try:
#             income_statement = ak.stock_financial_report_sina(
#                 stock=f"sh{symbol}", symbol="利润表")
#             if not income_statement.empty:
#                 latest_income = income_statement.iloc[0]
#                 previous_income = income_statement.iloc[1] if len(
#                     income_statement) > 1 else income_statement.iloc[0]
#                 logger.info("✓ Income statement fetched")
#             else:
#                 logger.warning("Failed to get income statement")
#                 logger.error("No income statement data found")
#                 latest_income = pd.Series()
#                 previous_income = pd.Series()
#         except Exception as e:
#             logger.warning("Failed to get income statement")
#             logger.error(f"Error getting income statement: {e}")
#             latest_income = pd.Series()
#             previous_income = pd.Series()

#         # 获取现金流量表数据
#         logger.info("Fetching cash flow statement...")
#         try:
#             cash_flow = ak.stock_financial_report_sina(
#                 stock=f"sh{symbol}", symbol="现金流量表")
#             if not cash_flow.empty:
#                 latest_cash_flow = cash_flow.iloc[0]
#                 previous_cash_flow = cash_flow.iloc[1] if len(
#                     cash_flow) > 1 else cash_flow.iloc[0]
#                 logger.info("✓ Cash flow statement fetched")
#             else:
#                 logger.warning("Failed to get cash flow statement")
#                 logger.error("No cash flow data found")
#                 latest_cash_flow = pd.Series()
#                 previous_cash_flow = pd.Series()
#         except Exception as e:
#             logger.warning("Failed to get cash flow statement")
#             logger.error(f"Error getting cash flow statement: {e}")
#             latest_cash_flow = pd.Series()
#             previous_cash_flow = pd.Series()

#         # 构建财务数据
#         line_items = []
#         try:
#             # 处理最新期间数据
#             current_item = {
#                 # 从利润表获取
#                 "net_income": float(latest_income.get("净利润", 0)),
#                 "operating_revenue": float(latest_income.get("营业总收入", 0)),
#                 "operating_profit": float(latest_income.get("营业利润", 0)),

#                 # 从资产负债表计算营运资金
#                 "working_capital": float(latest_balance.get("流动资产合计", 0)) - float(latest_balance.get("流动负债合计", 0)),

#                 # 从现金流量表获取
#                 "depreciation_and_amortization": float(latest_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
#                 "capital_expenditure": abs(float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
#                 "free_cash_flow": float(latest_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
#             }
#             line_items.append(current_item)
#             logger.info("✓ Latest period data processed successfully")

#             # 处理上一期间数据
#             previous_item = {
#                 "net_income": float(previous_income.get("净利润", 0)),
#                 "operating_revenue": float(previous_income.get("营业总收入", 0)),
#                 "operating_profit": float(previous_income.get("营业利润", 0)),
#                 "working_capital": float(previous_balance.get("流动资产合计", 0)) - float(previous_balance.get("流动负债合计", 0)),
#                 "depreciation_and_amortization": float(previous_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
#                 "capital_expenditure": abs(float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
#                 "free_cash_flow": float(previous_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
#             }
#             line_items.append(previous_item)
#             logger.info("✓ Previous period data processed successfully")

#         except Exception as e:
#             logger.error(f"Error processing financial data: {e}")
#             default_item = {
#                 "net_income": 0,
#                 "operating_revenue": 0,
#                 "operating_profit": 0,
#                 "working_capital": 0,
#                 "depreciation_and_amortization": 0,
#                 "capital_expenditure": 0,
#                 "free_cash_flow": 0
#             }
#             line_items = [default_item, default_item]

#         return line_items

#     except Exception as e:
#         logger.error(f"Error getting financial statements: {e}")
#         default_item = {
#             "net_income": 0,
#             "operating_revenue": 0,
#             "operating_profit": 0,
#             "working_capital": 0,
#             "depreciation_and_amortization": 0,
#             "capital_expenditure": 0,
#             "free_cash_flow": 0
#         }
#         return [default_item, default_item]
# #############旧版################
def get_price_history_from_wind(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    使用 Wind 接口获取股票历史价格数据（前复权），字段与 Akshare 格式兼容。
    """
    try:
        from WindPy import w
        w.start()

        # 补全 Wind 股票代码后缀
        def add_wind_suffix(code):
            if code.startswith(('6', '9')):
                return f"{code}.SH"
            elif code.startswith(('0', '3')):
                return f"{code}.SZ"
            else:
                raise ValueError(f"Invalid stock code: {code}")

        wind_code = add_wind_suffix(symbol)

        fields = "open,high,low,close,volume,amt"
        result = w.wsd(wind_code, fields, start_date, end_date, "PriceAdj=F")

        if result.ErrorCode != 0:
            raise ValueError(f"Wind WSD Error: {result.Data}")

        df = pd.DataFrame({field: col for field, col in zip(result.Fields, result.Data)})
        df["date"] = pd.to_datetime(result.Times)
        df = df.rename(columns={
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOLUME": "volume",
            "AMT": "amount"
        })

        df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
        df = df.dropna()
        df = df.sort_values("date").reset_index(drop=True)

        return df

    except Exception as e:
        logger.error(f"Error fetching price history from Wind: {e}")
        return pd.DataFrame()

# def get_market_data(symbol: str) -> Dict[str, Any]:
#     """获取市场数据"""
#     try:
#         # 获取实时行情
#         realtime_data = ak.stock_zh_a_spot_em()
#         stock_data = realtime_data[realtime_data['代码'] == symbol].iloc[0]

#         return {
#             "market_cap": float(stock_data.get("总市值", 0)),
#             "volume": float(stock_data.get("成交量", 0)),
#             # A股没有平均成交量，暂用当日成交量
#             "average_volume": float(stock_data.get("成交量", 0)),
#             "fifty_two_week_high": float(stock_data.get("52周最高", 0)),
#             "fifty_two_week_low": float(stock_data.get("52周最低", 0))
#         }

#     except Exception as e:
#         logger.error(f"Error getting market data: {e}")
#         return {}

def _ensure_wind_started() -> None:
    """启动 WindPy，如果已经连好则忽略。"""
    if not w.isconnected():
        w.start()

def _calc_from_hist(code: str,
                    trade_date: str,
                    lookback: int = 260,
                    avg_days: int = 20
                    ) -> Dict[str, Union[str, float]]:
    """
    用 w.wsd 取过去 lookback 个交易日的 high/low/volume，
    计算 52 周高低价和 N 日均量（默认 20 日）。
    """
    # 1) 计算日期区间：先把 trade_date 转回 Wind 支持的格式
    end = datetime.strptime(trade_date, "%Y%m%d").strftime("%Y-%m-%d")
    # 往前推 lookback 个交易日
    start_offset = w.tdaysoffset(-lookback, end).Data[0][0].strftime("%Y-%m-%d")  # :contentReference[oaicite:4]{index=4}

    # 2) 拉取历史序列
    hist = w.wsd(
        code,
        "high,low,volume",
        start_offset,
        end,
        "PriceAdj=F;Fill=Previous;CurrencyType="  # 前复权并补齐缺口
    )

    if hist.ErrorCode != 0:
        raise RuntimeError(f"WSD error {hist.ErrorCode} for {code}")

    df = pd.DataFrame(hist.Data, index=hist.Fields).T  # 高、低、量
    fifty_two_week_high = float(df["HIGH"].max())
    fifty_two_week_low = float(df["LOW"].min())
    average_volume = float(df["VOLUME"].tail(avg_days).mean())

    return dict(
        fifty_two_week_high=fifty_two_week_high,
        fifty_two_week_low=fifty_two_week_low,
        average_volume=average_volume,
    )

def get_market_data(symbol: str,
                       trade_date: str,
                       avg_days: int = 20
                       ) -> Dict[str, Any]:
    """
    通用版本：总市值、当日成交量、N 日均量、52 周高/低。
    自动按沪/深市场补后缀，量化字段缺失就回退手算。
    """
    _ensure_wind_started()

    wind_symbol = f"{symbol}.SH" if symbol.startswith(("6", "9")) else f"{symbol}.SZ"

    # ---------- 1) 量化字段优先 ----------
    fields_qlib = "mkt_cap,volume"
    opts = f"unit=1;tradeDate={trade_date}"
    data = w.wss(wind_symbol, fields_qlib, opts)

    if data.ErrorCode == 0 and all(v is not None for v in data.Data[2:]):
        return {
            "trade_date": datetime.strptime(trade_date, "%Y%m%d").strftime("%Y-%m-%d"),
            "market_cap": float(data.Data[0][0] or 0),
            "volume": float(data.Data[1][0] or 0)
        }


    try:
        hist_stats = _calc_from_hist(wind_symbol, trade_date, 260, avg_days)
        # 今日快照（总市值 & 今日量）
        snap = w.wss(wind_symbol, "mkt_cap,volume", opts)

        if snap.ErrorCode != 0:
            raise RuntimeError(f"WSS snapshot error {snap.ErrorCode}")

        return {
            "trade_date": datetime.strptime(trade_date, "%Y%m%d").strftime("%Y-%m-%d"),
            "market_cap": float(snap.Data[0][0] or 0),
            "volume": float(snap.Data[1][0] or 0),
            **hist_stats,
        }
    except Exception as e:

        import logging
        logging.getLogger(__name__).error(f"[Wind] fallback failed: {e}")
        return {}


def get_price_history(symbol: str, start_date: str = None, end_date: str = None, adjust: str = "qfq") -> pd.DataFrame:
    """获取历史价格数据

    Args:
        symbol: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD，如果为None则默认获取过去一年的数据
        end_date: 结束日期，格式：YYYY-MM-DD，如果为None则使用昨天作为结束日期
        adjust: 复权类型，可选值：
               - "": 不复权
               - "qfq": 前复权（默认）
               - "hfq": 后复权

    Returns:
        包含以下列的DataFrame：
        - date: 日期
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量（手）
        - amount: 成交额（元）
        - amplitude: 振幅（%）
        - pct_change: 涨跌幅（%）
        - change_amount: 涨跌额（元）
        - turnover: 换手率（%）

        技术指标：
        - momentum_1m: 1个月动量
        - momentum_3m: 3个月动量
        - momentum_6m: 6个月动量
        - volume_momentum: 成交量动量
        - historical_volatility: 历史波动率
        - volatility_regime: 波动率区间
        - volatility_z_score: 波动率Z分数
        - atr_ratio: 真实波动幅度比率
        - hurst_exponent: 赫斯特指数
        - skewness: 偏度
        - kurtosis: 峰度
    """
    try:
        # 获取当前日期和昨天的日期
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)

        # 如果没有提供日期，默认使用昨天作为结束日期
        if not end_date:
            end_date = yesterday  # 使用昨天作为结束日期
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            # 确保end_date不会超过昨天
            if end_date > yesterday:
                end_date = yesterday

        if not start_date:
            start_date = end_date - timedelta(days=365)  # 默认获取一年的数据
        else:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        logger.info(f"\nGetting price history for {symbol}...")
        logger.info(f"Start date: {start_date.strftime('%Y-%m-%d')}")
        logger.info(f"End date: {end_date.strftime('%Y-%m-%d')}")

        def get_and_process_data(start_date, end_date):
            """获取并处理数据，包括重命名列等操作"""
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust=adjust
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # 重命名列以匹配技术分析代理的需求
            df = df.rename(columns={
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
                "换手率": "turnover"
            })

            # 确保日期列为datetime类型
            df["date"] = pd.to_datetime(df["date"])
            return df

        # 获取历史行情数据
        df = get_and_process_data(start_date, end_date)

        if df is None or df.empty:
            logger.warning(
                f"Warning: No price history data found for {symbol}")
            df = get_price_history_from_wind(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            if df.empty:
                logger.error("Failed to retrieve data from both Akshare and Wind.")
                return pd.DataFrame()

        # 检查数据量是否足够
        min_required_days = 120  # 至少需要120个交易日的数据
        if len(df) < min_required_days:
            logger.warning(
                f"Warning: Insufficient data ({len(df)} days) for all technical indicators")
            logger.info("Attempting to fetch more data...")

            # 扩大时间范围到2年
            start_date = end_date - timedelta(days=730)
            df = get_and_process_data(start_date, end_date)

            if len(df) < min_required_days:
                logger.warning(
                    f"Warning: Even with extended time range, insufficient data ({len(df)} days)")

        # 计算动量指标
        df["momentum_1m"] = df["close"].pct_change(periods=20)  # 20个交易日约等于1个月
        df["momentum_3m"] = df["close"].pct_change(periods=60)  # 60个交易日约等于3个月
        df["momentum_6m"] = df["close"].pct_change(
            periods=120)  # 120个交易日约等于6个月

        # 计算成交量动量（相对于20日平均成交量的变化）
        df["volume_ma20"] = df["volume"].rolling(window=20).mean()
        df["volume_momentum"] = df["volume"] / df["volume_ma20"]

        # 计算波动率指标
        # 1. 历史波动率 (20日)
        returns = df["close"].pct_change()
        df["historical_volatility"] = returns.rolling(
            window=20).std() * np.sqrt(252)  # 年化

        # 2. 波动率区间 (相对于过去120天的波动率的位置)
        volatility_120d = returns.rolling(window=120).std() * np.sqrt(252)
        vol_min = volatility_120d.rolling(window=120).min()
        vol_max = volatility_120d.rolling(window=120).max()
        vol_range = vol_max - vol_min
        df["volatility_regime"] = np.where(
            vol_range > 0,
            (df["historical_volatility"] - vol_min) / vol_range,
            0  # 当范围为0时返回0
        )

        # 3. 波动率Z分数
        vol_mean = df["historical_volatility"].rolling(window=120).mean()
        vol_std = df["historical_volatility"].rolling(window=120).std()
        df["volatility_z_score"] = (
            df["historical_volatility"] - vol_mean) / vol_std

        # 4. ATR比率
        tr = pd.DataFrame()
        tr["h-l"] = df["high"] - df["low"]
        tr["h-pc"] = abs(df["high"] - df["close"].shift(1))
        tr["l-pc"] = abs(df["low"] - df["close"].shift(1))
        tr["tr"] = tr[["h-l", "h-pc", "l-pc"]].max(axis=1)
        df["atr"] = tr["tr"].rolling(window=14).mean()
        df["atr_ratio"] = df["atr"] / df["close"]

        # 计算统计套利指标
        # 1. 赫斯特指数 (使用过去120天的数据)
        def calculate_hurst(series):
            """
            计算Hurst指数。

            Args:
                series: 价格序列

            Returns:
                float: Hurst指数，或在计算失败时返回np.nan
            """
            try:
                series = series.dropna()
                if len(series) < 30:  # 降低最小数据点要求
                    return np.nan

                # 使用对数收益率
                log_returns = np.log(series / series.shift(1)).dropna()
                if len(log_returns) < 30:  # 降低最小数据点要求
                    return np.nan

                # 使用更小的lag范围
                # 减少lag范围到2-10天
                lags = range(2, min(11, len(log_returns) // 4))

                # 计算每个lag的标准差
                tau = []
                for lag in lags:
                    # 计算滚动标准差
                    std = log_returns.rolling(window=lag).std().dropna()
                    if len(std) > 0:
                        tau.append(np.mean(std))

                # 基本的数值检查
                if len(tau) < 3:  # 进一步降低最小要求
                    return np.nan

                # 使用对数回归
                lags_log = np.log(list(lags))
                tau_log = np.log(tau)

                # 计算回归系数
                reg = np.polyfit(lags_log, tau_log, 1)
                hurst = reg[0] / 2.0

                # 只保留基本的数值检查
                if np.isnan(hurst) or np.isinf(hurst):
                    return np.nan

                return hurst

            except Exception as e:
                return np.nan

        # 使用对数收益率计算Hurst指数
        log_returns = np.log(df["close"] / df["close"].shift(1))
        df["hurst_exponent"] = log_returns.rolling(
            window=120,
            min_periods=60  # 要求至少60个数据点
        ).apply(calculate_hurst)

        # 2. 偏度 (20日)
        df["skewness"] = returns.rolling(window=20).skew()

        # 3. 峰度 (20日)
        df["kurtosis"] = returns.rolling(window=20).kurt()

        # 按日期升序排序
        df = df.sort_values("date")

        # 重置索引
        df = df.reset_index(drop=True)

        logger.info(
            f"Successfully fetched price history data ({len(df)} records)")

        # 检查并报告NaN值
        nan_columns = df.isna().sum()
        if nan_columns.any():
            logger.warning(
                "\nWarning: The following indicators contain NaN values:")
            for col, nan_count in nan_columns[nan_columns > 0].items():
                logger.warning(f"- {col}: {nan_count} records")

        return df

    except Exception as e:
        logger.error(f"Error getting price history: {e}")
        return pd.DataFrame()


def prices_to_df(prices):
    """Convert price data to DataFrame with standardized column names"""
    try:
        df = pd.DataFrame(prices)

        # 标准化列名映射
        column_mapping = {
            '收盘': 'close',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change_percent',
            '涨跌额': 'change_amount',
            '换手率': 'turnover_rate'
        }

        # 重命名列
        for cn, en in column_mapping.items():
            if cn in df.columns:
                df[en] = df[cn]

        # 确保必要的列存在
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0  # 使用0填充缺失的必要列

        return df
    except Exception as e:
        logger.error(f"Error converting price data: {str(e)}")
        # 返回一个包含必要列的空DataFrame
        return pd.DataFrame(columns=['close', 'open', 'high', 'low', 'volume'])


def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """获取股票价格数据

    Args:
        ticker: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD

    Returns:
        包含价格数据的DataFrame
    """
    return get_price_history(ticker, start_date, end_date)



# import unittest
# import pandas as pd
# from datetime import datetime, timedelta
# import logging

# # 配置日志（可选）
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class TestFinancialData(unittest.TestCase):
#     SYMBOL = "688787"  # 贵州茅台（有效股票代码）
#     START_DATE = "2024-01-01"
#     END_DATE = "2025-03-31"


#     def test_financial_statements_values(self):
#         """测试财务报表具体数值"""
#         statements = get_financial_statements(self.SYMBOL)
#         self.assertEqual(len(statements), 2, "应该包含两个期间的报表")

#         # 定义需要检查的关键指标
#         key_metrics = {
#             "current": {
#                 "net_income": "净利润",
#                 "operating_revenue": "营业总收入",
#                 "free_cash_flow": "自由现金流",
#                 "working_capital": "营运资金"
#             },
#             "previous": {
#                 "net_income": "净利润",
#                 "operating_revenue": "营业总收入",
#                 "free_cash_flow": "自由现金流",
#                 "working_capital": "营运资金"
#             }
#         }

#         print("\n### 财务报表详细数值测试 ###")
#         for i, period in enumerate(statements):
#             period_type = "current" if i == 0 else "previous"
#             print(f"\n--- {period_type.upper()} PERIOD ---")
            
#             for metric, name in key_metrics[period_type].items():
#                 value = period.get(metric, 0)
                
#                 # 基本数值验证
#                 self.assertIsInstance(value, (int, float)), f"{name} 数据类型错误"
#                 self.assertGreaterEqual(value, -1e18), f"{name} 出现异常负值"
                
#                 # 打印详细数值
#                 print(f"{name}: {value}")
                
#                 # 关键指标阈值验证（示例）
#                 if metric == "net_income":
#                     self.assertGreater(value, -1e12, "净利润出现极端负值")
#                 if metric == "operating_revenue":
#                     self.assertGreater(value, 0, "营业收入不能为零")

#         # 跨期间增长验证（示例）
#         current_rev = statements[0]["operating_revenue"]
#         prev_rev = statements[1]["operating_revenue"]
#         if prev_rev != 0:
#             growth_rate = (current_rev - prev_rev) / prev_rev
#             print(f"\n营业收入增长率: {growth_rate:.2%}")
#             self.assertGreater(growth_rate, -0.5, "营业收入同比下降超过50%")
            


# if __name__ == '__main__':
#     unittest.main()