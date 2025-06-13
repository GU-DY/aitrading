from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
import json

import json
from langchain.schema import HumanMessage
from WindPy import w


def add_wind_suffix(stock_code: str) -> str:
    """
    为股票代码补充 Wind 后缀。根据股票代码前缀判断是沪市还是深市。
    """
    if stock_code.startswith(('6', '9')):  # 沪市
        return f"{stock_code}.SH"
    elif stock_code.startswith(('0', '3')):  # 深市
        return f"{stock_code}.SZ"
    else:
        raise ValueError(f"无法识别的股票代码前缀: {stock_code}")


def valuation_agent(state: AgentState):
    """Responsible for valuation analysis with corrected calculations that handle negative growth."""
    show_workflow_status("Valuation Agent")
    show_reasoning = state["metadata"]["show_reasoning"]

    data = state["data"]
    metrics = data["financial_metrics"][0]
    current_financial_line_item = data["financial_line_items"][0]
    previous_financial_line_item = data["financial_line_items"][1]
    market_cap = data["market_cap"]

    # Instead of forcing 0.05 <= growth <= 0.3, we allow -0.3 to +0.3 as "normal"
    # Adjust this range to whatever your acceptable thresholds may be.
    if not -0.3 <= metrics["earnings_growth"] <= 0.3:
        print(f"Warning: Extreme growth rate {metrics['earnings_growth']}")

    # Validate market_cap
    if market_cap <= 0:
        try:
            w.start()
            stock_code = data.get("ticker", "600519.SH")  # 默认使用贵州茅台
            stock_code = add_wind_suffix(stock_code)
            data_df = w.wss(stock_code, "mkt_cap", usedf=True)[1]
            market_cap = data_df.iloc[0, 0]
            if market_cap <= 0:
                raise ValueError("Fetched market cap is still invalid.")
        except Exception as e:
                raise ValueError(f"Failed to fetch market cap from Wind: {e}")

    reasoning = {}

    # Calculate the change in working capital
    working_capital_change = calculate_working_capital_change(
        current=current_financial_line_item.get('working_capital', 0),
        previous=previous_financial_line_item.get('working_capital', 0)
    )

    # Compute Owner Earnings valuation (revised to allow negative growth)
    owner_earnings_value = calculate_owner_earnings_value(
        net_income=current_financial_line_item.get('net_income', 0),
        depreciation=current_financial_line_item.get('depreciation_and_amortization', 0),
        capex=current_financial_line_item.get('capital_expenditure', 0),
        working_capital_change=working_capital_change,
        growth_rate=metrics["earnings_growth"],
        required_return=0.12,      # Adjusted to 12% from original 15%
        margin_of_safety=0.20,    # 20% margin of safety
        num_years=5
    )

    # Compute DCF valuation (revised to allow negative growth)
    dcf_value = calculate_intrinsic_value(
        free_cash_flow=current_financial_line_item.get('free_cash_flow', 0),
        growth_rate=metrics["earnings_growth"],
        discount_rate=0.08,        # Adjusted discount rate
        terminal_growth_rate=0.025, # More realistic terminal growth rate
        num_years=5,
    )

    # Calculate valuation gaps
    dcf_gap = (dcf_value - market_cap) / market_cap
    owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap
    valuation_gap = (dcf_gap + owner_earnings_gap) / 2

    # Generate signal based on combined valuation gap
    signal_rules = {
        'bullish': valuation_gap > 0.15,   # 15%+ undervalued
        'bearish': valuation_gap < -0.25,  # 25%+ overvalued
        'neutral': True
    }
    signal = next(k for k, v in signal_rules.items() if v)

    # Build reasoning details
    reasoning["dcf_analysis"] = {
        "signal": "bullish" if dcf_gap > 0.15 else "bearish" if dcf_gap < -0.25 else "neutral",
        "details": f"Intrinsic Value: {dcf_value:,.2f}, Market Cap: {market_cap:,.2f}, Gap: {dcf_gap:.1%}"
    }
    reasoning["owner_earnings_analysis"] = {
        "signal": "bullish" if owner_earnings_gap > 0.15 else "bearish" if owner_earnings_gap < -0.25 else "neutral",
        "details": f"Owner Earnings Value: {owner_earnings_value:,.2f}, Market Cap: {market_cap:,.2f}, Gap: {owner_earnings_gap:.1%}"
    }

    message_content = {
        "signal": signal,
        "confidence": f"{abs(valuation_gap):.0%}",
        "reasoning": reasoning
    }

    # Prepare output message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="valuation_agent",
    )

    # Optionally show reasoning in logs
    if show_reasoning:
        show_agent_reasoning(message_content, "Valuation Analysis Agent")
        state["metadata"]["agent_reasoning"] = message_content

    show_workflow_status("Valuation Agent", "completed")

    return {
        "messages": [message],
        "data": {**data, "valuation_analysis": message_content},
        "metadata": state["metadata"],
    }


def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.12,
    margin_of_safety: float = 0.20,
    num_years: int = 5
) -> float:
    """
    Revised Owner Earnings model that handles negative growth gracefully.
    """
    try:
        # Basic type checks
        inputs = [net_income, depreciation, capex, working_capital_change]
        if not all(isinstance(v, (int, float)) for v in inputs):
            print("Invalid input types for Owner Earnings calculation.")
            return 0.0

        # Compute base owner earnings
        owner_earnings = net_income + depreciation - capex - working_capital_change
        print(f"[DEBUG] Owner Earnings Components: "
              f"NI={net_income}, Dep={depreciation}, Capex={capex}, WCΔ={working_capital_change} => OE={owner_earnings}")

        # If negative, decide how to handle. Here, we take absolute value for projection
        # to reflect the possibility of improvement in the future.
        if owner_earnings <= 0:
            print(f"Warning: Negative Owner Earnings {owner_earnings}, using absolute value for projection.")
            owner_earnings = abs(owner_earnings)

        # Use actual growth rate, allowing it to be negative
        # (or optionally clamp if you want some sanity limits, e.g. -0.3 to 0.3).
        base_growth = growth_rate

        # Optionally adjust required return for extreme (positive or negative) growth
        if abs(growth_rate) > 0.15:
            required_return += 0.02
            print(f"Adjusted required return to {required_return:.1%} for extreme growth {growth_rate:.1%}")

        # Calculate undiscounted future values
        undiscounted = []
        for year in range(1, num_years + 1):
            future_value = owner_earnings * ((1 + base_growth) ** year)
            undiscounted.append(future_value)
            print(f"[DEBUG] Year {year}: Growth={base_growth:.1%}, FV={future_value:.2f}")

        # Estimate a terminal growth assumption; for positive growth we might reduce it,
        # for negative growth we might assume it becomes less negative, etc.
        # Below is just an example logic:
        if base_growth >= 0:
            # If growth is positive, halve it or cap at 3.5%
            terminal_growth = min(base_growth * 0.5, 0.035)
        else:
            # If growth is negative, reduce negativity or cap at -2%
            terminal_growth = max(base_growth * 0.5, -0.02)

        terminal_cf = undiscounted[-1] * (1 + terminal_growth)
        if required_return <= terminal_growth:
            # Fallback if the difference is too small or negative
            terminal_value = 0.0
            print("Warning: Required return <= terminal_growth, setting terminal value to 0.")
        else:
            terminal_value = terminal_cf / (required_return - terminal_growth)

        # Discount everything back
        discounted = [
            fv / ((1 + required_return) ** i)
            for i, fv in enumerate(undiscounted, 1)
        ]
        terminal_discounted = terminal_value / ((1 + required_return) ** num_years)
        total_value = sum(discounted) + terminal_discounted

        # Apply margin of safety
        value_after_margin = total_value * (1 - margin_of_safety)
        return max(value_after_margin, 0.0)

    except Exception as e:
        print(f"Owner Earnings Error: {str(e)}")
        return 0.0


def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.08,
    terminal_growth_rate: float = 0.03,
    num_years: int = 5,
) -> float:
    """
    Revised DCF model that allows negative growth.
    """
    try:
        # Handle negative/zero FCF. Here, we let negative be negative,
        # or do a custom approach if you prefer.
        if free_cash_flow < 0:
            print(f"Warning: Negative FCF={free_cash_flow}, continuing with negative value in the model.")

        base_growth = growth_rate

        # Optionally adjust discount rate for extreme growth (positive or negative)
        if abs(base_growth) > 0.15:
            discount_rate += 0.02
            print(f"[DCF] Adjusted discount rate to {discount_rate:.1%} for extreme growth={base_growth:.1%}")

        present_values = []
        # Project free cash flows over num_years
        for year in range(1, num_years + 1):
            future_cf = free_cash_flow * ((1 + base_growth) ** year)
            pv = future_cf / ((1 + discount_rate) ** year)
            present_values.append(pv)
            print(f"[DCF] Year={year}, CF={future_cf:.2f}, PV={pv:.2f}")

        # Terminal value: if growth is negative, we can either clamp terminal growth or allow it
        # For example, let final_growth be base_growth * 0.6
        final_growth = base_growth * 0.6
        # Ensure discount_rate > final_growth to avoid division by zero or negative
        if discount_rate <= final_growth:
            print("Warning: discount_rate <= final_growth; forcing terminal value to 0.")
            terminal_value = 0.0
        else:
            terminal_cf = free_cash_flow * ((1 + base_growth) ** num_years) * (1 + final_growth)
            terminal_value = terminal_cf / (discount_rate - final_growth)

        terminal_pv = terminal_value / ((1 + discount_rate) ** num_years)
        print(f"[DCF] Terminal CF={terminal_value:.2f}, PV(Terminal)={terminal_pv:.2f}")

        return sum(present_values) + terminal_pv

    except Exception as e:
        print(f"DCF Error: {str(e)}")
        return 0.0


def calculate_working_capital_change(current: float, previous: float) -> float:
    """
    Computes the change in working capital (current - previous).
    """
    if all(isinstance(v, (int, float)) for v in [current, previous]):
        return current - previous
    return 0


# def valuation_agent(state: AgentState):
#     """Responsible for valuation analysis"""
#     show_workflow_status("Valuation Agent")
#     show_reasoning = state["metadata"]["show_reasoning"]
#     data = state["data"]
#     metrics = data["financial_metrics"][0]
#     current_financial_line_item = data["financial_line_items"][0]
#     previous_financial_line_item = data["financial_line_items"][1]
#     market_cap = data["market_cap"]

#     reasoning = {}

#     # Calculate working capital change
#     working_capital_change = (current_financial_line_item.get(
#         'working_capital') or 0) - (previous_financial_line_item.get('working_capital') or 0)

#     # Owner Earnings Valuation (Buffett Method)
#     owner_earnings_value = calculate_owner_earnings_value(
#         net_income=current_financial_line_item.get('net_income'),
#         depreciation=current_financial_line_item.get(
#             'depreciation_and_amortization'),
#         capex=current_financial_line_item.get('capital_expenditure'),
#         working_capital_change=working_capital_change,
#         growth_rate=metrics["earnings_growth"],
#         required_return=0.15,
#         margin_of_safety=0.25
#     )
    

#     # DCF Valuation
#     dcf_value = calculate_intrinsic_value(
#         free_cash_flow=current_financial_line_item.get('free_cash_flow'),
#         growth_rate=metrics["earnings_growth"],
#         discount_rate=0.10,
#         terminal_growth_rate=0.03,
#         num_years=5,
#     )
   
#     # Calculate combined valuation gap (average of both methods)
#     print(f'dcf_value: {dcf_value}')
#     dcf_gap = (dcf_value - market_cap) / market_cap
#     owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap
#     valuation_gap = (dcf_gap + owner_earnings_gap) / 2

#     if valuation_gap > 0.10:  # Changed from 0.15 to 0.10 (10% undervalued)
#         signal = 'bullish'
#     elif valuation_gap < -0.20:  # Changed from -0.15 to -0.20 (20% overvalued)
#         signal = 'bearish'
#     else:
#         signal = 'neutral'

#     reasoning["dcf_analysis"] = {
#         "signal": "bullish" if dcf_gap > 0.10 else "bearish" if dcf_gap < -0.20 else "neutral",
#         "details": f"Intrinsic Value: ${dcf_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {dcf_gap:.1%}"
#     }

#     reasoning["owner_earnings_analysis"] = {
#         "signal": "bullish" if owner_earnings_gap > 0.10 else "bearish" if owner_earnings_gap < -0.20 else "neutral",
#         "details": f"Owner Earnings Value: ${owner_earnings_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {owner_earnings_gap:.1%}"
#     }

#     message_content = {
#         "signal": signal,
#         "confidence": f"{abs(valuation_gap):.0%}",
#         "reasoning": reasoning
#     }

#     message = HumanMessage(
#         content=json.dumps(message_content),
#         name="valuation_agent",
#     )

#     if show_reasoning:
#         show_agent_reasoning(message_content, "Valuation Analysis Agent")
#         # 保存推理信息到metadata供API使用
#         state["metadata"]["agent_reasoning"] = message_content

#     show_workflow_status("Valuation Agent", "completed")
#     return {
#         "messages": [message],
#         "data": {
#             **data,
#             "valuation_analysis": message_content
#         },
#         "metadata": state["metadata"],
#     }


# def calculate_owner_earnings_value(
#     net_income: float,
#     depreciation: float,
#     capex: float,
#     working_capital_change: float,
#     growth_rate: float = 0.05,
#     required_return: float = 0.12,
#     margin_of_safety: float = 0.15,
#     num_years: int = 5
# ) -> float:
#     """
#     使用改进的所有者收益法计算公司价值。

#     Args:
#         net_income: 净利润
#         depreciation: 折旧和摊销
#         capex: 资本支出
#         working_capital_change: 营运资金变化
#         growth_rate: 预期增长率
#         required_return: 要求回报率
#         margin_of_safety: 安全边际
#         num_years: 预测年数

#     Returns:
#         float: 计算得到的公司价值
#     """
#     try:
#         # 数据有效性检查
#         if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]):
#             return 0

#         # 计算初始所有者收益
#         owner_earnings = (
#             net_income +
#             depreciation -
#             capex -
#             working_capital_change
#         )

#         if owner_earnings <= 0:
#             return 0

#         # 调整增长率，确保合理性
#         growth_rate = min(max(growth_rate, 0), 0.25)  # 限制在0-25%之间

#         # 计算预测期收益现值
#         future_values = []
#         undiscounted = []
#         discounted = [] 
#         for year in range(1, num_years + 1):
#             # 使用递减增长率模型
#             year_growth = growth_rate * (1 - year / (2 * num_years))  # 增长率逐年递减
#             future_value = owner_earnings * (1 + year_growth) ** year
#             discounted_value = future_value / (1 + required_return) ** year
#             future_values.append(discounted_value)
#             discounted.append(discounted_value)
#             undiscounted.append(owner_earnings)

#         # 计算永续价值
#         # terminal_growth = min(growth_rate * 0.4, 0.03)  # 永续增长率取增长率的40%或3%的较小值
#         # terminal_value = (
#         #     future_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
#         # terminal_value_discounted = terminal_value / \
#         #     (1 + required_return) ** num_years
        
#         terminal_growth = min(growth_rate * 0.4, 0.03)
#         terminal_cf = undiscounted[-1] * (1 + terminal_growth)
#         terminal_value = terminal_cf / (required_return - terminal_growth)
#         terminal_value_discounted = terminal_value / (1 + required_return) ** num_years
        
#         # 计算总价值并应用安全边际
#         intrinsic_value = sum(future_values) + terminal_value_discounted
#         value_with_safety_margin = intrinsic_value * (1 - margin_of_safety)

#         return max(value_with_safety_margin, 0)  # 确保不返回负值

#     except Exception as e:
#         print(f"所有者收益计算错误: {e}")
#         return 0


# def calculate_intrinsic_value(
#     free_cash_flow: float,
#     growth_rate: float = 0.05,
#     discount_rate: float = 0.10,
#     terminal_growth_rate: float = 0.02,
#     num_years: int = 5,
# ) -> float:
#     """
#     使用改进的DCF方法计算内在价值，考虑增长率和风险因素。

#     Args:
#         free_cash_flow: 自由现金流
#         growth_rate: 预期增长率
#         discount_rate: 基础折现率
#         terminal_growth_rate: 永续增长率
#         num_years: 预测年数

#     Returns:
#         float: 计算得到的内在价值
#     """
#     try:
#         if not isinstance(free_cash_flow, (int, float)) or free_cash_flow <= 0:
#             return 0

#         # 调整增长率，确保合理性
#         growth_rate = min(max(growth_rate, 0), 0.25)  # 限制在0-25%之间

#         # 调整永续增长率，不能超过经济平均增长
#         terminal_growth_rate = min(growth_rate * 0.4, 0.03)  # 取增长率的40%或3%的较小值

#         # 计算预测期现金流现值
#         present_values = []
#         for year in range(1, num_years + 1):
#             future_cf = free_cash_flow * (1 + growth_rate) ** year
#             present_value = future_cf / (1 + discount_rate) ** year
#             present_values.append(present_value)

#         # 计算永续价值
#         terminal_year_cf = free_cash_flow * (1 + growth_rate) ** num_years
#         terminal_value = terminal_year_cf * \
#             (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
#         terminal_present_value = terminal_value / \
#             (1 + discount_rate) ** num_years

#         # 总价值
#         total_value = sum(present_values) + terminal_present_value
        
#         print(f'free_cash_flow: {free_cash_flow}')
#         return max(total_value, 0)  # 确保不返回负值

#     except Exception as e:
#         print(f"DCF计算错误: {e}")
#         return 0


# def calculate_working_capital_change(
#     current_working_capital: float,
#     previous_working_capital: float,
# ) -> float:
#     """
#     Calculate the absolute change in working capital between two periods.
#     A positive change means more capital is tied up in working capital (cash outflow).
#     A negative change means less capital is tied up (cash inflow).

#     Args:
#         current_working_capital: Current period's working capital
#         previous_working_capital: Previous period's working capital

#     Returns:
#         float: Change in working capital (current - previous)
#     """
#     return current_working_capital - previous_working_capital