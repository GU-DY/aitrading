from datetime import datetime, timedelta
import argparse
from src.agents.valuation import valuation_agent
from src.agents.state import AgentState
from src.agents.sentiment import sentiment_agent
from src.agents.risk_manager import risk_management_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.market_data import market_data_agent
from src.agents.fundamentals import fundamentals_agent
from src.agents.researcher_bull import researcher_bull_agent
from src.agents.researcher_bear import researcher_bear_agent
from src.agents.debate_room import debate_room_agent
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
import akshare as ak
import pandas as pd

from src.utils.output_logger import OutputLogger
import sys

# Initialize output logging
# This will create a timestamped log file in the logs directory
sys.stdout = OutputLogger()


##### Run the Hedge Fund #####
def run_hedge_fund(ticker: str, start_date: str, end_date: str, portfolio: dict, show_reasoning: bool = False, num_of_news: int = 5):
    final_state = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "num_of_news": num_of_news,
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            }
        },
    )
    return final_state["messages"][-1].content


# Define the new workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("market_data_agent", market_data_agent)
workflow.add_node("technical_analyst_agent", technical_analyst_agent)
workflow.add_node("fundamentals_agent", fundamentals_agent)
workflow.add_node("sentiment_agent", sentiment_agent)
workflow.add_node("valuation_agent", valuation_agent)
workflow.add_node("researcher_bull_agent", researcher_bull_agent)
workflow.add_node("researcher_bear_agent", researcher_bear_agent)
workflow.add_node("debate_room_agent", debate_room_agent)
workflow.add_node("risk_management_agent", risk_management_agent)
workflow.add_node("portfolio_management_agent", portfolio_management_agent)

# Define the workflow
workflow.set_entry_point("market_data_agent")

# Market Data to Analysts
workflow.add_edge("market_data_agent", "technical_analyst_agent")
workflow.add_edge("market_data_agent", "fundamentals_agent")
workflow.add_edge("market_data_agent", "sentiment_agent")
workflow.add_edge("market_data_agent", "valuation_agent")

# Analysts to Researchers
workflow.add_edge("technical_analyst_agent", "researcher_bull_agent")
workflow.add_edge("fundamentals_agent", "researcher_bull_agent")
workflow.add_edge("sentiment_agent", "researcher_bull_agent")
workflow.add_edge("valuation_agent", "researcher_bull_agent")

workflow.add_edge("technical_analyst_agent", "researcher_bear_agent")
workflow.add_edge("fundamentals_agent", "researcher_bear_agent")
workflow.add_edge("sentiment_agent", "researcher_bear_agent")
workflow.add_edge("valuation_agent", "researcher_bear_agent")

# Researchers to Debate Room
workflow.add_edge("researcher_bull_agent", "debate_room_agent")
workflow.add_edge("researcher_bear_agent", "debate_room_agent")

# Debate Room to Risk Management
workflow.add_edge("debate_room_agent", "risk_management_agent")

# Risk Management to Portfolio Management
workflow.add_edge("risk_management_agent", "portfolio_management_agent")
workflow.add_edge("portfolio_management_agent", END)

app = workflow.compile()

# Add this at the bottom of the file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the hedge fund trading system')
    parser.add_argument('--ticker', type=str,
                        help='Stock ticker symbol',default='002843')
    parser.add_argument('--start-date', type=str,default='2025-04-01',
                        help='Start date (YYYY-MM-DD). Defaults to 1 year before end date')
    parser.add_argument('--end-date', type=str,default='2025-04-21',
                        help='End date (YYYY-MM-DD). Defaults to yesterday')
    parser.add_argument('--show-reasoning', action='store_true',
                        help='Show reasoning from each agent')
    parser.add_argument('--num-of-news', type=int, default=5,
                        help='Number of news articles to analyze for sentiment (default: 5)')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                        help='Initial cash amount (default: 100,000)')
    parser.add_argument('--initial-position', type=int, default=0,
                        help='Initial stock position (default: 0)')

    args = parser.parse_args()

    # Set end date to yesterday if not specified
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = yesterday if not args.end_date else min(
        datetime.strptime(args.end_date, '%Y-%m-%d'), yesterday)

    # Set start date to one year before end date if not specified
    if not args.start_date:
        start_date = end_date - timedelta(days=365)  # 默认获取一年的数据
    else:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')

    # Validate dates
    if start_date > end_date:
        raise ValueError("Start date cannot be after end date")

    # Validate num_of_news
    if args.num_of_news < 1:
        raise ValueError("Number of news articles must be at least 1")
    if args.num_of_news > 100:
        raise ValueError("Number of news articles cannot exceed 100")

    # Configure portfolio
    portfolio = {
        "cash": args.initial_capital,
        "stock": args.initial_position
    }

    result = run_hedge_fund(
        ticker=args.ticker,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        num_of_news=args.num_of_news
    )
    print("\nFinal Result:")
    print(result)


def get_historical_data(symbol: str) -> pd.DataFrame:
    """Get historical market data for a given stock symbol.
    If we can't get the full year of data, use whatever is available."""
    # Calculate date range
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = yesterday  # Use yesterday as end date
    target_start_date = yesterday - \
        timedelta(days=365)  # Target: 1 year of data

    print(f"\n正在获取 {symbol} 的历史行情数据...")
    print(f"目标开始日期：{target_start_date.strftime('%Y-%m-%d')}")
    print(f"结束日期：{end_date.strftime('%Y-%m-%d')}")

    try:
        # Get historical data
        df = ak.stock_zh_a_hist(symbol=symbol,
                                period="daily",
                                start_date=target_start_date.strftime(
                                    "%Y%m%d"),
                                end_date=end_date.strftime("%Y%m%d"),
                                adjust="qfq")

        actual_days = len(df)
        target_days = 365  # Target: 1 year of data

        if actual_days < target_days:
            print(f"提示：实际获取到的数据天数({actual_days}天)少于目标天数({target_days}天)")
            print(f"将使用可获取到的所有数据进行分析")

        print(f"成功获取历史行情数据，共 {actual_days} 条记录\n")
        return df

    except Exception as e:
        print(f"获取历史数据时发生错误: {str(e)}")
        print("将尝试获取最近可用的数据...")

        # Try to get whatever data is available
        try:
            df = ak.stock_zh_a_hist(symbol=symbol,
                                    period="daily",
                                    adjust="qfq")
            print(f"成功获取历史行情数据，共 {len(df)} 条记录\n")
            return df
        except Exception as e:
            print(f"获取历史数据失败: {str(e)}")
            return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Run the hedge fund trading system')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol (e.g. 601398)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--show-reasoning', action='store_true', help='Show reasoning from each agent')
    parser.add_argument('--num-of-news', type=int, default=5, help='Number of news articles to analyze (1~100)')
    parser.add_argument('--initial-capital', type=float, default=100000.0, help='Initial capital in cash')
    parser.add_argument('--initial-position', type=int, default=0, help='Initial stock position')

    args = parser.parse_args()

    # 处理日期
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else yesterday
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else (end_date - timedelta(days=365))

    # 日期校验
    if start_date > end_date:
        raise ValueError("Start date cannot be after end date.")
    if args.num_of_news < 1 or args.num_of_news > 100:
        raise ValueError("Number of news articles must be between 1 and 100.")

    # 初始化投资组合
    portfolio = {
        "cash": args.initial_capital,
        "stock": args.initial_position
    }

    result = run_hedge_fund(
        ticker=args.ticker,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        num_of_news=args.num_of_news
    )

    print("\nFinal Decision Output:\n" + "-"*30)
    print(result)




