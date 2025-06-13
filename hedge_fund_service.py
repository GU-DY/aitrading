from datetime import datetime, timedelta
from typing import Dict, List
import io, logging, contextlib, os, json, re

# -------- Agents / Workflow imports --------
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

# ---------------- DeepSeek client ----------------
try:
    from openai import OpenAI  # type: ignore
    deepseek_key = 'sk-c8741e8aa02e448f9da78161e00c4f44'
    deep_client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com") if deepseek_key else None
except Exception:
    deep_client = None

# ---------------- Build workflow ----------------
workflow = StateGraph(AgentState)
for n, node in [
    ("market_data_agent", market_data_agent),
    ("technical_analyst_agent", technical_analyst_agent),
    ("fundamentals_agent", fundamentals_agent),
    ("sentiment_agent", sentiment_agent),
    ("valuation_agent", valuation_agent),
    ("researcher_bull_agent", researcher_bull_agent),
    ("researcher_bear_agent", researcher_bear_agent),
    ("debate_room_agent", debate_room_agent),
    ("risk_management_agent", risk_management_agent),
    ("portfolio_management_agent", portfolio_management_agent),
]:
    workflow.add_node(n, node)
workflow.set_entry_point("market_data_agent")
for analyst in ["technical_analyst_agent","fundamentals_agent","sentiment_agent","valuation_agent"]:
    workflow.add_edge("market_data_agent", analyst)
    workflow.add_edge(analyst, "researcher_bull_agent")
    workflow.add_edge(analyst, "researcher_bear_agent")
workflow.add_edge("researcher_bull_agent", "debate_room_agent")
workflow.add_edge("researcher_bear_agent", "debate_room_agent")
workflow.add_edge("debate_room_agent", "risk_management_agent")
workflow.add_edge("risk_management_agent", "portfolio_management_agent")
workflow.add_edge("portfolio_management_agent", END)
app = workflow.compile()

# ---------------- Helper functions ----------------

def _ask_llm(prompt: str) -> str:
    if deep_client is None:
        return "[未配置 DEEPSEEK_API_KEY]"
    res = deep_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": "You are a helpful assistant"},
                  {"role": "user", "content": prompt}],
        stream=False,
    )
    return res.choices[0].message.content.strip()

def _translate(text: str) -> str:
    if not text:
        return ""
    return _ask_llm("请将以下所有内容翻译成中文\n。包括特定的名词，不需要额外的任何解释和说明。" + text)

def _summarise(reasoning: str) -> str:
    if deep_client is None:
        return "[未配置 DEEPSEEK_API_KEY，无法生成摘要]"
    prompt = (
        "你是金融分析助手。阅读下列英文推理信息，提取每个智能体给出的关键信号、数据及其依据，最后给出总结性评价。以中文输出。\n\n"
        + reasoning
    )
    return _ask_llm(prompt)

# ---------------- Utility ----------------

def _strip_code_block(text: str) -> str:
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:])
        if text.endswith("```"):
            text = "\n".join(text.splitlines()[:-1])
    return text


def _collect_reasonings(msgs: List[HumanMessage]) -> str:
    blocks = []
    for m in msgs:
        t = _strip_code_block(m.content.strip())
        if re.search(r'"reasoning"', t) or re.search(r'"agent_signals"', t):
            blocks.append(t)
    return "\n\n".join(blocks)

# ---------------- Core API ----------------

def run_hedge_fund(ticker: str, start_date: str, show_reasoning: bool = True, num_of_news: int = 5) -> Dict[str, str]:
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    portfolio = {"cash": 100_000, "stock": 0}

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    try:
        with contextlib.redirect_stdout(buf):
            fs = app.invoke({
                "messages": [HumanMessage(content="Make a trading decision based on the provided data.")],
                "data": {
                    "ticker": ticker,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "num_of_news": num_of_news,
                },
                "metadata": {"show_reasoning": show_reasoning},
            })
    finally:
        root.removeHandler(handler)

    # -------- decision --------
    final_raw = _strip_code_block(fs["messages"][-1].content.strip())
    try:
        js = json.loads(final_raw)
        decision_en = f"操作: {js.get('action')}, 数量: {js.get('quantity')}, 置信度: {js.get('confidence')},预计涨跌幅: {js.get('predicted_return_pct')}%"
        reasoning_en = f"智能体信号：{js.get('agent_signals')}"
    except Exception:
        decision_en = final_raw[:200]

    analysis_detail_en = _collect_reasonings(fs["messages"])

    decision_cn  = _translate(decision_en)
    reasoning_cn = _translate(reasoning_en)
    analysis_cn  = _summarise(analysis_detail_en)

    return {
        "result": decision_cn,
        "reasoning": reasoning_cn,
        "analysis": analysis_cn,
    }



if __name__ == "__main__":
    import argparse, textwrap
    p = argparse.ArgumentParser(description="CLI test for run_hedge_fund")
    p.add_argument("ticker")
    p.add_argument("start_date")
    args = p.parse_args()
    out = run_hedge_fund(args.ticker, args.start_date)
    print("\n=== 结论 (CN) ===\n", out["result"])
    print("\n=== 摘要 ===\n", textwrap.shorten(out["analysis"], 400000))