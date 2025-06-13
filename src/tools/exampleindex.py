from datetime import datetime
from indexdata import get_index_constituents, predict_index_return, predict_index_return_agent

import re



# 示例


if __name__ == "__main__":
    date = "2025-06-09"
    index_code = "000016.SH"
    df = get_index_constituents(index_code, date)
    print(df.head(10))
    pred = predict_index_return(index_code, date)
    print(f"Predicted index return: {pred:.2%}")

    pred_agent = predict_index_return_agent(index_code, date, start_date="2025-06-06")
    print(f"Agent-based prediction: {pred_agent:.2%}")