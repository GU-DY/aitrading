from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import simplejson as json                         # 更健壮的 JSON 库
from backtestrunner import run_backtest # ① 重要：导入上面的包装函数
from hedge_fund_service import run_hedge_fund

def create_app():
    app = Flask(__name__,
                static_folder="frontend",        # JS / CSS / 图片
                static_url_path="")              # 让 /xxx 直接映射到 frontend/xxx
    CORS(app)

    @app.route("/", methods=["GET"])
    def serve_index():
        return send_from_directory(app.static_folder, "index.html")


    @app.route("/api/backtest", methods=["POST"])
    def backtest_endpoint():
        data = request.get_json(force=True)       # 前端发 JSON
        try:
            df = run_backtest(
                ticker=data["ticker"],
                start_date=data["start_date"],
                end_date=data["end_date"],
                initial_capital=float(data.get("initial_capital", 100000)),
                num_of_news=int(data.get("num_of_news", 5))
            )

            # ③ orient='records'  -> [{col:val,...}, ...]
            return jsonify(df)

        except Exception as e:
            # 统一错误格式：{"error": "..."}
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/predict", methods=["POST"])
    def predict_endpoint():
        data = request.get_json(force=True)
        try:
            resp = run_hedge_fund(
                ticker=data["ticker"],
                start_date=data["start_date"],
                show_reasoning=True   # 总是返回推理
            )
            return jsonify(resp)      # -> {"result": "...", "reasoning": "..."}
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return app

# ④ 方便 gunicorn/flask run 调用
app = create_app()

if __name__ == "__main__":
    # 开发环境直接 python app.py
    app.run(host="0.0.0.0", port=8000, debug=True)