import sys
from pathlib import Path
# 添加 association 目录到 Python 路径
association_path = Path(__file__).parent / "association"
sys.path.insert(0, str(association_path))
from flask import Flask, request, jsonify, send_file, render_template_string
import pandas as pd
import os
import uuid
from datetime import datetime
from data_receiver import AdaptiveDataReceiver
from data_preprocessor import DataPreprocessor
from predictor import TrajectoryPredictor
from traj_association_module import associate_trajectories

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)

@app.route("/")
def index():
    html = open("map.html", "r", encoding="utf-8").read()
    return render_template_string(html)

@app.route("/upload", methods=["POST"])
def run():
    file = request.files["file"]
    fid = str(uuid.uuid4())
    path = f"uploads/{fid}.csv"
    file.save(path)

    try:
        # 1. 读取
        receiver = AdaptiveDataReceiver()
        raw = receiver.load_csv(path)

        # 2. 清洗
        prep = DataPreprocessor(receiver.config)
        clean = prep.process(raw)

        # 3. 预测
        pred = TrajectoryPredictor(receiver.config)
        grouped = pred.group_by_id(clean)
        pred_res = pred.predict_hybrid(grouped)

        # 4. 关联
        dev_data = pred.convert_to_develop_interface(pred_res)
        assoc = associate_trajectories(data=dev_data, ds_id=1, pos_dim=3)

        # ======================
        # 输出地图格式数据
        # ======================
        # ====================== 输出地图格式数据（NaT 修复版）======================
        history_by_id = {}
        for d in clean:
            # 🔥 核心修复：历史数据时间 NaT 兜底
            ts_dt = d.ts
            if pd.isna(ts_dt):
                ts_dt = datetime(2000, 1, 1, 0, 0, 0)
            ts = ts_dt.timestamp()

            i = d.id
            if i not in history_by_id:
                history_by_id[i] = []
            history_by_id[i].append({
                "ts": ts,
                "lng": d.x,
                "lat": d.y,
                "z": d.z
            })

        predict_data = []
        for tid, v in pred_res.items():
            # 处理多步预测结果
            predictions = v.get("predictions", [])
            for pred in predictions:
                # 使用字符串时间格式（与原CSV保持一致）
                time_str = pred.get("predict_ts_str", "00:00.000")

                predict_data.append({
                    "id": tid,
                    "ts": time_str,
                    "lng": pred["x"],
                    "lat": pred["y"],
                    "z": pred["z"]
                })

        assoc_data = []
        for tra_id, pts in assoc.items():
            for p in pts:
                assoc_data.append({
                    "tra_id": tra_id,
                    "ts": p["ts"],
                    "lng": p["x"],
                    "lat": p["y"],
                    "z": p["z"]
                })

        return jsonify({
            "ok": True,
            "history": history_by_id,
            "predict": predict_data,
            "assoc": assoc_data
        })

    except Exception as e:
        import traceback
        traceback.print_exc()  # 控制台打印完整错误栈
        return jsonify({"ok": False, "err": str(e)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5201, debug=True)