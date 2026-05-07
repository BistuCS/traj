import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from typing import List, Dict, Any, Union
from data_receiver import RadarData
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import os
import csv
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


# ====================== PyTorch LSTM 模型定义 ======================
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=3, predict_steps=15):
        super(TrajectoryLSTM, self).__init__()
        self.predict_steps = predict_steps
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * predict_steps)  # 输出多个时间步

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        out = out.view(-1, self.predict_steps, 3)  # 重塑为 (batch, predict_steps, 3)
        return out


# ====================== 预测核心类（PyTorch + 开发接口适配）======================
class TrajectoryPredictor:
    def __init__(self, config: dict):
        self.config = config
        self.predict_steps = config["predict"]["predict_steps"]
        # 卡尔曼参数
        self.q = config["predict"]["kalman"]["process_noise"]
        self.r = config["predict"]["kalman"]["measure_noise"]
        # LSTM参数
        self.look_back = config["predict"]["lstm"]["look_back"]
        self.epochs = config["predict"]["lstm"]["epochs"]
        self.batch_size = config["predict"]["lstm"]["batch_size"]
        self.hidden_size = config["predict"]["lstm"]["units"]
        self.lr = config["predict"]["lstm"]["lr"]
        self.model_dir = config["predict"]["lstm"]["model_save_dir"]
        os.makedirs(self.model_dir, exist_ok=True)
        # 混合预测权重
        self.k_w = config["predict"]["hybrid"]["kalman_weight"]
        self.l_w = config["predict"]["hybrid"]["lstm_weight"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- 基础工具 ----------------------
    def group_by_id(self, data_list: List[RadarData]) -> Dict[Any, List[RadarData]]:
        grouped = defaultdict(list)
        for data in data_list:
            grouped[data.id].append(data)
        for tid in grouped:
            grouped[tid].sort(key=lambda x: x.ts)
        return grouped

    # ---------------------- 1. 卡尔曼滤波预测 ----------------------
    def _kalman_predict_multi(self, x_list, y_list, z_list, steps):
        """多步卡尔曼预测"""
        n = len(x_list)
        x_est, y_est, z_est = np.zeros(n), np.zeros(n), np.zeros(n)
        x_est[0], y_est[0], z_est[0] = x_list[0], y_list[0], z_list[0]
        p = 1.0
        for k in range(1, n):
            p_predict = p + self.q
            k_gain = p_predict / (p_predict + self.r)
            x_est[k] = x_est[k - 1] + k_gain * (x_list[k] - x_est[k - 1])
            y_est[k] = y_est[k - 1] + k_gain * (y_list[k] - y_est[k - 1])
            z_est[k] = z_est[k - 1] + k_gain * (z_list[k] - z_est[k - 1])
            p = (1 - k_gain) * p_predict
        
        # 多步递推预测
        x_preds, y_preds, z_preds = [], [], []
        last_x, last_y, last_z = x_est[-1], y_est[-1], z_est[-1]
        prev_x, prev_y, prev_z = x_est[-2], y_est[-2], z_est[-2]
        
        for i in range(steps):
            # 基于速度外推
            vx = last_x - prev_x
            vy = last_y - prev_y
            vz = last_z - prev_z
            
            next_x = last_x + vx
            next_y = last_y + vy
            next_z = last_z + vz
            
            x_preds.append(next_x)
            y_preds.append(next_y)
            z_preds.append(next_z)
            
            # 更新状态
            prev_x, prev_y, prev_z = last_x, last_y, last_z
            last_x, last_y, last_z = next_x, next_y, next_z
        
        return x_preds, y_preds, z_preds

    def predict_kalman(self, grouped_data):
        results = {}
        for tid, data_list in grouped_data.items():
            if len(data_list) < 2:
                continue
            xs = np.array([d.x for d in data_list])
            ys = np.array([d.y for d in data_list])
            zs = np.array([d.z for d in data_list])
            
            # 多步预测
            x_preds, y_preds, z_preds = self._kalman_predict_multi(xs, ys, zs, self.predict_steps)
            
            last_ts = data_list[-1].ts
            try:
                dt = (data_list[-1].ts - data_list[-2].ts).total_seconds()
                if pd.isna(dt) or dt <= 0:
                    dt = 1.0
            except:
                dt = 1.0
            
            # 生成多个预测点
            predictions = []
            for i in range(self.predict_steps):
                if pd.isna(last_ts):
                    pred_ts = datetime(2000, 1, 1, 0, 0, 0)
                    pred_ts_str = "00:00.0"
                else:
                    # 安全地计算时间增量，避免任何溢出
                    # 直接提取分秒部分进行计算，不依赖pandas Timestamp运算
                    try:
                        # 尝试正常计算
                        total_seconds_offset = dt * (i + 1)
                        pred_ts = last_ts + timedelta(seconds=total_seconds_offset)
                    except (OverflowError, OSError, pd.errors.OutOfBoundsDatetime, pd.errors.OutOfBoundsTimedelta):
                        # 如果溢出，只使用最后时间的分秒部分手动计算
                        base_total_seconds = last_ts.minute * 60 + last_ts.second + last_ts.microsecond / 1000000.0
                        new_total_seconds = base_total_seconds + dt * (i + 1)
                        minutes = int(new_total_seconds // 60) % 60
                        seconds = new_total_seconds % 60
                        pred_ts = datetime(2000, 1, 1, 0, minutes, int(seconds), int((seconds % 1) * 1000000))
                    
                    # 格式化时间为 MM:SS.m 格式（与原CSV保持一致）
                    pred_ts_str = pred_ts.strftime("%M:%S.%f")[:-5]  # 保留1位毫秒
                
                predictions.append({
                    "predict_ts": pred_ts,
                    "predict_ts_str": pred_ts_str,
                    "x": round(x_preds[i], 6),
                    "y": round(y_preds[i], 6),
                    "z": round(z_preds[i], 6),
                    "step": i + 1
                })
            
            results[tid] = {
                "predictions": predictions,
                "history_count": len(data_list)
            }
        return results

    # ---------------------- 2. PyTorch LSTM 预测 ----------------------
    def _create_sequences(self, data):
        X, y = [], []
        for i in range(self.look_back, len(data)):
            X.append(data[i - self.look_back:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def _save_model(self, model, scaler, target_id):
        path = os.path.join(self.model_dir, f"model_{target_id}.pth")
        scaler_path = os.path.join(self.model_dir, f"scaler_{target_id}.npy")
        torch.save({
            'model_state_dict': model.state_dict(),
            'predict_steps': self.predict_steps
        }, path)
        np.save(scaler_path, [scaler.min_, scaler.scale_])

    def _load_model(self, target_id):
        path = os.path.join(self.model_dir, f"model_{target_id}.pth")
        scaler_path = os.path.join(self.model_dir, f"scaler_{target_id}.npy")
        if not os.path.exists(path):
            return None, None
        checkpoint = torch.load(path, map_location=self.device)
        model = TrajectoryLSTM(hidden_size=self.hidden_size, predict_steps=self.predict_steps).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        scaler = MinMaxScaler()
        scaler.min_, scaler.scale_ = np.load(scaler_path)
        return model, scaler

    def predict_lstm(self, grouped_data, train_new=True):
        results = {}
        for tid, data_list in grouped_data.items():
            if len(data_list) < self.look_back + 1:
                continue

            model, scaler = self._load_model(tid)
            coords = np.array([[d.x, d.y, d.z] for d in data_list])

            if model is None and train_new:
                scaler = MinMaxScaler()
                coords_scaled = scaler.fit_transform(coords)
                X, y = self._create_sequences(coords_scaled)
                X = torch.tensor(X, dtype=torch.float32).to(self.device)
                y = torch.tensor(y, dtype=torch.float32).to(self.device)

                model = TrajectoryLSTM(hidden_size=self.hidden_size, predict_steps=self.predict_steps).to(self.device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=self.lr)

                model.train()
                for _ in range(self.epochs):
                    optimizer.zero_grad()
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

                self._save_model(model, scaler, tid)
                coords_scaled = scaler.transform(coords)
            else:
                coords_scaled = scaler.transform(coords)

            model.eval()
            with torch.no_grad():
                last_seq = coords_scaled[-self.look_back:]
                last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_scaled = model(last_seq).cpu().numpy()  # shape: (1, predict_steps, 3)
            
            # 反标准化所有预测点
            pred_scaled_reshaped = pred_scaled.reshape(-1, 3)
            pred = scaler.inverse_transform(pred_scaled_reshaped).reshape(self.predict_steps, 3)

            last_ts = data_list[-1].ts
            try:
                dt = (data_list[-1].ts - data_list[-2].ts).total_seconds()
                if pd.isna(dt) or dt <= 0:
                    dt = 1.0
            except:
                dt = 1.0

            # 生成多个预测点
            predictions = []
            for i in range(self.predict_steps):
                if pd.isna(last_ts):
                    pred_ts = datetime(2000, 1, 1, 0, 0, 0)
                    pred_ts_str = "00:00.0"
                else:
                    # 安全地计算时间增量，避免任何溢出
                    # 直接提取分秒部分进行计算，不依赖pandas Timestamp运算
                    try:
                        # 尝试正常计算
                        total_seconds_offset = dt * (i + 1)
                        pred_ts = last_ts + timedelta(seconds=total_seconds_offset)
                    except (OverflowError, OSError, pd.errors.OutOfBoundsDatetime, pd.errors.OutOfBoundsTimedelta):
                        # 如果溢出，只使用最后时间的分秒部分手动计算
                        base_total_seconds = last_ts.minute * 60 + last_ts.second + last_ts.microsecond / 1000000.0
                        new_total_seconds = base_total_seconds + dt * (i + 1)
                        minutes = int(new_total_seconds // 60) % 60
                        seconds = new_total_seconds % 60
                        pred_ts = datetime(2000, 1, 1, 0, minutes, int(seconds), int((seconds % 1) * 1000000))
                    
                    # 格式化时间为 MM:SS.m 格式（与原CSV保持一致）
                    pred_ts_str = pred_ts.strftime("%M:%S.%f")[:-5]  # 保留1位毫秒
                
                predictions.append({
                    "predict_ts": pred_ts,
                    "predict_ts_str": pred_ts_str,
                    "x": round(pred[i][0], 6),
                    "y": round(pred[i][1], 6),
                    "z": round(pred[i][2], 6),
                    "step": i + 1
                })
            
            results[tid] = {
                "predictions": predictions,
                "history_count": len(data_list)
            }
        return results

    # ---------------------- 3. 混合预测 ----------------------
    def predict_hybrid(self, grouped_data):
        kalman_res = self.predict_kalman(grouped_data)
        lstm_res = self.predict_lstm(grouped_data)
        hybrid_res = {}

        for tid in lstm_res.keys():
            if tid not in kalman_res:
                continue
            
            k_predictions = kalman_res[tid]["predictions"]
            l_predictions = lstm_res[tid]["predictions"]
            
            # 对每个预测步进行加权融合
            hybrid_predictions = []
            for i in range(self.predict_steps):
                k = k_predictions[i]
                l = l_predictions[i]
                
                hybrid_predictions.append({
                    "predict_ts": l["predict_ts"],
                    "predict_ts_str": l["predict_ts_str"],
                    "x": round(self.k_w * k["x"] + self.l_w * l["x"], 6),
                    "y": round(self.k_w * k["y"] + self.l_w * l["y"], 6),
                    "z": round(self.k_w * k["z"] + self.l_w * l["z"], 6),
                    "step": i + 1
                })
            
            hybrid_res[tid] = {
                "predictions": hybrid_predictions,
                "history_count": len(grouped_data[tid])
            }
        return hybrid_res

    # ---------------------- 导出CSV ----------------------
    def export_to_csv(self, predict_result, csv_path=None):
        csv_path = csv_path or self.config["predict"]["output"]["result_csv_path"]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["ts", "id", "x", "y", "z"])

            for tid, res in predict_result.items():
                predictions = res.get("predictions", [])
                for pred in predictions:
                    # 使用字符串时间格式（与原CSV保持一致）
                    time_str = pred.get("predict_ts_str", "00:00.000")
                    
                    writer.writerow([
                        time_str, tid, pred["x"], pred["y"], pred["z"]
                    ])
        print(f"✅ 预测结果已导出：{csv_path}，共 {len(predict_result)} 个目标，每个目标 {self.predict_steps} 个预测点")

    # ====================== ✅ 开发接口转换函数（核心）======================

    def convert_to_develop_interface(self, predict_result: Dict) -> Dict[str, List[Dict]]:
        """
        将预测结果转换为开发接口格式
        ts字段使用与原CSV相同的字符串格式（如 "MM:SS.mmm"）
        """
        interface_data = defaultdict(list)

        for target_id, info in predict_result.items():
            predictions = info.get("predictions", [])
            
            for pred in predictions:
                # 使用字符串时间格式（与原CSV保持一致）
                time_str = pred.get("predict_ts_str", "00:00.000")

                # 构造标准信息字典（不包含step字段）
                information = {
                    "ts": time_str,
                    "id": target_id,
                    "x": pred["x"],
                    "y": pred["y"],
                    "z": pred["z"]
                }

                interface_data[time_str].append(information)

        return dict(interface_data)