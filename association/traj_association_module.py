from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def _geo_distance_m(alng: float, alat: float, blng: float, blat: float) -> float:
    """计算经纬度点间距离（米）。优先 geopy，不可用时回退 haversine。"""
    try:
        from geopy.distance import distance as geopy_distance

        return float(geopy_distance((alat, alng), (blat, blng)).m)
    except Exception:
        # Haversine fallback
        import math

        r = 6371000.0
        lat1 = math.radians(alat)
        lat2 = math.radians(blat)
        dlat = lat2 - lat1
        dlon = math.radians(blng - alng)
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return float(2 * r * math.asin(math.sqrt(a)))


@dataclass
class AssociationConfig:
    """轨迹点关联配置。"""

    threshold_m: float = 1900.0
    timeout_steps: int = 9
    cycle_by_ds: Dict[int, int] = field(
        default_factory=lambda: {1: 6, 2: 6, 3: 8, 4: 8, 5: 6, 6: 6}
    )
    predict_history_len: int = 20
    process_noise_sigma: float = 0.01
    measure_noise_sigma: float = 0.1


class PointAssociator:
    """基于匈牙利算法的点级关联器。"""

    @staticmethod
    def _dist_general(point1: Sequence[float], point2: Sequence[float], pos_dim: int) -> float:
        dist_squared = 0.0
        for dim in range(pos_dim):
            dist_squared += (point1[dim] - point2[dim]) ** 2
        return float(np.sqrt(dist_squared) * 100000)

    def associate(
        self,
        new_points: List[List[float]],
        pred_traj_points: List[List[float]],
        threshold_d: float,
        pos_dim: int,
    ) -> List[Tuple[int, int]]:
        len_new = len(new_points)
        len_pred = len(pred_traj_points)

        if len_pred == 0 or len_new == 0:
            return []

        cost = np.zeros((len_pred, len_new), dtype=float)

        for i in range(len_pred):
            mn = float("inf")
            for j in range(len_new):
                if pos_dim == 2:
                    dis = _geo_distance_m(
                        pred_traj_points[i][0],
                        pred_traj_points[i][1],
                        new_points[j][0],
                        new_points[j][1],
                    )
                else:
                    dis = self._dist_general(pred_traj_points[i][:pos_dim], new_points[j][:pos_dim], pos_dim)

                cost[i, j] = dis
                mn = min(mn, dis)

            if mn > threshold_d:
                cost[i, :] = mn

        row_ind, col_ind = linear_sum_assignment(cost)

        match_pair: List[Tuple[int, int]] = []
        for idx_pre, idx_new in zip(row_ind, col_ind):
            if cost[idx_pre, idx_new] <= threshold_d:
                match_pair.append((int(idx_pre), int(idx_new)))
        return match_pair


class TrajectoryPredictor:
    """轨迹点预测器（卡尔曼滤波），与原项目逻辑一致。"""

    def __init__(self, config: AssociationConfig):
        self.config = config

    def predict_next_point(self, traj: List[List[float]], pos_dim: int) -> List[float]:
        num = self.config.predict_history_len

        if len(traj) >= num:
            dim_x = 2 * pos_dim
            dim_z = pos_dim
            kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

            dt = 1.0
            f = np.eye(dim_x)
            for i in range(pos_dim):
                f[i, i + pos_dim] = dt
            kf.F = f

            h = np.zeros((dim_z, dim_x))
            for i in range(dim_z):
                h[i, i] = 1
            kf.H = h

            kf.R = np.diag([self.config.measure_noise_sigma] * dim_z)

            q = self.config.process_noise_sigma**2
            q_mat = np.zeros((dim_x, dim_x))
            for i in range(pos_dim):
                q_mat[i, i] = q * dt**4 / 4
                q_mat[i, i + pos_dim] = q * dt**3 / 2
                q_mat[i + pos_dim, i] = q * dt**3 / 2
                q_mat[i + pos_dim, i + pos_dim] = q * dt**2
            kf.Q = q_mat

            sigma = 0.05
            r = np.diag([sigma**2] * dim_z)
            p_diag = list(r.diagonal()) + [1.0] * pos_dim
            kf.P = np.diag(p_diag)

            history = np.array(traj, dtype=float)[-num:, :pos_dim]
            kf.x = np.zeros(dim_x)
            kf.x[:pos_dim] = history[0]

            for i in range(history.shape[0]):
                z = history[i].reshape((dim_z, 1))
                kf.predict()
                kf.update(z)

            last_point = kf.x[:pos_dim].tolist()
            last_t = traj[-1][pos_dim] - traj[-2][pos_dim] + traj[-1][pos_dim]
        else:
            last_point = list(traj[-1][:pos_dim])
            last_t = traj[-1][pos_dim]

        return last_point + [last_t]


class TrajectoryAssociator:
    """
    可复用的单数据源轨迹点关联模块。

        输入格式:
        - {time_step: [{"ts": ts, "id": id, "x": x, "y": y, "z": z?}, ...]}
            其中 z 缺失时默认 0。

        输出格式:
        - {traj_id: [{"ts": ts, "id": traj_id, "x": x, "y": y, "z": z}, ...]}
            输出统一包含 5 个字段，z 恒存在。
    """

    def __init__(self, config: AssociationConfig | None = None):
        self.config = config or AssociationConfig()
        self.point_associator = PointAssociator()
        self.predictor = TrajectoryPredictor(self.config)

    def associate(
        self,
        data: Dict[float, List[Dict[str, Any]]],
        ds_id: int,
        pos_dim: int,
    ) -> Dict[int, List[Dict[str, float | int]]]:
        ob1_ps: List[List[List[float]]] = []
        normalized_data = self._normalize_input_data(data=data, pos_dim=pos_dim)
        timestep = sorted(list(normalized_data.keys()))

        cycle = self.config.cycle_by_ds.get(ds_id, 1)
        last_trajs_points: List[List[float]] = []
        pred_trajs_points: List[List[float]] = []
        dict_idx_points: Dict[Tuple[float, ...], int] = {}

        for cur_t in timestep:
            new_points: List[List[float]] = []
            for point in normalized_data[cur_t]:
                new_points.append(list(point[:pos_dim]) + [cur_t])

            match_pairs = self.point_associator.associate(
                new_points=new_points,
                pred_traj_points=pred_trajs_points,
                threshold_d=self.config.threshold_m,
                pos_dim=pos_dim,
            )

            matched: List[int] = []
            for i, j in match_pairs:
                key = tuple(pred_trajs_points[i][:pos_dim])
                if key in dict_idx_points:
                    ob1_ps[dict_idx_points[key]].append(new_points[j])
                else:
                    ob1_ps.append([new_points[j]])
                    dict_idx_points[key] = len(ob1_ps) - 1
                matched.append(j)

            unmatched = set(range(len(new_points))) - set(matched)
            for j in unmatched:
                ob1_ps.append([new_points[j]])

            cur_trajs_idx: List[int] = []
            for idx, trajs in enumerate(ob1_ps):
                last_point = trajs[-1][: pos_dim + 1]
                t = last_point[-1]
                if t >= cur_t - self.config.timeout_steps * cycle:
                    cur_trajs_idx.append(idx)

            last_trajs_points = []
            pred_trajs_points = []
            dict_idx_points = {}

            for idx in cur_trajs_idx:
                traj = ob1_ps[idx]
                last_trajs_points.append(traj[-1][:pos_dim])
                predict_result = self.predictor.predict_next_point(traj, pos_dim)
                pred_trajs_points.append(predict_result[: pos_dim + 1])
                dict_idx_points[tuple(predict_result[:pos_dim])] = idx

        trajs_ds: DefaultDict[int, List[Dict[str, float | int]]] = defaultdict(list)
        for traj_id, trajs in enumerate(ob1_ps):
            for point in trajs:
                t = float(point[-1])
                x = float(point[0])
                y = float(point[1])
                z = float(point[2]) if pos_dim == 3 else 0.0
                trajs_ds[traj_id].append(
                    {
                        "ts": t,
                        "id": int(traj_id),
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                )

        output_dict = dict(trajs_ds)
        self._validate_output_data(output_dict)

        return output_dict

    @staticmethod
    def _normalize_input_data(
        data: Dict[float, List[Dict[str, Any]]], pos_dim: int
    ) -> Dict[float, List[List[float]]]:
        """
        归一化输入数据为: {time_step: [[x, y, ...], ...]}
        - information 格式字段: ts/id/x/y/z（z 可缺省，默认 0）
        """
        normalized: Dict[float, List[List[float]]] = defaultdict(list)

        if pos_dim not in (2, 3):
            raise ValueError("information 输入仅支持 pos_dim 为 2 或 3")

        if not isinstance(data, dict):
            raise TypeError("data 必须是字典: Dict[time_step, List[information]]")

        for t_key, points in data.items():
            if not isinstance(points, list):
                raise TypeError("data 的每个 time_step 对应值必须是 List[information]")

            for point in points:
                if not isinstance(point, dict):
                    raise TypeError("输入点必须为 information 字典，包含 ts/id/x/y/z 字段")

                required_fields = {"ts", "id", "x", "y"}
                optional_fields = {"z"}
                keys = set(point.keys())

                missing = required_fields - keys
                if missing:
                    raise ValueError(f"information 缺少必填字段: {sorted(missing)}")

                extra = keys - required_fields - optional_fields
                if extra:
                    raise ValueError(f"information 包含未定义字段: {sorted(extra)}")

                ts = float(point["ts"])
                if float(t_key) != ts:
                    raise ValueError("information.ts 必须与外层 time_step 键一致")

                _ = int(point["id"])
                x = float(point["x"])
                y = float(point["y"])
                z = float(point.get("z", 0.0))

                vec = [x, y] if pos_dim == 2 else [x, y, z]

                normalized[ts].append(vec)

        return dict(normalized)

    @staticmethod
    def _validate_output_data(output_data: Dict[int, List[Dict[str, Any]]]) -> None:
        """严格校验输出格式为 information 结构体。"""
        if not isinstance(output_data, dict):
            raise TypeError("输出必须是字典: Dict[traj_id, List[information]]")

        for traj_id, infos in output_data.items():
            if not isinstance(traj_id, int):
                raise TypeError("输出的 traj_id 必须是 int")
            if not isinstance(infos, list):
                raise TypeError("输出每个 traj_id 对应值必须是 List[information]")

            for info in infos:
                if not isinstance(info, dict):
                    raise TypeError("输出点必须是 information 字典")

                expected_keys = {"ts", "id", "x", "y", "z"}
                keys = set(info.keys())
                if keys != expected_keys:
                    raise ValueError("输出 information 字段必须且只能是 ts/id/x/y/z")

                _ = float(info["ts"])
                out_id = int(info["id"])
                if out_id != traj_id:
                    raise ValueError("输出 information.id 必须等于外层 traj_id")
                _ = float(info["x"])
                _ = float(info["y"])
                _ = float(info["z"])


def associate_trajectories(
    data: Dict[float, List[Dict[str, Any]]],
    ds_id: int,
    pos_dim: int,
    config: AssociationConfig | None = None,
) -> Dict[int, List[Dict[str, float | int]]]:
    """函数式调用入口，便于其他项目直接调用。"""
    return TrajectoryAssociator(config=config).associate(data=data, ds_id=ds_id, pos_dim=pos_dim)
