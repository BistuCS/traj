from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter

_PREDICTION_DIR = Path(__file__).resolve().parents[1] / "prediction"
if _PREDICTION_DIR.exists() and str(_PREDICTION_DIR) not in sys.path:
    sys.path.insert(0, str(_PREDICTION_DIR))
try:
    from filter import Filter as _PredictionFilter
except Exception:
    _PredictionFilter = None


def _geo_distance_m(alng: float, alat: float, blng: float, blat: float) -> float:
    try:
        from geopy.distance import distance as geopy_distance

        return float(geopy_distance((alat, alng), (blat, blng)).m)
    except Exception:
        r = 6371000.0
        lat1 = math.radians(alat)
        lat2 = math.radians(blat)
        dlat = lat2 - lat1
        dlon = math.radians(blng - alng)
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return float(2 * r * math.asin(math.sqrt(a)))


def _point_distance(point1: Sequence[float], point2: Sequence[float], pos_dim: int) -> float:
    if pos_dim == 2:
        return _geo_distance_m(point1[0], point1[1], point2[0], point2[1])

    horizontal = _geo_distance_m(point1[0], point1[1], point2[0], point2[1])
    vertical = abs(float(point1[2]) - float(point2[2]))
    return float(np.sqrt(horizontal**2 + vertical**2))


@dataclass
class AssociationConfig:
    # Keep these defaults aligned with 9y-6-10-v0.1/config.py.
    threshold_m: float = 120.0
    traj_stop_dt: float = 30.0
    fuse_turn_dt: float = 30.0
    fuse_min_duration: float = 60.0
    predict_minnum: int = 5
    timeout_steps: int = 9
    cycle_by_ds: Dict[int, int] = field(
        default_factory=lambda: {1: 6, 2: 6, 3: 8, 4: 8, 5: 6, 6: 6}
    )
    predict_history_len: int = 20
    process_noise_sigma: float = 0.01
    measure_noise_sigma: float = 0.1


class _MotionFilter:
    def __init__(self, pos: Sequence[float], pos_dim: int, config: AssociationConfig):
        self.pos_dim = pos_dim
        self.external_filter = None
        if _PredictionFilter is not None:
            self.external_filter = _PredictionFilter(self._as_3d(pos))
            return

        dim_x = 2 * pos_dim
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=pos_dim)
        self.kf.x = np.array(list(pos[:pos_dim]) + [0.0] * pos_dim, dtype=float)
        self.kf.F = np.eye(dim_x)

        h = np.zeros((pos_dim, dim_x))
        for i in range(pos_dim):
            h[i, i] = 1.0
        self.kf.H = h

        self.kf.Q = np.eye(dim_x) * float(config.process_noise_sigma)
        self.kf.R = np.eye(pos_dim) * float(config.measure_noise_sigma)
        self.kf.P = np.diag([1.0] * pos_dim + [1000.0] * pos_dim)

    def predict(self, dt: float) -> None:
        if self.external_filter is not None:
            self.external_filter.predict(dt)
            return
        for i in range(self.pos_dim):
            self.kf.F[i, i + self.pos_dim] = dt
        self.kf.predict()

    def update(self, pos: Sequence[float]) -> None:
        if self.external_filter is not None:
            self.external_filter.update(self._as_3d(pos))
            return
        self.kf.update(np.array(pos[: self.pos_dim], dtype=float))

    def now_pos(self) -> List[float]:
        if self.external_filter is not None:
            return list(map(float, self.external_filter.now_pos()))[: self.pos_dim]
        return self.kf.x[: self.pos_dim].astype(float).tolist()

    def _as_3d(self, pos: Sequence[float]) -> List[float]:
        vec = list(map(float, pos[: self.pos_dim]))
        while len(vec) < 3:
            vec.append(0.0)
        return vec[:3]


class _SourceTrajectory:
    def __init__(
        self,
        traj_id: int,
        t: float,
        pos: Sequence[float],
        pos_dim: int,
        config: AssociationConfig,
        stop_dt: float,
        predict_minnum: int,
    ):
        self.id = traj_id
        self.pos_dim = pos_dim
        self.config = config
        self.stop_dt = stop_dt
        self.predict_minnum = predict_minnum
        self.filter = _MotionFilter(pos, pos_dim, config)
        self.positions: List[List[float]] = [list(map(float, pos[:pos_dim]))]
        self.times: List[float] = [float(t)]
        self.is_real_points: List[bool] = [True]
        self.isValid = True
        self.isFused = False
        self.isFusing = False
        self.isStop = False
        self.last_real_time = float(t)
        self.real_points_num = 1
        self.point_confidences: List[float] = [0.95]

    def move(self, t: float) -> None:
        if self.isStop:
            return
        if t - self.last_real_time > self.stop_dt:
            self.isStop = True
            return

        last_time = self.times[-1]
        dt = float(t - last_time)
        if dt < 0:
            return

        self.filter.predict(dt)
        predicted_pos = self.filter.now_pos()
        if self.real_points_num <= self.predict_minnum:
            predicted_pos = list(self.positions[-1])

        self.positions.append(predicted_pos)
        self.times.append(float(t))
        self.is_real_points.append(False)
        self.point_confidences.append(self._confidence_for_virtual_point(t))

    def update(self, pos: Sequence[float]) -> None:
        vec = list(map(float, pos[: self.pos_dim]))
        if self.is_real_points[-1]:
            self.positions[-1] = vec
            self.filter.update(vec)
            self.last_real_time = self.times[-1]
            self.point_confidences[-1] = 0.95
            return

        self.is_real_points[-1] = True
        self.real_points_num += 1
        self.last_real_time = self.times[-1]
        self.filter.update(vec)
        self.positions[-1] = self.filter.now_pos()
        self.point_confidences[-1] = 0.95

    def _confidence_for_virtual_point(self, t: float) -> float:
        time_diff = max(0.0, float(t - self.last_real_time))
        t_limit = max(1.0, self.stop_dt)
        if time_diff <= 0.2 * t_limit:
            a = (0.5 - 0.95) / math.log((0.1 * t_limit + 1.0) / 1.0)
            confidence = a * math.log(time_diff + 1.0) + 0.95
        else:
            min_confidence = 0.001
            d = (min_confidence - 0.5) / math.log((t_limit + 1.0) / (0.1 * t_limit + 1.0))
            f = 0.5 - d * math.log(0.1 * t_limit + 1.0)
            confidence = d * math.log(time_diff + 1.0) + f
        return float(max(0.001, min(0.95, confidence)))

    def output_points(self) -> List[Tuple[List[float], float, bool]]:
        return [
            (pos, ts, is_real)
            for pos, ts, is_real in zip(self.positions, self.times, self.is_real_points)
        ]


class _FusedTrajectory:
    def __init__(self, traj_id: int, source_traj: _SourceTrajectory):
        self.id = traj_id
        self.pos_dim = source_traj.pos_dim
        self.oritrajs: List[_SourceTrajectory] = [source_traj]
        self.positions = [list(pos) for pos in source_traj.positions]
        self.times = list(source_traj.times)
        self.is_real_points = list(source_traj.is_real_points)
        self.confidences_sum = list(source_traj.point_confidences)

    def update(self, t: float) -> None:
        updated = [traj for traj in self.oritrajs if traj.times[-1] == t]
        if not updated:
            return

        total_position = np.zeros(self.pos_dim, dtype=float)
        total_confidence = 0.0
        has_real = False
        for traj in updated:
            confidence = float(traj.point_confidences[-1])
            total_position += np.array(traj.positions[-1], dtype=float) * confidence
            total_confidence += confidence
            has_real = has_real or bool(traj.is_real_points[-1])

        if total_confidence <= 0:
            return

        self.positions.append((total_position / total_confidence).astype(float).tolist())
        self.times.append(float(t))
        self.is_real_points.append(has_real)
        self.confidences_sum.append(total_confidence)

    def match(self, source_traj: _SourceTrajectory) -> Tuple[bool, float]:
        start_time = max(self.times[0], source_traj.times[0])
        self_start = next((i for i, ts in enumerate(self.times) if ts >= start_time), len(self.times))
        src_start = next((i for i, ts in enumerate(source_traj.times) if ts >= start_time), len(source_traj.times))

        weighted_distance_sum = 0.0
        weight_sum = 0.0
        i = self_start
        j = src_start
        while i < len(self.times) and j < len(source_traj.times):
            dist = _point_distance(self.positions[i], source_traj.positions[j], self.pos_dim)
            weight = float(self.confidences_sum[i]) * float(source_traj.point_confidences[j])
            weighted_distance_sum += dist * weight
            weight_sum += weight
            i += 1
            j += 1

        if weight_sum <= 0:
            return False, float("inf")

        avg_weighted_distance = weighted_distance_sum / weight_sum
        return avg_weighted_distance < source_traj.config.threshold_m, avg_weighted_distance

    def fuse(self, source_traj: _SourceTrajectory) -> None:
        self.oritrajs.append(source_traj)
        new_times: List[float] = []
        new_positions: List[List[float]] = []
        new_is_real: List[bool] = []
        new_confidences: List[float] = []

        i = 0
        j = 0
        while i < len(self.times) and j < len(source_traj.times):
            if self.times[i] == source_traj.times[j]:
                self_conf = float(self.confidences_sum[i])
                src_conf = float(source_traj.point_confidences[j])
                total_conf = self_conf + src_conf
                merged = (
                    np.array(self.positions[i], dtype=float) * self_conf
                    + np.array(source_traj.positions[j], dtype=float) * src_conf
                ) / total_conf
                new_times.append(self.times[i])
                new_positions.append(merged.astype(float).tolist())
                new_is_real.append(bool(self.is_real_points[i] or source_traj.is_real_points[j]))
                new_confidences.append(total_conf)
                i += 1
                j += 1
            elif self.times[i] < source_traj.times[j]:
                new_times.append(self.times[i])
                new_positions.append(list(self.positions[i]))
                new_is_real.append(bool(self.is_real_points[i]))
                new_confidences.append(float(self.confidences_sum[i]))
                i += 1
            else:
                new_times.append(source_traj.times[j])
                new_positions.append(list(source_traj.positions[j]))
                new_is_real.append(bool(source_traj.is_real_points[j]))
                new_confidences.append(float(source_traj.point_confidences[j]))
                j += 1

        while i < len(self.times):
            new_times.append(self.times[i])
            new_positions.append(list(self.positions[i]))
            new_is_real.append(bool(self.is_real_points[i]))
            new_confidences.append(float(self.confidences_sum[i]))
            i += 1

        while j < len(source_traj.times):
            new_times.append(source_traj.times[j])
            new_positions.append(list(source_traj.positions[j]))
            new_is_real.append(bool(source_traj.is_real_points[j]))
            new_confidences.append(float(source_traj.point_confidences[j]))
            j += 1

        self.times = new_times
        self.positions = new_positions
        self.is_real_points = new_is_real
        self.confidences_sum = new_confidences

    def output_points(self) -> List[Tuple[List[float], float, bool]]:
        return [
            (pos, ts, is_real)
            for pos, ts, is_real in zip(self.positions, self.times, self.is_real_points)
        ]


class TrajectoryAssociator:
    def __init__(self, config: AssociationConfig | None = None):
        self.config = config or AssociationConfig()

    def associate(
        self,
        data: Dict[float, List[Dict[str, Any]]],
        ds_id: int,
        pos_dim: int,
    ) -> Dict[int, List[Dict[str, float | int]]]:
        normalized_data = self._normalize_input_data(data=data, pos_dim=pos_dim)
        if not normalized_data:
            return {}

        stop_dt = float(self.config.traj_stop_dt)
        fuse_turn_dt = float(self.config.fuse_turn_dt)
        fuse_min_duration = float(self.config.fuse_min_duration)
        predict_minnum = int(self.config.predict_minnum)

        source_trajs: Dict[int, _SourceTrajectory] = {}
        fused_trajs: Dict[int, _FusedTrajectory] = {}
        fusing_trajs: List[_SourceTrajectory] = []
        next_fuse_id = 0
        last_turn_t: float | None = None

        for cur_t in sorted(normalized_data.keys()):
            if last_turn_t is None:
                last_turn_t = cur_t

            for traj in source_trajs.values():
                if traj.isValid:
                    traj.move(cur_t)

            for point in normalized_data[cur_t]:
                src_id = int(point["id"])
                pos = [float(point["x"]), float(point["y"])]
                if pos_dim == 3:
                    pos.append(float(point.get("z", 0.0)))

                if src_id in source_trajs and source_trajs[src_id].isValid:
                    source_trajs[src_id].update(pos)
                else:
                    source_trajs[src_id] = _SourceTrajectory(
                        traj_id=src_id,
                        t=cur_t,
                        pos=pos,
                        pos_dim=pos_dim,
                        config=self.config,
                        stop_dt=stop_dt,
                        predict_minnum=predict_minnum,
                    )

            for fuse_traj in fused_trajs.values():
                fuse_traj.update(cur_t)

            if cur_t - last_turn_t >= fuse_turn_dt:
                fusing_trajs.clear()
                for traj in source_trajs.values():
                    if traj.isFusing:
                        if (traj.times[-1] - traj.times[0]) >= fuse_min_duration:
                            traj.isFused = True
                            traj.isFusing = False
                            fusing_trajs.append(traj)
                    elif traj.isValid and not traj.isFused:
                        if traj.last_real_time - traj.times[0] >= fuse_turn_dt:
                            if traj.real_points_num >= 6:
                                traj.isFusing = True
                            else:
                                traj.isValid = False

                for traj in fusing_trajs:
                    for fuse_traj in fused_trajs.values():
                        is_matched, _ = fuse_traj.match(traj)
                        if is_matched:
                            fuse_traj.fuse(traj)
                            break
                    else:
                        fused_trajs[next_fuse_id] = _FusedTrajectory(next_fuse_id, traj)
                        next_fuse_id += 1

                last_turn_t = cur_t

        output = self._build_output(source_trajs, fused_trajs, pos_dim)
        self._validate_output_data(output)
        return output

    @staticmethod
    def _build_output(
        source_trajs: Dict[int, _SourceTrajectory],
        fused_trajs: Dict[int, _FusedTrajectory],
        pos_dim: int,
    ) -> Dict[int, List[Dict[str, float | int]]]:
        trajs_ds: DefaultDict[int, List[Dict[str, float | int]]] = defaultdict(list)
        used_ori_ids = set()

        next_out_id = 0
        for _, fused in sorted(fused_trajs.items(), key=lambda item: item[0]):
            for ori in fused.oritrajs:
                used_ori_ids.add(ori.id)

            for pos, ts, is_real in fused.output_points():
                x = float(pos[0])
                y = float(pos[1])
                z = float(pos[2]) if pos_dim == 3 else 0.0
                trajs_ds[next_out_id].append(
                    {
                        "ts": float(ts),
                        "id": next_out_id,
                        "tra_id": next_out_id,
                        "x": x,
                        "y": y,
                        "z": z,
                        "is_predicted": int(not is_real),
                    }
                )
            next_out_id += 1

        for _, traj in sorted(source_trajs.items(), key=lambda item: item[0]):
            if traj.id in used_ori_ids:
                continue
            if not traj.isValid or traj.isStop:
                continue

            for pos, ts, is_real in traj.output_points():
                x = float(pos[0])
                y = float(pos[1])
                z = float(pos[2]) if pos_dim == 3 else 0.0
                trajs_ds[next_out_id].append(
                    {
                        "ts": float(ts),
                        "id": next_out_id,
                        "tra_id": next_out_id,
                        "x": x,
                        "y": y,
                        "z": z,
                        "is_predicted": int(not is_real),
                    }
                )
            next_out_id += 1

        return dict(trajs_ds)

    @staticmethod
    def _normalize_input_data(
        data: Dict[float, List[Dict[str, Any]]], pos_dim: int
    ) -> Dict[float, List[Dict[str, float | int]]]:
        normalized: Dict[float, List[Dict[str, float | int]]] = defaultdict(list)

        if pos_dim not in (2, 3):
            raise ValueError("information input only supports pos_dim 2 or 3")
        if not isinstance(data, dict):
            raise TypeError("data must be Dict[time_step, List[information]]")

        for t_key, points in data.items():
            if not isinstance(points, list):
                raise TypeError("each time_step value must be List[information]")

            ts_key = float(t_key)
            for point in points:
                if not isinstance(point, dict):
                    raise TypeError("each input point must be a dict")

                required_fields = {"ts", "id", "x", "y"}
                optional_fields = {"z"}
                keys = set(point.keys())

                missing = required_fields - keys
                if missing:
                    raise ValueError(f"information missing required fields: {sorted(missing)}")

                extra = keys - required_fields - optional_fields
                if extra:
                    raise ValueError(f"information contains undefined fields: {sorted(extra)}")

                ts = float(point["ts"])
                if ts != ts_key:
                    raise ValueError("information.ts must equal outer time_step key")

                item: Dict[str, float | int] = {
                    "ts": ts,
                    "id": int(point["id"]),
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                }
                if pos_dim == 3:
                    item["z"] = float(point.get("z", 0.0))

                normalized[ts].append(item)

        return dict(normalized)

    @staticmethod
    def _validate_output_data(output_data: Dict[int, List[Dict[str, Any]]]) -> None:
        if not isinstance(output_data, dict):
            raise TypeError("output must be Dict[traj_id, List[information]]")

        for traj_id, infos in output_data.items():
            if not isinstance(traj_id, int):
                raise TypeError("output traj_id must be int")
            if not isinstance(infos, list):
                raise TypeError("each output traj_id value must be List[information]")

            for info in infos:
                if not isinstance(info, dict):
                    raise TypeError("each output point must be a dict")

                expected_keys = {"ts", "id", "tra_id", "x", "y", "z", "is_predicted"}
                if set(info.keys()) != expected_keys:
                    raise ValueError("output information fields must be exactly ts/id/tra_id/x/y/z/is_predicted")

                _ = float(info["ts"])
                out_id = int(info["id"])
                out_tra_id = int(info["tra_id"])
                if out_id != traj_id:
                    raise ValueError("output information.id must equal outer traj_id")
                if out_tra_id != traj_id:
                    raise ValueError("output information.tra_id must equal outer traj_id")
                _ = float(info["x"])
                _ = float(info["y"])
                _ = float(info["z"])
                _ = int(info["is_predicted"])


def associate_trajectories(
    data: Dict[float, List[Dict[str, Any]]],
    ds_id: int,
    pos_dim: int,
    config: AssociationConfig | None = None,
) -> Dict[int, List[Dict[str, float | int]]]:
    return TrajectoryAssociator(config=config).associate(data=data, ds_id=ds_id, pos_dim=pos_dim)
