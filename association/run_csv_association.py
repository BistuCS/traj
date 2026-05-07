from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

from traj_association_module import associate_trajectories


PointKey = Tuple[float, float, float, float]


def format_seconds_to_ts(seconds: float) -> str:
    """将秒数格式化为 MM:SS.s（与原始 CSV 常见格式一致）。"""
    total = float(seconds)
    mm = int(total // 60)
    ss = total - mm * 60
    return f"{mm:02d}:{ss:04.1f}"


def parse_ts_to_seconds(ts_value: Any) -> float:
    """支持数值、MM:SS.s、HH:MM:SS.s。"""
    if isinstance(ts_value, (int, float)):
        return float(ts_value)

    s = str(ts_value).strip()
    if not s:
        raise ValueError("ts 不能为空")

    if ":" not in s:
        return float(s)

    parts = s.split(":")
    if len(parts) == 2:
        mm = float(parts[0])
        ss = float(parts[1])
        return mm * 60.0 + ss
    if len(parts) == 3:
        hh = float(parts[0])
        mm = float(parts[1])
        ss = float(parts[2])
        return hh * 3600.0 + mm * 60.0 + ss

    raise ValueError(f"无法解析 ts: {ts_value}")


def load_input_csv_first5(
    input_csv: str | Path,
    pos_dim: int,
) -> tuple[Dict[float, List[Dict[str, Any]]], Dict[PointKey, List[int]], Dict[float, str]]:
    """
    从 CSV 读取前 5 列作为接口字段：ts,id,x,y,z。
    当 pos_dim=2 时仅使用 x,y；pos_dim=3 使用 x,y,z。
    """
    if pos_dim not in (2, 3):
        raise ValueError("pos_dim 仅支持 2 或 3")

    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    data: DefaultDict[float, List[Dict[str, Any]]] = defaultdict(list)
    src_id_index: DefaultDict[PointKey, List[int]] = defaultdict(list)
    ts_text_index: Dict[float, str] = {}

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV 为空")

        for line_no, row in enumerate(reader, start=2):
            if len(row) < 4:
                raise ValueError(f"第 {line_no} 行列数不足，至少需要 4 列(ts,id,x,y)")

            try:
                ts_raw = str(row[0]).strip()
                ts = parse_ts_to_seconds(ts_raw)
                point: Dict[str, Any] = {
                    "ts": ts,
                    "id": int(float(row[1])),
                    "x": float(row[2]),
                    "y": float(row[3]),
                }
                if pos_dim == 3:
                    z = float(row[4]) if len(row) >= 5 and str(row[4]).strip() != "" else 0.0
                    point["z"] = z

                key: PointKey = (
                    float(point["ts"]),
                    float(point["x"]),
                    float(point["y"]),
                    float(point.get("z", 0.0)),
                )
                src_id_index[key].append(int(point["id"]))

                if ts not in ts_text_index:
                    ts_text_index[ts] = ts_raw if ts_raw else format_seconds_to_ts(ts)

                data[ts].append(point)
            except Exception as e:
                raise ValueError(f"第 {line_no} 行解析失败: {e}") from e

    return dict(data), dict(src_id_index), ts_text_index


def save_output_csv(
    output: Dict[int, List[Dict[str, float | int]]],
    output_csv: str | Path,
    src_id_index: Dict[PointKey, List[int]],
    ts_text_index: Dict[float, str],
) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float | int]] = []
    for _, infos in output.items():
        rows.extend(infos)

    resolved_rows: List[Dict[str, float | int]] = []
    for r in rows:
        key: PointKey = (
            float(r["ts"]),
            float(r["x"]),
            float(r["y"]),
            float(r["z"]),
        )
        src_ids = src_id_index.get(key, [])
        src_id = src_ids.pop(0) if src_ids else int(r["id"])
        resolved_rows.append(
            {
                "ts": float(r["ts"]),
                "id": int(src_id),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "z": float(r["z"]),
            }
        )

    grouped_rows: DefaultDict[int, List[Dict[str, float | int]]] = defaultdict(list)
    for r in resolved_rows:
        grouped_rows[int(r["id"])].append(r)

    for gid in grouped_rows:
        grouped_rows[gid].sort(key=lambda r: float(r["ts"]))

    ordered_ids = sorted(
        grouped_rows.keys(),
        key=lambda gid: (float(grouped_rows[gid][0]["ts"]), int(gid)),
    )

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "id", "x", "y", "z"])
        writer.writeheader()
        for gid in ordered_ids:
            for r in grouped_rows[gid]:
                writer.writerow(
                    {
                        "ts": ts_text_index.get(float(r["ts"]), format_seconds_to_ts(float(r["ts"]))),
                        "id": int(r["id"]),
                        "x": float(r["x"]),
                        "y": float(r["y"]),
                        "z": float(r["z"]),
                    }
                )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV -> 轨迹关联 -> CSV")
    parser.add_argument("--input", default="data/input/1.csv", help="输入 CSV 路径")
    parser.add_argument("--output", default="data/output/1_associated.csv", help="输出 CSV 路径")
    parser.add_argument("--ds-id", type=int, default=1, help="数据源 ID")
    parser.add_argument("--pos-dim", type=int, choices=[2, 3], default=3, help="位置维度")
    args = parser.parse_args()

    data, src_id_index, ts_text_index = load_input_csv_first5(args.input, pos_dim=args.pos_dim)
    result = associate_trajectories(data=data, ds_id=args.ds_id, pos_dim=args.pos_dim)
    out_path = save_output_csv(
        result,
        args.output,
        src_id_index=src_id_index,
        ts_text_index=ts_text_index,
    )
    print(f"完成: {out_path}")


if __name__ == "__main__":
    main()
