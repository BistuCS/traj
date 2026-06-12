from __future__ import annotations

import argparse
import csv
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

from traj_association_module import associate_trajectories


PointKey = Tuple[float, float, float, float]


def format_seconds_to_ts(seconds: float) -> str:
    dt = datetime.datetime.fromtimestamp(float(seconds))
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def parse_ts_to_seconds(ts_value: Any) -> float:
    if isinstance(ts_value, (int, float)):
        return float(ts_value)

    s = str(ts_value).strip()
    if not s:
        raise ValueError("ts cannot be empty")

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass

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

    raise ValueError(f"cannot parse ts: {ts_value}")


def _first_present(row: Dict[str, Any], field_names: List[str]) -> str:
    for name in field_names:
        if name in row and str(row[name]).strip() != "":
            return str(row[name]).strip()
    raise KeyError(f"missing any of fields: {field_names}")


def load_input_csv_by_schema(
    input_csv: str | Path,
    pos_dim: int,
) -> tuple[Dict[float, List[Dict[str, Any]]], Dict[PointKey, List[int]], Dict[float, str]]:
    if pos_dim not in (2, 3):
        raise ValueError("pos_dim only supports 2 or 3")

    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"input file does not exist: {input_path}")

    data: DefaultDict[float, List[Dict[str, Any]]] = defaultdict(list)
    src_id_index: DefaultDict[PointKey, List[int]] = defaultdict(list)
    ts_text_index: Dict[float, str] = {}
    source_id_map: Dict[str, int] = {}
    next_source_id = 1

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV is empty")

        for line_no, row in enumerate(reader, start=2):
            try:
                ts_raw = _first_present(row, ["ts"])
                # The original 9y flow floors timestamps to integer seconds before processing frames.
                ts = float(int(parse_ts_to_seconds(ts_raw)))

                # Match the original sender: uuid is the primary track id, tarbatnum is the fallback.
                raw_id = _first_present(row, ["uuid", "tarbatnum", "id"])
                try:
                    point_id = int(float(raw_id))
                except ValueError:
                    if raw_id not in source_id_map:
                        source_id_map[raw_id] = next_source_id
                        next_source_id += 1
                    point_id = source_id_map[raw_id]

                point: Dict[str, Any] = {
                    "ts": ts,
                    "id": point_id,
                    "x": float(_first_present(row, ["x", "tarlon", "lon", "lng"])),
                    "y": float(_first_present(row, ["y", "tarlat", "lat"])),
                }
                if pos_dim == 3:
                    z_raw = ""
                    for z_name in ["z", "tarheight", "height", "alt"]:
                        if z_name in row and str(row[z_name]).strip() != "":
                            z_raw = str(row[z_name]).strip()
                            break
                    point["z"] = float(z_raw) if z_raw else 0.0

                key: PointKey = (
                    float(point["ts"]),
                    float(point["x"]),
                    float(point["y"]),
                    float(point.get("z", 0.0)),
                )
                src_id_index[key].append(int(point["id"]))
                if ts not in ts_text_index:
                    ts_text_index[ts] = ts_raw

                data[ts].append(point)
            except Exception as e:
                raise ValueError(f"failed to parse line {line_no}: {e}") from e

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

    rows.sort(key=lambda r: (int(r["tra_id"]), float(r["ts"])))

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "id", "tra_id", "x", "y", "z", "point_type", "is_predicted"])
        writer.writeheader()
        for r in rows:
            key: PointKey = (
                float(r["ts"]),
                float(r["x"]),
                float(r["y"]),
                float(r["z"]),
            )
            is_predicted = int(r.get("is_predicted", 0))
            src_ids = src_id_index.get(key, []) if not is_predicted else []
            src_id = src_ids.pop(0) if src_ids else -1

            writer.writerow(
                {
                    "ts": ts_text_index.get(float(r["ts"]), format_seconds_to_ts(float(r["ts"]))),
                    "id": int(src_id),
                    "tra_id": int(r["tra_id"]),
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                    "z": float(r["z"]),
                    "point_type": "predicted" if is_predicted else "real",
                    "is_predicted": is_predicted,
                }
            )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV -> trajectory association -> CSV")
    parser.add_argument("--input", default="data/input/data_611.csv", help="input CSV path")
    parser.add_argument("--output", default="data/output/data_611_associated.csv", help="output CSV path")
    parser.add_argument("--ds-id", type=int, default=1, help="data source ID")
    parser.add_argument("--pos-dim", type=int, choices=[2, 3], default=3, help="position dimension")
    args = parser.parse_args()

    data, src_id_index, ts_text_index = load_input_csv_by_schema(args.input, pos_dim=args.pos_dim)
    result = associate_trajectories(data=data, ds_id=args.ds_id, pos_dim=args.pos_dim)
    out_path = save_output_csv(
        result,
        args.output,
        src_id_index=src_id_index,
        ts_text_index=ts_text_index,
    )
    print(f"completed: {out_path}")


if __name__ == "__main__":
    main()
