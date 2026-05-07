import yaml
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import os


# ====================== 标准数据结构体（自动适配所有辅助字段）======================
@dataclass
class RadarData:
    """
    固定必选字段 + 动态辅助字段
    辅助字段自动识别，无需任何配置
    """
    # 修复：兼容低版本Python的类型写法
    ts: Union[datetime, pd.Timestamp]
    id: Any
    x: float
    y: float
    z: float
    aux: Dict[str, Any] = field(default_factory=dict)  # 动态辅助字段容器


# ====================== 终极自适应数据接收器 ======================
class AdaptiveDataReceiver:
    """
    核心特性：
    1. 必选字段：ts/id/x/y/z 强制校验
    2. 辅助字段：全自动识别，无代码硬编码
    3. 时间解析：无固定日期，自动适配任意格式
    4. 任意甲方CSV，直接运行，零修改
    """

    def __init__(self, config_path: str = "./config.yaml"):
        self.config = self._load_config(config_path)
        self.required_fields = self.config["fixed_fields"]["required"]
        self.type_rules = self.config["type_rules"]
        self.csv_conf = self.config["csv"]
        self.error_conf = self.config["error"]

    def _load_config(self, path: str) -> Dict:
        """加载配置文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在：{path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _convert_value(self, field: str, value: Any) -> Any:
        """
        智能类型转换：
        - 必选字段按规则转换
        - 时间：自动解析任意格式，无固定日期
        - 辅助字段：自动保留原始类型
        """
        if pd.isna(value):
            return self.error_conf["default_value"]

        rule = self.type_rules.get(field)
        if rule == "datetime":
            # 自动解析任意时间格式
            # 如果是 MM:SS.m 格式，需要添加默认日期
            try:
                dt = pd.to_datetime(value, errors="coerce")
                if pd.isna(dt):
                    # 尝试解析 MM:SS.m 格式
                    import re
                    match = re.match(r'^(\d+):(\d+\.\d+)$', str(value))
                    if match:
                        minutes = match.group(1)
                        seconds = match.group(2)
                        # 添加默认日期 2000-01-01
                        dt = pd.to_datetime(f"2000-01-01 00:{minutes}:{seconds}")
                return dt
            except:
                return pd.NaT
        elif rule == "float":
            return float(value)
        elif rule == "int":
            return int(value)
        return value

    def _process_row(self, row: pd.Series, aux_columns: List[str]) -> Optional[RadarData]:
        """单行数据处理：校验 + 转换 + 封装结构体"""
        try:
            # 1. 校验必选字段非空
            for field in self.required_fields:
                if pd.isna(row[field]):
                    raise ValueError(f"必选字段 [{field}] 为空")

            # 2. 转换必选字段类型
            fixed_data = {
                field: self._convert_value(field, row[field])
                for field in self.required_fields
            }

            # 3. 自动装载动态辅助字段
            aux_data = {col: row[col] for col in aux_columns if not pd.isna(row[col])}

            # 4. 封装为标准结构体
            return RadarData(
                ts=fixed_data["ts"],
                id=fixed_data["id"],
                x=fixed_data["x"],
                y=fixed_data["y"],
                z=fixed_data["z"],
                aux=aux_data
            )

        except Exception as e:
            if self.error_conf["skip_invalid_row"]:
                print(f"⚠️  跳过无效行 {row.name + 1}：{str(e)}")
                return None
            raise

    def load_csv(self, file_path: str) -> List[RadarData]:
        """
        统一入口：加载甲方CSV文件
        :param file_path: CSV路径
        :return: 结构化数据列表
        """
        # 读取文件
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")

        df = pd.read_csv(
            file_path,
            encoding=self.csv_conf["default_encoding"],
            delimiter=self.csv_conf["delimiter"],
            header=self.csv_conf["header_row"]
        )

        # 自动识别动态辅助字段
        all_columns = df.columns.tolist()
        aux_columns = [col for col in all_columns if col not in self.required_fields]

        print(f"📊 自动识别完成：")
        print(f"   必选字段：{self.required_fields}")
        print(f"   动态辅助字段：{len(aux_columns)} 个 -> {aux_columns}")

        # 校验必选字段完整性
        missing_fields = [f for f in self.required_fields if f not in all_columns]
        if missing_fields:
            raise ValueError(f"CSV缺失必选字段：{missing_fields}")

        # 批量处理数据
        result = [self._process_row(row, aux_columns) for _, row in df.iterrows()]
        result = [r for r in result if r is not None]

        print(f"✅ 数据加载完成：总计 {len(df)} 行，有效 {len(result)} 行")
        return result


# 快速测试
if __name__ == "__main__":
    receiver = AdaptiveDataReceiver()
    # 替换成你的CSV文件路径！
    data = receiver.load_csv("1.csv")
    if data:
        print("\n🔍 第一条数据预览：")
        print(data[0])