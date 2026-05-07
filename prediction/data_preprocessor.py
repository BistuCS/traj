from typing import List
from data_receiver import RadarData

class DataPreprocessor:
    """
    数据预处理核心模块
    功能：
    1. 必选字段完全重复数据去重
    2. XYZ坐标异常值过滤
    """
    def __init__(self, config: dict):
        # 加载预处理配置
        self.x_min, self.x_max = config["preprocess"]["x_range"]
        self.y_min, self.y_max = config["preprocess"]["y_range"]
        self.z_min, self.z_max = config["preprocess"]["z_range"]

    def _is_xyz_valid(self, data: RadarData) -> bool:
        """校验XYZ坐标是否符合逻辑"""
        try:
            x, y, z = data.x, data.y, data.z
            return (self.x_min <= x <= self.x_max and
                    self.y_min <= y <= self.y_max and
                    self.z_min <= z <= self.z_max)
        except:
            return False

    def _deduplicate_data(self, data_list: List[RadarData]) -> List[RadarData]:
        """
        必选字段完全重复去重
        唯一标识：ts + id + x + y + z
        """
        unique_keys = set()
        unique_data = []

        for data in data_list:
            # 生成唯一键（5个必选字段）
            key = (data.ts, data.id, round(data.x, 6), round(data.y, 6), round(data.z, 6))
            if key not in unique_keys:
                unique_keys.add(key)
                unique_data.append(data)

        return unique_data

    def process(self, data_list: List[RadarData]) -> List[RadarData]:
        """
        统一预处理入口
        :param data_list: 原始数据列表
        :return: 预处理后干净数据
        """
        total = len(data_list)
        if total == 0:
            return []

        # 1. 去重
        data_after_dedup = self._deduplicate_data(data_list)
        dedup_count = total - len(data_after_dedup)

        # 2. 过滤XYZ异常值
        clean_data = [d for d in data_after_dedup if self._is_xyz_valid(d)]
        invalid_count = len(data_after_dedup) - len(clean_data)

        # 打印预处理统计
        print(f"\n🧹 数据预处理完成：")
        print(f"   原始数据：{total} 条")
        print(f"   删除重复数据：{dedup_count} 条")
        print(f"   删除XYZ异常数据：{invalid_count} 条")
        print(f"   最终有效数据：{len(clean_data)} 条")

        return clean_data