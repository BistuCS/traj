import sys
from pathlib import Path

# 添加 association 目录到 Python 路径
association_path = Path(__file__).parent / "association"
sys.path.insert(0, str(association_path))



from data_receiver import AdaptiveDataReceiver
from data_preprocessor import DataPreprocessor
from predictor import TrajectoryPredictor
from traj_association_module import associate_trajectories

if __name__ == "__main__":
    # ============== 1. 读取 + 预处理数据 ==============
    receiver = AdaptiveDataReceiver()
    # 替换为你的CSV文件路径
    raw_data = receiver.load_csv("data/1.csv")

    preprocessor = DataPreprocessor(receiver.config)
    clean_data = preprocessor.process(raw_data)

    if not clean_data:
        print("❌ 无有效数据")
        exit()

    # ============== 2. 目标分组 ==============
    predictor = TrajectoryPredictor(receiver.config)
    grouped_data = predictor.group_by_id(clean_data)
    print(f"\n🎯 目标总数：{len(grouped_data)}")

    # ============== 🔥 三种模式自由选择 ==============
    # 可选模式：KALMAN（卡尔曼）、LSTM、HYBRID（混合，推荐）
    PREDICT_MODE = "KALMAN"

    # 根据选择执行对应预测
    if PREDICT_MODE == "KALMAN":
        print("\n🚀 启动 卡尔曼滤波预测...")
        predict_result = predictor.predict_kalman(grouped_data)
    elif PREDICT_MODE == "LSTM":
        print("\n🤖 启动 PyTorch LSTM 预测...")
        predict_result = predictor.predict_lstm(grouped_data)
    elif PREDICT_MODE == "HYBRID":
        print("\n⚡ 启动 卡尔曼+LSTM 混合预测...")
        predict_result = predictor.predict_hybrid(grouped_data)
    else:
        print("❌ 预测模式错误，请选择 KALMAN/LSTM/HYBRID")
        exit()

    # ============== 3. 统一导出预测结果CSV ==============
    if predict_result:
        predictor.export_to_csv(predict_result)
    else:
        print("❌ 无预测结果（轨迹数据不足）")
        exit()

    # ============== 4. 统一转换为开发对接接口格式 ==============
    developer_interface_data = predictor.convert_to_develop_interface(predict_result)
    print("\n🤝 成功生成【开发对接数据】，格式完全符合接口要求！")

    #  🔥 调用关联（核心步骤）
    # ==========================================
    associated_result = associate_trajectories(
        data=developer_interface_data,
        ds_id=1,
        pos_dim=3
    )

    # 5. 输出结果
    print("\n关联完成！输出如下：")
    for traj_id, points in associated_result.items():
        print(f"轨迹 {traj_id}: {len(points)} 点")
