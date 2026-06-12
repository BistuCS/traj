from filterpy.kalman import KalmanFilter
import numpy as np


class Filter:
    def __init__(self, pos):
        # 初始化卡尔曼滤波器，状态维度为 6（3 个位置 + 3 个速度），观测维度为 3（位置）
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # 初始状态，前三个为位置，后三个为速度初始设为 0
        self.kf.x = np.array([pos[0], pos[1], pos[2], 0., 0., 0.])

        # 状态转移矩 nk  x阵，初始化为单位矩阵，后续根据 dt 调整8
        self.kf.F = np.eye(6)

        # 观测矩阵，只观测位置
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0]])

        # 过程噪声协方差矩阵，初始设为较小值，让滤波器快速收敛
        self.kf.Q = np.eye(6) * 0.01

        # 观测噪声协方差矩阵，初始设为较小值
        self.kf.R = np.eye(3) * 0.1

        # 初始估计误差协方差矩阵，给速度部分较大的不确定性，让滤波器快速收敛
        self.kf.P = np.diag([1., 1., 1., 1000., 1000., 1000.])

    def predict(self, dt):
        # 根据 dt 更新状态转移矩阵
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt

        # 进行预测
        self.kf.predict()

    def now_pos(self):
        return self.kf.x[0:3]

    def update(self, pos):
        # 根据观测位置更新状态
        self.kf.update(np.array(pos))