import Gaussian_progress_dynamic as gp
import numpy as np

# 总共三架无人机，采用多智能体分布式优化的思想
# 每架无人机分别测量自己所处位置的信号强度值
# 如果只有一架无人机，无人机获取位置和信号强度以及时间信息，该信息存储在对应的train_X和train_Y里
# 如果有三架无人机采用多智能体分布式优化的思想，理论上讲高斯过程的输入都是一样的，因为无人机获取到信号强度信息后会广播自己获取到的信息
# 实际情况下可能会出现无人机不在通信范围内的情况，因此有些无人机无法获取到全部信息，因此每架无人机的信息分别存储


train_X_1 = []
train_Y_1 = []
train_X_2 = []
train_Y_2 = []
train_X_3 = []
train_Y_3 = []
position_1 = []
position_2 = []
position_3 = []
t = 0.0




# 返回值为ALC方差最大的候选点的索引
def ALC(self, train_X, candidates_list, Kff_inv, time_track):
    gpr = gp.GPR(optimize=True)
    # 参考点列表
    reference_points_list = self.get_reference_points_list(time_track)
    # 候选点列表
    # candidates_angle_list = self.calculate_candidates_points_angle()
    # candidates_list = self.calculate_candidates_points(current_position, candidates_angle_list)
    # 直接用gp.kernel计算ALC
    # 对于每一个候选点，计算候选点的ALC公式的值，计算时需要计算每个参考点对应的值，将所有参考点的结果加起来，作为每个候选点的值
    train_data_num = len(train_X)
    ALC_list = [0.0 for i in range(len(candidates_list))]
    # ALC_list = [0.0 for i in range(len(candidates_list))]*len(candidates_list)
    for i in range(0, len(candidates_list)):
        candidate = candidates_list[i]
        m = []  # the N-vector of covariances between the present training data points and the query candidate
        for j in range(0, train_data_num):
            train_data = train_X[j]
            # print('test-------')
            # print(train_data)
            # print(candidate)
            m.append(gpr.kernel(np.asarray(train_data), np.asarray(candidate))[0])
        m = np.atleast_2d(m)
        C_N_inv = Kff_inv
        C_xx = gpr.kernel(np.asarray(candidate), np.asarray(candidate))  # x代表候选点
        # 对于每一个参考点
        for j in range(0, len(reference_points_list)):
            k_N = []  # the vector of covariances between the training data and a reference data point
            reference_point = reference_points_list[j]
            k_xc_xr = gpr.kernel(candidate, reference_point)
            k_xc_X = gpr.kernel(candidate, train_X)
            k_X_xr = gpr.kernel(train_X, reference_point)
            cov_x_x_star = k_xc_xr - np.dot(np.dot(k_xc_X, Kff_inv), k_X_xr)
            ALC_list[i] = ALC_list[i] + cov_x_x_star

    max_ALC = ALC_list[0]
    max_ALC_index = 0
    for i in range(0, len(ALC_list)):
        if ALC_list[i] > max_ALC:
            max_ALC = ALC_list[i]
            max_ALC_index = i
    return max_ALC_index