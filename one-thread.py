import copy
import random
import queue
import math
import parameters as pm
import pandas as pd
import numpy as np
import Gaussian_progress_dynamic as gp
import active_data_selection as ads

source_v_x = pm.source_v_x  # 源在x轴方向的运动速度
source_v_y = pm.source_v_y  # 源在y轴方向的运动速度
random_distance = pm.random_distance
v_track = pm.v_track
point_max_rssi = []  # 信号强度最大的点的位置
grid_point_list = []  # 栅格化的所有点的list
rssi_distribution_list = []  # 所有点对应的信号强度值的list
rssi_noise = pm.rssi_noise
random_distance = pm.random_distance  # 刚开始随机方向移动每次移动的距离
v_track = pm.v_track  # 追踪方移动速度
random_times = pm.random_times  # 刚开始随机移动的次数
distance_1 = pm.distance_1  # 向计算出的强度最大的位置移动的距离
distance_2 = pm.distance_2
threshold_distance = pm.threshold_distance
times_threshold_value = pm.time_to_target  # 两个阶段划分的阈值
d_max = pm.communication_threshold_value
d_min = pm.collision_threshold_value
left_boundary = pm.left_boundary
right_boundary = pm.right_boundary
source_point = [left_boundary, left_boundary]
# 用两个队列分别存储每架无人机的数据队列和位置队列，通过线程通信模拟无人机间通信
rssi_data_list = []
loc_data_list = []
uav_num = 3
initial_position = [[0.0, 0.0], [-0.5, 0.5], [0.5, 0.5]]
candidates_number = pm.candidates_number
k_for_cal = pm.k_for_calculate
times_csv = 'multi_times_1.csv'
'''
rssi_data_1 = queue.Queue()
rssi_data_2 = queue.Queue()
rssi_data_3 = queue.Queue()
loc_data_1 = queue.Queue()
loc_data_2 = queue.Queue()
loc_data_3 = queue.Queue()
'''


# time_track = 0.0


def init_rssi():
    data = pd.read_csv("rssi_distribution.csv", header=None)
    x_list = data.iloc[:, 0]
    y_list = data.iloc[:, 1]
    z_list = data.iloc[:, 2]
    max_loc = list(z_list).index(max(z_list))
    # print(z_list[max_loc])
    # print(max_loc)
    point_max_rssi.append(x_list[max_loc])
    point_max_rssi.append(y_list[max_loc])
    # print(point_max_rssi)
    for i in range(len(x_list)):
        coor = [0.0, 0.0]
        coor[0] = float(x_list[i])
        coor[1] = float(y_list[i])
        r = float(z_list[i])
        grid_point_list.append(coor)
        rssi_distribution_list.append(r)


def get_rssi(track_point, source):
    x = track_point[0] + point_max_rssi[0] - source[0]
    y = track_point[1] + point_max_rssi[1] - source[1]
    if x >= 4.9:
        x = 4.9
    if y >= 4.9:
        y = 4.9
    if x < -5.0:
        x = -5.0
    if y < -5.0:
        y = -5.0
    coordinate_approximate = [round(x, 1), round(y, 1)]
    i = grid_point_list.index(coordinate_approximate)
    rssi = rssi_distribution_list[i] + np.random.normal(0, rssi_noise)
    return rssi


def fin_max(ls):
    max_rssi = -100
    index = -1
    for i in range(0, len(ls)):
        if ls[i] > max_rssi:
            max_rssi = ls[i]
            index = i
    return index, max_rssi


def get_max_point(time_track, coordinate, rssi_value):
    gpr = gp.GPR(optimize=True)
    test_d1 = np.arange(left_boundary, right_boundary, 0.1)
    test_d2 = np.arange(left_boundary, right_boundary, 0.1)
    test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
    # test_X_array = np.asarray(test_X)
    # test_X_t = np.reshape(test_X_array, )
    total_sample = coordinate.qsize()
    coordinate_list = list(coordinate.queue)
    rssi_value_list = list(rssi_value.queue)
    t_list = [time_track] * len(test_d1) * len(test_d2)
    test_X = [[d1, d2, d3] for d1, d2, d3 in zip(test_d1.ravel(), test_d2.ravel(), t_list)]

    for i in (0, total_sample):
        gpr.fit(coordinate_list, rssi_value_list)
        mu, cov = gpr.predict(test_X)
        Kff_inv = gpr.get_Kff_inv()
        max_index, max_mu = fin_max(mu)
        return test_X[max_index], max_mu, Kff_inv, mu, cov


# 计算方位角
def calculate_angle(x1, y1, x2, y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        if dy > 0:
            angle = 0.0
        elif dy < 0:
            angle = math.pi
    elif dy == 0:
        if dx > 0:
            angle = math.pi / 2.0
        elif dy < 0:
            angle = math.pi * 3 / 2.0
    elif dx > 0:
        if dy > 0:
            angle = math.atan(dx / dy)
        elif dy < 0:
            angle = math.pi / 2 + math.atan(-dy / dx)
    elif dx < 0:
        if dy > 0:
            angle = 3.0 * math.pi / 2 + math.atan(dy / -dx)
        elif dy < 0:
            angle = math.pi + math.atan(dx / dy)
    return angle


# 模拟移动，防止出界
def move(track_point, random_x, random_y):
    if track_point[0] + random_x >= right_boundary or track_point[0] + random_x <= left_boundary:
        track_point[0] = track_point[0] - random_x
    else:
        track_point[0] = track_point[0] + random_x

    if track_point[1] + random_y >= right_boundary or track_point[1] + random_y <= left_boundary:
        track_point[1] = track_point[1] - random_y
    else:
        track_point[1] = track_point[1] + random_y
    return track_point


# 计算两点之间的距离
def distance(point_1, point_2):
    return math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)


def calculate_candidate_point(other_locs, track_point, candidates_angle_list, time_next, step_dis, max_angle,
                              uav_index):
    candidate_point = []
    for angle in candidates_angle_list:
        if abs(angle - max_angle) < math.pi / 2:
            candidates_x = track_point[0] + step_dis * math.sin(angle)
            candidates_y = track_point[1] + step_dis * math.cos(angle)
            if right_boundary >= candidates_x >= left_boundary and right_boundary >= candidates_y >= left_boundary:
                candidate_point.append([candidates_x, candidates_y, time_next])
    candidate_point_result = []
    loc_to_compute = []
    for i in range(uav_num):
        if i == uav_index:
            continue
        if not other_locs[i]:
            continue
        loc = other_locs[i]
        if d_min + 2 * step_dis < distance(loc, track_point) < d_max - 2 * step_dis:
            continue
        else:
            loc_to_compute.append(loc)
    for point in candidate_point:
        judge = True
        for loc in loc_to_compute:
            if d_min < distance(loc, point) < d_max:
                continue
            else:
                judge = False
                break
        if judge:
            candidate_point_result.append(point)
    return candidate_point_result


def track_test(track_point, csv_index):
    csv_name_1 = 'data/test2/multi_simulation_' + str(csv_index) + '_uav_0.csv'
    csv_name_2 = 'data/test2/multi_simulation_' + str(csv_index) + '_uav_1.csv'
    csv_name_3 = 'data/test2/multi_simulation_' + str(csv_index) + '_uav_2.csv'
    csv_list = [csv_name_1, csv_name_2, csv_name_3]
    global rssi_data_list, loc_data_list
    index = 0
    time_track = 0.0
    coordinate = queue.Queue()  # 三架无人机共用一组coordinate和rssi_value
    rssi_value = queue.Queue()

    # print(coordinate.qsize())
    source = copy.copy(source_point)
    distance_u_s = [12.5, 12.5, 12.5]  # 对于动态源追踪的场景，要满足连续两次追踪无人机与信号源之间的距离小于阈值才认为追踪成功

    while True:
        other_locs_desi = []  # 记录其他无人机当前轮次的计算结果，用于加入train_X，计算当前无人机下一步的位置
        # 先测量信号强度值并记录时间
        # time_track += random.uniform(0.0, 1.0)
        for i in range(uav_num):
            rssi = get_rssi(track_point[i],
                            [source_point[0] + time_track * source_v_x, source_point[1] + time_track * source_v_y])
            # 将数据存入自己的队列中，并放入其他无人机的消息队列
            if time_track != 0.0:
                message = [track_point[i][0], track_point[i][1], time_track]
                coordinate.put(message)
                rssi_value.put(rssi)

        if time_track > 50:
            df = pd.DataFrame({"file_index": [csv_index], "fly_times": [index], "time_spent": time_track})
            df.to_csv(times_csv, mode='a', index=None, header=None)
            print("track failed.")
            return
        if index < 3:
            # 随机运动三次
            random_angel = random.uniform(0.0, math.pi * 2)
            random_x = random_distance * math.sin(random_angel)
            random_y = random_distance * math.cos(random_angel)
            track_point[0] = move(track_point[0], random_x, random_y)
            track_point[1] = move(track_point[1], random_x, random_y)
            track_point[2] = move(track_point[2], random_x, random_y)

            time_track += random_distance / v_track
            source[0] = source_point[0] + time_track * source_v_x
            source[1] = source_point[1] + time_track * source_v_y
            for i in range(uav_num):
                df = pd.DataFrame({"coordinate": [track_point[i]], "max_rssi_pos": [track_point[i]]
                                      , "source_point": [source], "t": time_track, "rssi": rssi})
                df.to_csv(csv_list[i], mode='a', index=None, header=None)
        elif index < times_threshold_value:  # 探索阶段
            for i in range(uav_num):
                other_locs_desi.append([])
            # 计算
            # 首先获取并处理其他无人机发过来的位置信息
            time_track += distance_1 / v_track
            move_res = []
            for k in range(uav_num):
                move_res.append([])
            for k in range(0, k_for_cal):
                coor_change_list = []
                for j in range(uav_num):
                    t = track_point[j]
                    max_cordinate_predicted, max_rssi_predicted, kff_inv, mu, cov = get_max_point(time_track,
                                                                                                  coordinate,
                                                                                                  rssi_value)
                    max_angle = calculate_angle(t[0], t[1], max_cordinate_predicted[0], max_cordinate_predicted[1])
                    ad = ads.ADS(list(coordinate.queue), list(rssi_value.queue))
                    candidates_angle_list = ad.calculate_candidates_points_angle()
                    candidate_point_list = calculate_candidate_point(other_locs_desi, t,
                                                                     candidates_angle_list,
                                                                     time_track,
                                                                     distance_1, max_angle, j)
                    # 如果筛选完之后候选点数量为0，则保持当前位置不动
                    move_res[j] = track_point[j]
                    if candidate_point_list:
                        # 在coordinate上添加上已经计算过的其他无人机的预计目标位置作为ALC的输入train_x
                        train_X = list(coordinate.queue)
                        train_X.extend(coor_change_list)
                        point_index = ad.ALC(train_X, candidate_point_list, time_track, cov)
                        new_loc = candidate_point_list[point_index]
                        coor_change_list.append(new_loc)
                        move_res[j] = new_loc
                    other_locs_desi[j] = move_res[j]

            # 真正的飞行应发生在外面一层，需要存储位置的计算结果，并且在发送给其他无人机前就要判断目标位置是否合理
            for i in range(uav_num):
                track_point[i] = move(track_point[i], move_res[i][0]-track_point[i][0], move_res[i][1]-track_point[i][1])
            source[0] = source_point[0] + time_track * source_v_x
            source[1] = source_point[1] + time_track * source_v_y
            for i in range(uav_num):
                df = pd.DataFrame({"coordinate": [track_point[i]], "max_rssi_pos": [max_cordinate_predicted]
                                      , "source_point": [source], "t": time_track, "rssi": rssi})
                df.to_csv(csv_list[i], mode='a', index=None, header=None)

            for i in range(uav_num):
                t = track_point[i]
                distance_x_this = abs(source[0] - t[0])
                distance_y_this = abs(source[1] - t[1])
                distance_this_2 = distance_x_this ** 2 + distance_y_this ** 2
                if distance_this_2 <= threshold_distance and distance_u_s[i] <= threshold_distance:
                    df = pd.DataFrame({"file_index": [csv_index], "fly_times": [index], "time_spent": time_track})
                    df.to_csv(times_csv, mode='a', index=None, header=None)
                    print("track succeed!")
                    return
                distance_u_s[i] = distance_this_2

        else:
            # 利用阶段，直接向信号源飞行
            time_track += distance_2 / v_track
            max_cordinate_predicted, max_rssi_predicted, kff_inv, mu, cov = get_max_point(time_track, coordinate,
                                                                                          rssi_value)
            largest_x = max_cordinate_predicted[0]
            largest_y = max_cordinate_predicted[1]
            for i in range(uav_num):
                angle = calculate_angle(track_point[i][0], track_point[i][1], largest_x, largest_y)
                delta_x = distance_2 * math.sin(angle)
                delta_y = distance_2 * math.cos(angle)
                track_point[i] = move(track_point[i], delta_x, delta_y)

            # 判断距离
            source[0] = source_point[0] + time_track * source_v_x
            source[1] = source_point[1] + time_track * source_v_y

            for i in range(uav_num):
                df = pd.DataFrame({"coordinate": [track_point[i]], "max_rssi_pos": [max_cordinate_predicted]
                                      , "source_point": [source], "t": time_track, "rssi": rssi})
                df.to_csv(csv_list[i], mode='a', index=None, header=None)

            for i in range(uav_num):
                t = track_point[i]
                distance_x_this = abs(source[0] - t[0])
                distance_y_this = abs(source[1] - t[1])
                distance_this_2 = distance_x_this ** 2 + distance_y_this ** 2
                if distance_this_2 <= threshold_distance and distance_u_s[i] <= threshold_distance:
                    df = pd.DataFrame({"file_index": [csv_index], "fly_times": [index], "time_spent": time_track})
                    df.to_csv(times_csv, mode='a', index=None, header=None)
                    print("track succeed!")
                    return
                distance_u_s[i] = distance_this_2
        index += 1


if __name__ == '__main__':
    init_rssi()
    for i in range(100, 120):
        track_test(initial_position, i)
