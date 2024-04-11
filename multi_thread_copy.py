import threading
import time
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
uav_num = 1
initial_position = [[0.0, 0.0], [-0.5, 0.5], [0.5, 0.5]]
candidates_number = pm.candidates_number
k_for_cal = pm.k_for_calculate
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
    if x >= 5.0:
        x = 5.0
    if y >= 5.0:
        y = 5.0
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
    return math.sqrt((point_1[0] - point_2[0] ** 2 + (point_1[1] - point_2[1]) ** 2))


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


def track_test(track_point, start_time, uav_index, csv_index):
    file_name = 'test' + str(uav_index)
    csv_name = 'data/multi_simulation_' + str(csv_index) + '_uav_' + str(uav_index)
    global rssi_data_list, loc_data_list
    index = 0
    time_track = 0
    coordinate = queue.Queue()
    rssi_value = queue.Queue()
    recv_record = []  # 记录该无人机上次收到其他无人机的消息对应的迭代次数
    for i in range(uav_num):
        recv_record.append(-1)
    other_locs_last_index = []  # 维护其他无人机的位置信息，记录本次迭代开始之前无人机的位置
    other_locs_desi = []  # 记录其他无人机当前轮次的计算结果，用于加入trainx，计算当前无人机下一步的位置
    for i in range(uav_num):
        loc_i = [initial_position[i][0], initial_position[i][1]]
        other_locs_last_index.append(loc_i)
        # other_locs_desi.append(loc_i)
    # print(coordinate.qsize())
    distance = 12.5
    f = open(file_name, 'a')
    print(uav_index, end='', file=f)
    print(' start-------------------------', file=f)
    while True:
        # TODO 更新两个other_locs列表
        other_locs_last_index = other_locs_desi
        # 接收其他无人机发过来的数据
        while not rssi_data_list[uav_index].empty():
            data_recv = rssi_data_list[uav_index].get()
            coor_data = [data_recv[0], data_recv[1], data_recv[2]]
            coordinate.put(coor_data)
            rssi_value.put(data_recv[3])
            send_uav = data_recv[4]
            recv_record[send_uav] = index

        # 先测量信号强度值并记录时间
        rssi = get_rssi(track_point,
                            [source_point[0] + time_track * source_v_x, source_point[1] + time_track * source_v_y])
        # 将数据存入自己的队列中，并放入其他无人机的消息队列
        message = [track_point[0], track_point[1], time_track]
        coordinate.put(message)
        rssi_value.put(rssi)
        message.append(rssi)
        message.append(uav_index)
        for i in range(uav_num):
            if i == uav_index:
                continue
            else:
                rssi_data_list[i].put(message)
        distance = 12.5  # 对于动态源追踪的场景，要满足连续两次追踪无人机与信号源之间的距离小于阈值才认为追踪成功
        if time_track > 85:
            df = pd.DataFrame({"file_index": [csv_index], "fly_times": [index], "time_spent": time_track})
            df.to_csv('opti_times_13.csv', mode='a', index=None, header=None)
            print("track failed.")
            return
        if index < 3:
            # 随机运动三次
            # 如果是零号无人机，就计算随机角度，并将变化量发给其他无人机，如果不是零号无人机，从队列中取角度
            if uav_index == 0:
                random_angel = random.uniform(0.0, math.pi * 2)
                random_x = random_distance * math.sin(random_angel)
                random_y = random_distance * math.cos(random_angel)
                for i in range(1, uav_num):
                    loc_data_list[i].put(
                        [0, index, uav_index, random_x, random_y, time_track + random_distance / v_track])
                track_point = move(track_point, random_x, random_y)
            else:
                while loc_data_list[uav_index].empty():
                    time.sleep(0.1)
                random_move = loc_data_list[uav_index].get()
                random_x = random_move[1]
                random_y = random_move[2]
                track_point = move(track_point, random_x, random_y)
            time.sleep(random_distance / v_track)
            time_now = time.time() - start_time
            source_point[0] = source_point[0] + time_now*source_v_x
            source_point[1] = source_point[1] + time_now*source_v_y
            f = open(file_name, 'a')
            print('uav: %d, move to: ' % uav_index, end='', file=f)
            print(track_point, file=f)
            f.close()
            df = pd.DataFrame({"coordinate": [track_point], "max_rssi_pos": [track_point]
                                  , "source_point": [source_point], "t": time_now, "rssi": rssi})
            df.to_csv(csv_name, mode='a', index=None, header=None)
        elif index < times_threshold_value:  # 探索阶段
            # 计算
            # 首先获取并处理其他无人机发过来的位置信息
            new_loc = []
            for k in range(0, k_for_cal):
                for i in range(uav_num):
                    other_locs_desi.append([])
                # 定义一个list存储无人机接收到的其他无人机的位置变化信息
                coor_change_list = []
                flag_list = [False] * uav_num
                cal_flag = False  # 是否开始计算，是否已经收到了需要的所有数据
                while not cal_flag:
                    while not loc_data_list[uav_index].empty():
                        loc_recv = loc_data_list[uav_index].get()
                        if loc_recv[0] == 1 and loc_recv[1] >= index:  # 判断数据是否是本次迭代的数据，不使用历史数据
                            coor_change_list.append(loc_recv[3:])
                            other_locs_desi[loc_recv[2]] = loc_recv[3:]
                            flag_list[loc_recv[2]] = True
                            '''
                            if loc_recv[2] < uav_index:
                                coor_change_list.append(loc_recv[3:])
                                flag_list[loc_recv[2]] = True
                            '''
                    # 判断是否收到应该收到的所有无人机的数据，收到之后开始计算
                    recv_num = 0
                    if k == 0:
                        for i in range(uav_index - 1):
                            if flag_list[i]:
                                recv_num += 1
                    else:
                        for i in range(uav_num):
                            if flag_list[i]:
                                recv_num += 1

                    if k == 0 and recv_num == uav_index - 1:
                        cal_flag = True
                    elif k > 0 and recv_num == uav_num - 1:
                        cal_flag = True
                    else:
                        time.sleep(0.2)

                # 可以开始计算,调用ALC函数
                time_track = time.time() - start_time + distance_1/v_track
                max_cordinate_predicted, max_rssi_predicted, kff_inv, mu, cov = get_max_point(time_track, coordinate,
                                                                                              rssi_value)
                max_angle = calculate_angle(track_point[0], track_point[1], max_cordinate_predicted[0],
                                            max_cordinate_predicted[1])
                ad = ads.ADS(list(coordinate.queue), list(rssi_value.queue))
                candidates_angle_list = ad.calculate_candidates_points_angle()
                candidate_point_list = calculate_candidate_point(coor_change_list, track_point, candidates_angle_list,
                                                                 time_track,
                                                                 distance_1, max_angle, uav_index)
                # 在coordinate上添加上已经计算过的其他无人机的预计目标位置作为ALC的输入train_x
                train_X = coordinate
                train_X.extend(coor_change_list)
                index = ad.ALC(train_X, candidate_point_list, kff_inv, time_track, cov)
                # 向其他无人机发送信息,广播所有已有数据
                new_loc = candidate_point_list[index]
                other_locs_desi[uav_index] = new_loc
                for l in other_locs_desi:
                    if len(l)==0:
                        continue
                    for i in range(1, uav_num):
                        loc_data_list[i].put(
                            [0, index, uav_index, new_loc[0], new_loc[1], time_track + distance_1 / v_track])
            # 真正的飞行应发生在外面一层，需要存储位置的计算结果，并且在发送给其他无人机前就要判断目标位置是否合理
            track_point = move(track_point, new_loc[0]-track_point[0], new_loc[1]-track_point[1])
            time_now = time.time() - start_time
            source_point[0] = source_point[0] + time_now * source_v_x
            source_point[1] = source_point[1] + time_now * source_v_y
            df = pd.DataFrame({"coordinate": [track_point], "max_rssi_pos": [max_cordinate_predicted]
                                  , "source_point": [source_point], "t": time_now, "rssi": rssi})
            df.to_csv(csv_name, mode='a', index=None, header=None)
            distance_x_this = abs(source_point[0] - track_point[0])
            distance_y_this = abs(source_point[1] - track_point[1])
            distance_this_2 = distance_x_this ** 2 + distance_y_this ** 2
            if distance_this_2 <= threshold_distance and distance <= threshold_distance:
                df = pd.DataFrame({"file_index": [csv_index], "fly_times": [index], "time_spent": time_track})
                df.to_csv('opti_times_8.csv', mode='a', index=None, header=None)
                "track succeed!"
                return
            distance = distance_this_2

        else:
            # TODO 根据other_locs_last_index内的数据计算哪架无人机距离信号源最近
            # 利用阶段，直接向信号源飞行
            max_cordinate_predicted, max_rssi_predicted, kff_inv, mu, cov = get_max_point(time_track, coordinate,
                                                                                          rssi_value)
            largest_x = max_cordinate_predicted[0]
            largest_y = max_cordinate_predicted[1]
            # for i in other_locs_last_index:
            #     dis = distance([largest_x, largest_y], i)
            angle = calculate_angle(track_point[0], track_point[1], largest_x, largest_y)
            delta_x = distance_2 * math.sin(angle)
            delta_y = distance_2 * math.cos(angle)

            track_point = move(track_point, delta_x, delta_y)
            # 判断距离

            time_now = time.time() - start_time
            source_point[0] = source_point[0] + time_now * source_v_x
            source_point[1] = source_point[1] + time_now * source_v_y
            df = pd.DataFrame({"coordinate": [track_point], "max_rssi_pos": [track_point]
                                  , "source_point": [source_point], "t": time_now, "rssi": rssi})
            df.to_csv(csv_name, mode='a', index=None, header=None)
            distance_x_this = abs(source_point[0] - track_point[0])
            distance_y_this = abs(source_point[1] - track_point[1])
            distance_this_2 = distance_x_this ** 2 + distance_y_this ** 2
            if distance_this_2 <= threshold_distance and distance <= threshold_distance:
                df = pd.DataFrame({"file_index": [csv_index], "fly_times": [index], "time_spent": time_track})
                df.to_csv('opti_times_8.csv', mode='a', index=None, header=None)
                "track succeed!"
                return
            distance = distance_this_2
        index += 1


if __name__ == '__main__':
    init_rssi()
    for i in range(uav_num):
        rssi_data_list.append(queue.Queue())
        loc_data_list.append(queue.Queue())
    start_time = time.time()
    thread_list = []
    t1 = threading.Thread(target=track_test, args=([0.0, 0.0], start_time, 0, 1))
    # t2 = threading.Thread(target=track_test, args=([0.5, 0.5], start_time, 1, 1))
    # t3 = threading.Thread(target=track_test, args=([-0.5, -0.5], start_time, 2, 1))
    thread_list.append(t1)
    # thread_list.append(t2)
    # thread_list.append(t3)
    for i in range(uav_num):
        thread_list[i].start()
    for i in range(uav_num):
        thread_list[i].join()
