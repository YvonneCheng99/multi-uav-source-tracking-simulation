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
uav_num = 3
initial_position = [[0.0, 0.0], [-0.5, 0.5], [0.5, 0.5]]
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


# 追踪函数
def track(csv_index, track_point):
    csv_name = 'opti_simulation_data_13/opti_simulation_' + str(csv_index) + '.csv'
    index = 0
    time_track = 0.0
    # global source_point, coordinate, rssi_value
    coordinate = queue.Queue()
    rssi_value = queue.Queue()
    # print(coordinate.qsize())
    distance = 12.5
    print('start-------------------------')
    print('track_point', end='')
    print(track_point)
    while (1):
        if index < 3:
            # 随机运动三次
            random_angel = random.uniform(0.0, math.pi * 2)
            random_x = random_distance * math.sin(random_angel)
            random_y = random_distance * math.cos(random_angel)
            time_track += random_distance / v_track  # 增加时间
            # 移动
            # 模拟移动,出界向反方向移动
            if track_point[0] + random_x >= 1.5 or track_point[0] + random_x <= -1.5:
                track_point[0] = track_point[0] - random_x
            else:
                track_point[0] = track_point[0] + random_x

            if track_point[1] + random_y >= 1.5 or track_point[1] + random_y <= -1.5:
                track_point[1] = track_point[1] - random_y
            else:
                track_point[1] = track_point[1] + random_y

            coordinate.put([track_point[0], track_point[1], time_track])
            rssi = get_rssi()
            rssi_value.put(rssi)
            source_point[0] = source_point[0] + source_v_x * random_distance / v_track
            source_point[1] = source_point[1] + source_v_y * random_distance / v_track
            # 记录数据
            x = track_point[0]
            y = track_point[1]
            # df = pd.DataFrame({"coordinate": [track_point], "t": time_track, "rssi": rssi})
            df = pd.DataFrame(
                {"coordinate": [track_point], "max_rssi_pos": [track_point]
                    , "source_point": [source_point], "t": time_track, "rssi": rssi})
            df.to_csv(csv_name, mode='a', index=None, header=None)
            # 判断是否追踪到
            distance_x_this = abs(source_point[0] - track_point[0])
            distance_y_this = abs(source_point[1] - track_point[1])
            distance_this_2 = math.sqrt(distance_x_this ** 2 + distance_y_this ** 2)
            if distance_this_2 <= threshold_distance and distance <= threshold_distance:
                return
            distance = distance_this_2
            time_track += 2.0
            source_point[0] = source_point[0] + source_v_x * 2.0
            source_point[1] = source_point[1] + source_v_y * 2.0
            index += 1
        else:

            if time_track > 85:
                df = pd.DataFrame({"file_index": [csv_index], "fly_times": [index], "time_spent": time_track})
                df.to_csv('opti_times_13.csv', mode='a', index=None, header=None)
                print("track failed.")
                return

            if index < times_threshold_value:
                d = distance_1
            else:
                d = distance_2

            time_track += d / v_track
            # 高斯过程
            max_cordinate_predicted, max_rssi_predicted, kff_inv, mu, cov = get_max_point(time_track)
            max_angle = calculate_angle(track_point[0], track_point[1], max_cordinate_predicted[0],
                                        max_cordinate_predicted[1])
            '''
            ad = ads.ADS(list(coordinate.queue), list(rssi_value.queue))
            next_pos = ad.next_step(index, track_point, kff_inv, time_track, list(coordinate.queue), list(rssi_value.queue), mu, cov)
            print('next_pos', end='')
            print(next_pos)
            track_point[0] = next_pos[0]
            track_point[1] = next_pos[1]
            '''
            if index < times_threshold_value:
                ad = ads.ADS(list(coordinate.queue), list(rssi_value.queue))
                next_pos = ad.next_step(index, track_point, kff_inv, time_track, list(coordinate.queue),
                                        list(rssi_value.queue), mu, cov, distance_1, max_angle)
                print('next_pos', end='')
                print(next_pos)
                track_point[0] = next_pos[0]
                track_point[1] = next_pos[1]
            else:
                # 移动
                largest_x = max_cordinate_predicted[0]
                largest_y = max_cordinate_predicted[1]
                print("max_coordinate:")
                print(str(largest_x) + "," + str(largest_y))
                print("max rssi:")
                print(max_rssi_predicted)
                angle = calculate_angle(track_point[0], track_point[1], largest_x, largest_y)
                delta_x = d * math.sin(angle)
                delta_y = d * math.cos(angle)

                # 模拟移动,出界向反方向移动
                if track_point[0] + delta_x > 1.5 or track_point[0] + delta_x < -1.5:
                    track_point[0] = track_point[0] - delta_x
                else:
                    track_point[0] = track_point[0] + delta_x

                if track_point[1] + delta_y > 1.5 or track_point[1] + delta_y < -1.5:
                    track_point[1] = track_point[1] - delta_y
                else:
                    track_point[1] = track_point[1] + delta_y
            rssi = get_rssi()
            coordinate.put([track_point[0], track_point[1], time_track])
            rssi_value.put(rssi)
            source_point[0] = source_point[0] + source_v_x * d / v_track
            source_point[1] = source_point[1] + source_v_y * d / v_track
            # 记录数据
            df = pd.DataFrame({"coordinate": [track_point], "max_rssi_pos": [max_cordinate_predicted]
                                  , "source_point": [source_point], "t": time_track, "rssi": rssi})
            # df = pd.DataFrame({"coordinate": [track_point], "t": time_track, "rssi": rssi})
            df.to_csv(csv_name, mode='a', index=None, header=None)
            # 判断是否追追踪到
            distance_x_this = abs(source_point[0] - track_point[0])
            distance_y_this = abs(source_point[1] - track_point[1])
            distance_this_2 = math.sqrt(distance_x_this ** 2 + distance_y_this ** 2)
            if distance_this_2 <= threshold_distance and distance <= threshold_distance:
                df = pd.DataFrame({"file_index": [csv_index], "fly_times": [index], "time_spent": time_track})
                df.to_csv('opti_times_13.csv', mode='a', index=None, header=None)
                "track succeed!"
                return
            distance = distance_this_2
            time_track += 2.0
            source_point[0] = source_point[0] + source_v_x * 2.0
            source_point[1] = source_point[1] + source_v_y * 2.0
            index += 1


def track_test(track_point, start_time, uav_index):
    file_name = 'test' + str(uav_index)
    global rssi_data_list, loc_data_list
    index = 0
    coordinate = queue.Queue()
    rssi_value = queue.Queue()
    recv_record = []  # 记录该无人机上次收到其他无人机的消息对应的迭代次数
    for i in range(uav_num):
        recv_record.append(-1)
    other_locs = []
    for i in range(uav_num):
        
    # print(coordinate.qsize())
    distance = 12.5
    f = open(file_name, 'a')
    print(uav_index, end='', file=f)
    print(' start-------------------------', file=f)
    while True:
        # 接收其他无人机发过来的数据
        while not rssi_data_list[uav_index].empty():
            data_recv = rssi_data_list[uav_index].get()
            coor_data = [data_recv[0], data_recv[1], data_recv[2]]
            coordinate.put(coor_data)
            rssi_value.put(data_recv[3])
            send_uav = data_recv[4]
            recv_record[send_uav] = index
        if index < 3:
            # 先测量信号强度值并记录时间
            time_track = time.time() - start_time

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
            # 随机运动三次
            # 如果是零号无人机，就计算随机角度，并将变化量发给其他无人机，如果不是零号无人机，从队列中取角度
            if uav_index == 0:
                random_angel = random.uniform(0.0, math.pi * 2)
                random_x = random_distance * math.sin(random_angel)
                random_y = random_distance * math.cos(random_angel)
                for i in range(1, uav_num):
                    loc_data_list[i].put([0, index, uav_index, random_x, random_y, time_track+random_distance/v_track])
                track_point = move(track_point, random_x, random_y)
            else:
                while loc_data_list[uav_index].empty():
                    time.sleep(0.1)
                random_move = loc_data_list[uav_index].get()
                random_x = random_move[1]
                random_y = random_move[2]
                track_point = move(track_point, random_x, random_y)
            time.sleep(random_distance / v_track)
            f = open(file_name, 'a')
            print('uav: %d, move to: ' % uav_index, end='', file=f)
            print(track_point, file=f)
            f.close()
            index += 1
        elif index < times_threshold_value: # 探索阶段
            # 同样先获取信号强度值并记录时间
            time_track = time.time() - start_time
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

            # 计算
            # 首先获取并处理其他无人机发过来的位置信息
            # 定义一个list存储无人机接收到的其他无人机的位置变化信息
            coor_change_list = []
            flag_list = [False]*uav_index
            cal_flag = False # 是否开始计算，是否已经收到了需要的所有数据
            while not cal_flag:
                while not loc_data_list[uav_index].empty():
                    loc_recv = loc_data_list[uav_index].get()
                    if loc_recv[0] == 1 and loc_recv[1] >= index:  # 判断数据是否是本次迭代的数据，不使用历史数据
                        if loc_recv[2] < uav_index:
                            coor_change_list.append(loc_recv[3:])
                            flag_list[loc_recv[2]] = True
                # 判断是否收到了index<i的所有无人机的数据，收到之后开始计算
                recv_num = 0
                for i in range(uav_index-1):
                    if flag_list[i]:
                        recv_num+=1
                if recv_num==uav_index-1:
                    cal_flag = True
                else:
                    time.sleep(0.2)

            # 可以开始计算,调用ALC函数
            max_cordinate_predicted, max_rssi_predicted, kff_inv, mu, cov = get_max_point(time_track, coordinate, rssi_value)
            max_angle = calculate_angle(track_point[0], track_point[1], max_cordinate_predicted[0],
                                        max_cordinate_predicted[1])
            ad = ads.ADS(list(coordinate.queue), list(rssi_value.queue))
            candidate_angle_list = ad.calculate_candidates_points_angle()

            next_pos = ad.next_step(index, track_point, kff_inv, time_track, list(coordinate.queue),
                                        list(rssi_value.queue), mu, cov, distance_1, max_angle)
            print('next_pos', end='')
            print(next_pos)
            track_point[0] = next_pos[0]
            track_point[1] = next_pos[1]
        else:
            # 利用阶段，直接向信号源飞行


if __name__ == '__main__':
    init_rssi()
    for i in range(uav_num):
        rssi_data_list.append(queue.Queue())
        loc_data_list.append(queue.Queue())
    start_time = time.time()
    thread_list = []
    t1 = threading.Thread(target=track_test, args=([0.0, 0.0], start_time, 0))
    t2 = threading.Thread(target=track_test, args=([0.5, 0.5], start_time, 1))
    t3 = threading.Thread(target=track_test, args=([-0.5, -0.5], start_time, 2))
    thread_list.append(t1)
    thread_list.append(t2)
    thread_list.append(t3)
    for i in range(uav_num):
        thread_list[i].start()
    for i in range(uav_num):
        thread_list[i].join()
