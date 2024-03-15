import numpy as np
import pandas as pd
import queue
import random
import math
import Gaussian_progress_dynamic as gp
import parameters as pm


source_point = pm.source_point  # 源的初始位置
source_v_x = pm.source_v_x  # 源在x轴方向的运动速度
source_v_y = pm.source_v_y  # 源在y轴方向的运动速度
track_point_1 = pm.track_point_1  # 追踪方1的初始位置
track_point_2 = pm.track_point_2  # 追踪方2的初始位置
track_point_3 = pm.track_point_3  # 追踪方3的初始位置
random_distance = pm.random_distance  # 刚开始随机方向移动每次移动的距离
v_track = pm.v_track  # 追踪方移动速度
random_times = pm.random_times  # 刚开始随机移动的次数
distance_1 = pm.distance_1  # 向计算出的强度最大的位置移动的距离
distance_2 = pm.distance_2
threshold_distance = pm.threshold_distance
rssi_noise = pm.rssi_noise
point_max_rssi = []  # 信号强度最大的点的位置
grid_point_list = []  # 栅格化的所有点的list
rssi_distribution_list = []  # 所有点对应的信号强度值的list
# coordinate_1 = queue.Queue()  # 坐标和时间队列，作为高斯过程的输入
# rssi_value_1 = queue.Queue()  # 信号强度值队列，与坐标和时间队列为一一对应的关系
# coordinate_2 = queue.Queue()  # 坐标和时间队列，作为高斯过程的输入
# rssi_value_2 = queue.Queue()  # 信号强度值队列，与坐标和时间队列为一一对应的关系
# coordinate_3 = queue.Queue()  # 坐标和时间队列，作为高斯过程的输入
# rssi_value_3 = queue.Queue()  # 信号强度值队列，与坐标和时间队列为一一对应的关系
times_threshold_value = pm.time_to_target  # 两个阶段划分的阈值

class Drone:
    def __init__(self, x, y):
        self.position = [x,y]
        self.track_time = 0.0
        self.coordinate = queue.Queue()
        self.rssi_list = queue.Queue()

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

def get_rssi(track_point):
    x = track_point[0] + point_max_rssi[0] - source_point[0]
    y = track_point[1] + point_max_rssi[1] - source_point[1]
    coordinate_approximate = [round(x, 1), round(y, 1)]
    i = grid_point_list.index(coordinate_approximate)
    rssi = rssi_distribution_list[i] + np.random.normal(0, rssi_noise)
    return rssi

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

def fin_max(ls):
    max_rssi = -100
    index = -1
    for i in range(0, len(ls)):
        if ls[i] > max_rssi:
            max_rssi = ls[i]
            index = i
    return index, max_rssi

def get_max_point(time_track, drone):
    gpr = gp.GPR(optimize=True)
    test_d1 = np.arange(-2.5, 2.5, 0.1)
    test_d2 = np.arange(-2.5, 2.5, 0.1)
    test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
    # test_X_array = np.asarray(test_X)
    # test_X_t = np.reshape(test_X_array, )
    total_sample = drone.coordinate.qsize()
    coordinate_list = list(drone.coordinate.queue)
    rssi_value_list = list(drone.rssi_value.queue)
    t_list = [time_track] * len(test_d1) * len(test_d2)
    test_X = [[d1, d2, d3] for d1, d2, d3 in zip(test_d1.ravel(), test_d2.ravel(), t_list)]

    for i in (0, total_sample):
        gpr.fit(coordinate_list, rssi_value_list)
        mu, cov = gpr.predict(test_X)
        Kff_inv = gpr.get_Kff_inv()
        max_index, max_mu = fin_max(mu)
        return test_X[max_index], max_mu, Kff_inv, mu, cov

def random_move(drone):
    random_angel = random.uniform(0.0, math.pi * 2)
    random_x = random_distance * math.sin(random_angel)
    random_y = random_distance * math.cos(random_angel)
    track_point = drone.position
    # 模拟移动,出界向反方向移动
    if track_point[0] + random_x >= 2.5 or track_point[0] + random_x <= -2.5:
        track_point[0] = track_point[0] - random_x
    else:
        track_point[0] = track_point[0] + random_x

    if track_point[1] + random_y >= 2.5 or track_point[1] + random_y <= -2.5:
        track_point[1] = track_point[1] - random_y
    else:
        track_point[1] = track_point[1] + random_y

'''
def fixed_angle_move(drone, angle):
    x = random_distance * math.sin(angle)
    y = random_distance * math.cos(angle)
    track_point = drone.position
    # 模拟移动,出界向反方向移动
    if track_point[0] + x >= 2.5 or track_point[0] + x <= -2.5:
        track_point[0] = track_point[0] - x
    else:
        track_point[0] = track_point[0] + x

    if track_point[1] + y >= 2.5 or track_point[1] + y <= -2.5:
        track_point[1] = track_point[1] - y
    else:
        track_point[1] = track_point[1] + y
'''

def cal_distance(drone_1, drone_2):
    point_1 = drone_1.position
    point_2 = drone_2.position
    return math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)

def write_to_csv(drone, csv_name, max_pos, rssi):
    df = pd.DataFrame(
        {"coordinate": [drone.position], "max_rssi_pos": [max_pos], "t": drone.track_time, "rssi": rssi})
    df.to_csv(csv_name, mode='a', index=None, header=None)

def track(csv_index):
    csv_name_1 = 'data/multi_simulation_' + str(csv_index) + '_1.csv'
    csv_name_2 = 'data/multi_simulation_' + str(csv_index) + '_2.csv'
    csv_name_3 = 'data/multi_simulation_' + str(csv_index) + '_3.csv'
    index = 0
    # time_track = 0
    # 初始化三架无人机，这三架无人机一起完成追踪任务
    drone_1 = Drone(track_point_1[0], track_point_1[1])
    drone_2 = Drone(track_point_2[0], track_point_2[1])
    drone_3 = Drone(track_point_3[0], track_point_3[1])
    distance = 12.5 # 对于动态源追踪的场景，要满足连续两次追踪无人机与信号源之间的距离小于阈值才认为追踪成功
    print('start-------------------------')
    # print('track_point', end='')
    # print(track_point)
    col_th = pm.collision_threshold_value
    com_th = pm.communication_threshold_value

    while (1):
        if index < 3:
            # 随机运动三次
            # 三架无人机分别随机移动，按顺序来且要符合约束条件，不能飞出通信范围或发生碰撞
            random_move(drone_1)
            random_move(drone_2)
            random_move(drone_3)
            while cal_distance(drone_1, drone_2) > com_th or cal_distance(drone_1, drone_2) < col_th\
                    or cal_distance(drone_1, drone_3) > 2.0 or cal_distance(drone_1, drone_3) < 0.3\
                    or cal_distance(drone_2, drone_3) > 2.0 or cal_distance(drone_2, drone_3) < 0.3:
                random_move(drone_1)
                random_move(drone_2)
                random_move(drone_3)
            drone_1.track_time += random_distance / v_track  # 增加时间
            drone_2.track_time += random_distance / v_track  # 增加时间
            drone_3.track_time += random_distance / v_track  # 增加时间

            # 另一种方式：三架无人机保持最初的相对位置，随机移动三次
            '''
            random_angel = random.uniform(0.0, math.pi * 2)
            fixed_angle_move(drone_1, random_angle)
            fixed_angle_move(drone_2, random_angle)
            fixed_angle_move(drone_3, random_angle)
            '''
            # 模拟信号源的移动
            source_point[0] = source_point[0] + source_v_x * random_distance / v_track
            source_point[1] = source_point[1] + source_v_y * random_distance / v_track

            drone_1.coordinate.put([track_point_1[0], track_point_1[1], drone_1.track_time])
            rssi = get_rssi(drone_1.position)
            drone_1.rssi_list.put(rssi)

            # 记录数据
            write_to_csv(drone_1, csv_name_1, drone_1.position, rssi)
            write_to_csv(drone_2, csv_name_2, drone_2.position, rssi)
            write_to_csv(drone_3, csv_name_3, drone_3.position, rssi)

            # 判断是否追踪到
            # TODO 判断是否追踪到
            '''
            distance_x_this = abs(source_point[0] - track_point[0])
            distance_y_this = abs(source_point[1] - track_point[1])
            distance_this_2 = distance_x_this ** 2 + distance_y_this ** 2
            if distance_this_2 <= threshold_distance and distance <= threshold_distance:
                return
            distance = distance_this_2
            '''
            index += 1
        else:

            if drone_1.track_time > 50:
                df = pd.DataFrame({"file_index": [csv_index], "fly_times": [index], "time_spent": drone_1.track_time})
                df.to_csv('opti_times_8.csv', mode='a', index=None, header=None)
                print("track failed.")
                return

            if index < times_threshold_value:
                d = distance_1
            else:
                d = distance_2

            time_track += d / v_track
            # 高斯过程
            max_cordinate_predicted, max_rssi_predicted, kff_inv, mu, cov = get_max_point(time_track)
            max_angle = calculate_angle(track_point[0], track_point[1], max_cordinate_predicted[0], max_cordinate_predicted[1])
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
                if track_point[0] + delta_x > 2.5 or track_point[0] + delta_x < -2.5:
                    track_point[0] = track_point[0] - delta_x
                else:
                    track_point[0] = track_point[0] + delta_x

                if track_point[1] + delta_y > 2.5 or track_point[1] + delta_y < -2.5:
                    track_point[1] = track_point[1] - delta_y
                else:
                    track_point[1] = track_point[1] + delta_y

            source_point[0] = source_point[0] + source_v_x * d / v_track
            source_point[1] = source_point[1] + source_v_y * d / v_track
            rssi = get_rssi()
            coordinate.put([track_point[0], track_point[1], time_track])
            rssi_value.put(rssi)
            # 记录数据
            df = pd.DataFrame({"coordinate": [track_point], "max_rssi_pos": [max_cordinate_predicted], "t": time_track, "rssi": rssi})
            # df = pd.DataFrame({"coordinate": [track_point], "t": time_track, "rssi": rssi})
            df.to_csv(csv_name, mode='a', index=None, header=None)
            # 判断是否追追踪到
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
        '''


if __name__ == '__main__':
    init_rssi()
    for i in range(0, 1):
        track(i)