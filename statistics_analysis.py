import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

times = []
time_spent = []
times_2 = []


def init_data():
    global times, time_spent, times_2
    data = pd.read_csv("multi_times_5_2.csv", header=None)
    # print(data)
    data_1 = data.iloc[:, 0]
    index = np.array(data_1)

    data_2 = data.iloc[:, 1]
    times = np.array(data_2)

    data_3 = data.iloc[:, 2]
    time_spent = np.asarray(data_3)

    print(index)
    print(times)
    print(time_spent)
    '''
    data = pd.read_csv("multi_times_4.csv", header=None)
    data_2 = data.iloc[:, 1]
    times_2 = np.array(data_2)
    print(times_2)
    '''


init_data()
print(np.mean(times))
print(max(times))
print(min(times))
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
bin = math.ceil((max(times)-min(times))/5)
plt.hist(times, color='lightblue', bins=bin, edgecolor='black', linewidth=0.5, alpha=0.5, label='')
# plt.hist(times_2, color='lightgreen', bins=15, range=[0, 105], edgecolor='black', linewidth=0.5, alpha=0.5)
# plt.hist([times_2, times], bins=7, color=['orange', '#2ca02c'], range=[0, 105], edgecolor='black', linewidth=0.5,
#          alpha=0.85, label=['实验一', '实验五'])
plt.xlabel('完成任务所需移动步数')
plt.ylabel('频数')
plt.legend(loc='upper right')
plt.show()