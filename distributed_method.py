import numpy as np


# 目标函数：凸函数 f(x) = x^2
def objective_function(x):
    return np.sum(x ** 2)


# 分布式梯度下降算法
def distributed_gradient_descent(learning_rate, num_agents, num_iterations):
    # 初始化每个智能体的决策变量
    x = np.zeros(num_agents)

    for _ in range(num_iterations):
        # 计算每个智能体的梯度
        gradient = 2 * x

        # 更新每个智能体的决策变量
        x -= learning_rate * gradient / num_agents

        # 打印每次迭代的目标函数值
        print("Objective value:", objective_function(x))

    return x


if __name__ == "__main__":
    learning_rate = 0.1
    num_agents = 5
    num_iterations = 100

    final_solution = distributed_gradient_descent(learning_rate, num_agents, num_iterations)
    print("Final solution:", final_solution)