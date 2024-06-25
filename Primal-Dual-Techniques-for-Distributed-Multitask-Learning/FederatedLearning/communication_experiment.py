import numpy as np
import matplotlib.pyplot as plt

# 定义环境大小
GRID_SIZE = 5
GOAL_POSITION = (4, 4)


# 初始化Q表
def initialize_q_table():
    return np.zeros((GRID_SIZE, GRID_SIZE, 4))  # 上下左右四个动作


# 定义动作
actions = ['up', 'down', 'left', 'right']


# 获取下一个状态
def get_next_state(state, action):
    x, y = state
    if action == 'up' and x > 0:
        x -= 1
    elif action == 'down' and x < GRID_SIZE - 1:
        x += 1
    elif action == 'left' and y > 0:
        y -= 1
    elif action == 'right' and y < GRID_SIZE - 1:
        y += 1
    return (x, y)


# 选择动作
def choose_action(state, q_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        x, y = state
        return actions[np.argmax(q_table[x, y])]


# 更新Q值
def update_q_value(q_table, state, action, reward, next_state, alpha, gamma):
    x, y = state
    next_x, next_y = next_state
    action_idx = actions.index(action)
    q_table[x, y, action_idx] += alpha * (reward + gamma * np.max(q_table[next_x, next_y]) - q_table[x, y, action_idx])


# 运行实验
def run_experiment(communication_frequency, episodes=1000):
    q_tables = [initialize_q_table() for _ in range(5)]  # 5个agents
    epsilon = 0.1  # 探索率
    alpha = 0.1  # 学习率
    gamma = 0.9  # 折扣因子
    steps_to_goal = []

    for episode in range(episodes):
        states = [(np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)) for _ in range(5)]
        step_count = 0
        while any(state != GOAL_POSITION for state in states):
            for i, state in enumerate(states):
                action = choose_action(state, q_tables[i], epsilon)
                next_state = get_next_state(state, action)
                reward = 1 if next_state == GOAL_POSITION else -1
                update_q_value(q_tables[i], state, action, reward, next_state, alpha, gamma)
                states[i] = next_state

            # agents之间的通信
            if communication_frequency != float('inf') and step_count % communication_frequency == 0:
                avg_q_table = np.mean(q_tables, axis=0)
                q_tables = [avg_q_table.copy() for _ in range(5)]

            step_count += 1
        steps_to_goal.append(step_count)

    return np.mean(steps_to_goal)


# 不同通信频率下的实验
communication_frequencies = [
    float('inf'), 20, 15, 10, 8, 6, 5, 4, 3, 2, 1
]
results = []

for freq in communication_frequencies:
    avg_steps = run_experiment(freq)
    results.append(avg_steps)

# 绘制结果
plt.figure(figsize=(14, 8))
plt.plot(
    ['None', 'Every 20 steps', 'Every 15 steps', 'Every 10 steps', 'Every 8 steps', 'Every 6 steps', 'Every 5 steps',
     'Every 4 steps', 'Every 3 steps', 'Every 2 steps', 'Every step'], results, marker='o')
plt.title('Communication Frequency vs Average Steps to Goal')
plt.xlabel('Communication Frequency')
plt.ylabel('Average Steps to Goal')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
