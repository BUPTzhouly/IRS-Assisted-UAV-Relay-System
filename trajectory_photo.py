import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体，确保中文显示正常
plt.rcParams["axes.unicode_minus"] = False  # 解决图像中负号显示问题

def plot_selected_episodes_with_users(df, episodes, colors, labels, user_coords, scale_factor=800):
    fig, ax = plt.subplots(figsize=(10, 8))
    # 绘制放大的无人机轨迹
    for episode, color, label in zip(episodes, colors, labels):
        if episode <= df.shape[0]:
            row = df.iloc[episode - 1]  # Episodes are 1-indexed for users
            x_coords = row[::2] * scale_factor  # 放大X坐标
            y_coords = row[1::2] * scale_factor  # 放大Y坐标
            ax.plot(x_coords, y_coords, color=color, label=label)

    # 绘制UE位置，不放大
    ax.scatter(user_coords[:, 0], user_coords[:, 1], color='black', marker='o', label='UE位置')

    ax.set_title('无人机轨迹图')
    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    ax.legend()  # 显示图例
    plt.show()

# 示例数据路径和参数
file_path = r'C:\Users\TYGzh\Desktop\M40_0limit_T_P_THETA_Q_local_k12\M40_0limit_T_P_THETA_Q_local_k12\uavposi.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

user_coords = np.array([
    [233.65299231, 483.44192337],
    [374.13188976, 555.99480547],
    # [25.53279877, 172.33036207],
    [80.03512777, 348.27445229],
    [501.9065597, 278.50199273],
    # [521.3707465, 18.84917602],
    # [595.3178245, 503.36783947],
    [409.11222353, 616.04883753],
    [278.34021337, 393.90689167],
    [200.08247499, 307.25697315],
    # [764.23450749, 25.93601218],
    [612.83893037, 9.24855059]
])

episodes_to_plot = [1, 3000, 4000]  # 调整这些值以匹配您文件中的实际轨迹数
colors_to_use = ['yellow', 'blue', 'red']
labels_to_use = ['Episode 1', 'Episode 3000', 'Last Episode']

# 绘制图形
plot_selected_episodes_with_users(data, episodes_to_plot, colors_to_use, labels_to_use, user_coords)

plt.savefig(r'C:\Users\TYGzh\Desktop\M40_0limit_T_P_THETA_Q_local_k12\M40_0limit_T_P_THETA_Q_local_k12\uav_trajectory_plot.png', dpi=300)  # 保存为PNG文件，高分辨率
