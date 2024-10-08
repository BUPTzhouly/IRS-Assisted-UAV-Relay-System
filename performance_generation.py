import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch  # 确保导入这个类
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# 确保正确安装并使用 openpyxl 库来读取 .xlsx 文件
data = pd.read_excel(r'C:\Users\TYGzh\Desktop\M40_0limit_T_P_THETA_Q_local_k12\M40_0limit_T_P_THETA_Q_local_k12\performance_generation.xlsx', engine='openpyxl')

# 第一列是 x 坐标
x = data.iloc[:, 0]

# 绘制三条线，使用不同的标记和颜色
plt.figure(figsize=(10, 6))
plt.plot(x, data.iloc[:, 1], marker='^', linestyle='--', color='red', label='PPO(红)')  # 使用三角形标记，红色
plt.plot(x, data.iloc[:, 2], marker='s', linestyle='--', color='blue', label='随机分配(蓝)')  # 使用方块标记，蓝色
plt.plot(x, data.iloc[:, 3], marker='o', linestyle='--', color='green', label='平均分配(绿)')  # 使用圆形标记，绿色

# 添加图例
plt.legend()

# 设置坐标轴范围和刻度间隔
plt.xlim([10, 50.001])  # 自动根据x的最小值和最大值设置横坐标范围
plt.ylim([15, 25])  # 纵坐标范围设置为15到25

# 设置刻度间隔
plt.xticks(np.arange(10, 50.001, 10))  # 横坐标每格间隔
plt.yticks(np.arange(15, 25, 1))  # 纵坐标每格间隔

# 添加网格线
plt.grid(True)



# 添加标题和坐标轴标签
# plt.title('Line Plot with Different Markers, Colors, and Grids')
plt.xlabel('IRS反射单元数量')
plt.ylabel('用户速率之和(Mbps)')

# 显示图表
plt.show()
