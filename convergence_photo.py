import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# Read the CSV file
file_path = r'C:\Users\TYGzh\Desktop\M40_0limit_T_P_THETA_Q_local_k12\M40_0limit_T_P_THETA_Q_local_k12\convergence.csv'
# Change this to your actual file path
data = pd.read_csv(file_path)

# Assuming the first column is the x-axis and the next two are the y-values for the curves
x = data.iloc[:, 0]
y1 = data.iloc[:, 1]
y2 = data.iloc[:, 2]

# Plotting the curves
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='C_LR=0.0002', color='blue')  # First curve in blue
plt.plot(x, y2, label='C_LR=0.0001', color='red', linestyle='--')  # Second curve in red, dashed for distinction

# Setting titles and labels
plt.title('算法收敛图')  # "Algorithm Convergence Chart"
plt.xlabel('Episode')
plt.ylabel('Moving reward')

# Adding a legend and grid
plt.legend()
plt.grid(True)

# Adjust layout
plt.tight_layout()

# Save the plot to a file
plot_file_path = r'C:\Users\TYGzh\Desktop\M40_0limit_T_P_THETA_Q_local_k12\M40_0limit_T_P_THETA_Q_local_k12\updated_curves_plot.png'
# Change this to your desired save location
plt.savefig(plot_file_path)

# Note: If you encounter warnings about missing glyphs when running this code in an environment without
# support for Chinese characters in matplotlib, consider specifying a font that includes these glyphs.
# Example: plt.rcParams['font.family'] = 'Your_Font_That_Supports_Chinese'
