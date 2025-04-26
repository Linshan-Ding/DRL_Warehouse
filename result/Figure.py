import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 设置全局字体大小为14
plt.rcParams.update({'font.size': 14})

# 初始化矩阵大小
N_w, N_l = 3, 3

# 机器人当前排队数量矩阵
queue_count_matrix = np.array([
    [2, 0, 1],
    [1, 3, 0],
    [0, 0, 2]
])

# 是否有拣货员矩阵
picker_presence_matrix = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# 未拣货商品数量矩阵
pending_items_matrix = np.array([
    [5, 0, 3],
    [2, 7, 0],
    [0, 0, 4]
])

# 打印矩阵
print("Queue Count Matrix:")
print(queue_count_matrix)

print("\nPicker Presence Matrix:")
print(picker_presence_matrix)

print("\nPending Items Matrix:")
print(pending_items_matrix)


# 生成热力图函数
def plot_heatmap(matrix, title, cmap="viridis"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap, cbar=True)
    plt.title(title)
    plt.xlabel('Width (N_w)')
    plt.ylabel('Length (N_l)')
    plt.tight_layout()
    # 保存图片为svg
    plt.savefig("figure{}.pdf".format(title))
    plt.show()


# 绘制热力图
plot_heatmap(queue_count_matrix, "Queue Count Matrix", cmap="Blues")
plot_heatmap(picker_presence_matrix, "Picker Presence Matrix", cmap="Greens")
plot_heatmap(pending_items_matrix, "Pending Items Matrix", cmap="Oranges")
