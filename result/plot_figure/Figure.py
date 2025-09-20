import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 创建多图PDF
from matplotlib.backends.backend_pdf import PdfPages

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
    [1, 0, 1]
])

# 未拣货商品数量矩阵
pending_items_matrix = np.array([
    [5, 6, 3],
    [2, 7, 0],
    [8, 0, 4]
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
    return plt.gcf()

# 创建纵向合并的热力图
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

sns.heatmap(queue_count_matrix, annot=True, fmt="d", cmap="Blues", cbar=True, ax=axes[0])
axes[0].set_title("Queue Count Matrix")
axes[0].set_xlabel('Width (N_w)')
axes[0].set_ylabel('Length (N_l)')

sns.heatmap(picker_presence_matrix, annot=True, fmt="d", cmap="Greens", cbar=True, ax=axes[1])
axes[1].set_title("Picker Presence Matrix")
axes[1].set_xlabel('Width (N_w)')
axes[1].set_ylabel('Length (N_l)')

sns.heatmap(pending_items_matrix, annot=True, fmt="d", cmap="Oranges", cbar=True, ax=axes[2])
axes[2].set_title("Pending Items Matrix")
axes[2].set_xlabel('Width (N_w)')
axes[2].set_ylabel('Length (N_l)')

plt.tight_layout()

# 保存为单页PDF和SVG
with PdfPages('combined_heatmaps.pdf') as pdf:
    pdf.savefig(fig)
    plt.savefig('combined_heatmaps.svg')
    plt.close(fig)
