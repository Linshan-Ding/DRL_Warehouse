import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np


class WarehouseVisualizer:
    def __init__(self, params):
        self.params = params
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self._init_coordinate_system()

    def _init_coordinate_system(self):
        """初始化坐标系参数"""
        self.total_width = (2 * self.params["aisle_num"] * self.params["shelf_width"] +
                            (2 * self.params["aisle_num"] - 1) * self.params["aisle_width"] +
                            2 * self.params["entrance_width"])

        self.total_height = (self.params["shelf_capacity"] * self.params["shelf_length"] +
                             2 * self.params["aisle_width"])

        self.ax.set_xlim(0, self.total_width)
        self.ax.set_ylim(0, self.total_height)
        self.ax.set_aspect('equal')

    def draw_warehouse_layout(self):
        """绘制完整仓库布局"""
        self._draw_areas()
        self._draw_shelves()
        self._draw_pick_points()
        self._draw_depot()
        self._add_legends()

    def _draw_areas(self):
        """绘制仓库区域划分"""
        colors = ListedColormap(['#FFF9C4', '#E1BEE7', '#C8E6C9'])
        area_width = self.total_width / self.params["area_num"]

        for i in range(self.params["area_num"]):
            rect = patches.Rectangle(
                (i * area_width, 0), area_width, self.total_height,
                linewidth=2, linestyle='--',
                edgecolor='gray', facecolor=colors(i),
                alpha=0.2
            )
            self.ax.add_patch(rect)
            self.ax.text(i * area_width + area_width / 2, self.total_height - 2,
                         f'Zone {i + 1}', ha='center', fontsize=10)

    def _draw_shelves(self):
        """绘制货架单元"""
        shelf_color = '#607D8B'
        for nw in range(1, self.params["aisle_num"] * self.params["area_num"] + 1):
            x_base = self.params["entrance_width"] + (2 * nw - 1) * (
                        self.params["shelf_width"] + self.params["aisle_width"] / 2)

            # 绘制左侧货架
            for nl in range(1, self.params["shelf_capacity"] + 1):
                y_pos = self.params["aisle_width"] + (nl - 0.5) * self.params["shelf_length"]
                self.ax.add_patch(patches.Rectangle(
                    (x_base - self.params["shelf_width"], y_pos),
                    self.params["shelf_width"], self.params["shelf_length"],
                    facecolor=shelf_color, edgecolor='white'
                ))

            # 绘制右侧货架
            for nl in range(1, self.params["shelf_capacity"] + 1):
                y_pos = self.params["aisle_width"] + (nl - 0.5) * self.params["shelf_length"]
                self.ax.add_patch(patches.Rectangle(
                    (x_base, y_pos),
                    self.params["shelf_width"], self.params["shelf_length"],
                    facecolor=shelf_color, edgecolor='white'
                ))

    def _draw_pick_points(self):
        """绘制拣货位"""
        for nw in range(1, self.params["aisle_num"] * self.params["area_num"] + 1):
            x = (self.params["entrance_width"] +
                 (2 * nw - 1) * self.params["shelf_width"] +
                 (2 * nw - 1) / 2 * self.params["aisle_width"])

            for nl in range(1, self.params["shelf_capacity"] + 1):
                y = self.params["aisle_width"] + (2 * nl - 1) / 2 * self.params["shelf_length"]

                # 绘制拣货位标志
                self.ax.scatter(x, y, s=80, c='#FF5722',
                                marker='o', edgecolors='white',
                                label='Pick Point' if nw == 1 else "")

                # 添加拣货位编号
                if nl % 5 == 0:
                    self.ax.text(x + 0.5, y, f'PP-{nw}-{nl}',
                                 fontsize=8, rotation=45)

    def _draw_depot(self):
        """绘制包装台"""
        depot_pos = self.params["depot_position"]
        self.ax.scatter(depot_pos[0], depot_pos[1], s=200,
                        c='#2196F3', marker='s',
                        edgecolors='white', zorder=10,
                        label='Packaging Depot')
        self.ax.text(depot_pos[0], depot_pos[1] - 2, 'Depot',
                     ha='center', fontsize=10, color='#0D47A1')

    def _add_legends(self):
        """添加图例说明"""
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(),
                       loc='upper right', framealpha=0.9)

        # 添加比例尺
        scale_length = 10  # 10 meters
        self.ax.plot([5, 5 + scale_length], [5, 5],
                     color='black', linewidth=2)
        self.ax.text(5 + scale_length / 2, 6, f'{scale_length} m',
                     ha='center', fontsize=8)

if __name__ == "__main__":
    # 参数配置（需与仿真模型参数一致）
    warehouse_params = {
        "shelf_capacity": 20,
        "area_num": 3,
        "aisle_num": 4,
        "shelf_length": 1.2,
        "shelf_width": 2.5,
        "aisle_width": 3.0,
        "entrance_width": 5.0,
        "depot_position": (8.0, 2.0)
    }

    # 生成可视化图形
    visualizer = WarehouseVisualizer(warehouse_params)
    visualizer.draw_warehouse_layout()
    plt.title('智能仓库平面布局示意图', fontsize=14, pad=20)
    plt.xlabel('水平坐标 (米)', fontsize=10)
    plt.ylabel('垂直坐标 (米)', fontsize=10)
    plt.grid(linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('warehouse_layout.png', dpi=300)
    plt.show()