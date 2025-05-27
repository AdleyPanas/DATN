import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle
from Astar import *


def visualize_map(path_finder, path=None):
    """Hiển thị bản đồ với các cải tiến về hình ảnh"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Thiết lập tỷ lệ và lưới
    ax.set_xlim(-path_finder.grid_width/2-10, path_finder.grid_width/2 + 10)
    ax.set_ylim(-10, path_finder.grid_depth + 10)
    ax.set_aspect('equal')
    ax.grid(True, which='both', linestyle=':', color='gray', alpha=0.3)

    # Vẽ các node với cải tiến
    for node in path_finder.grid:
        facecolor = '#FF6B6B' if node.obstacle else '#CAD2C5'
        edgecolor = '#354F52' if not node.obstacle else '#FF0000'

        rect = Rectangle(
            (node.x, node.y),
            path_finder.grid_size,
            path_finder.grid_size,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.7,
            alpha=0.9 if node.obstacle else 0.6
        )
        ax.add_patch(rect)

        # Hiển thị tọa độ nếu ô đủ lớn
        if path_finder.grid_size >= 1500:
            ax.text(
                node.x + path_finder.grid_size / 2,
                node.y + path_finder.grid_size / 2,
                f'({node.x},{node.y})',
                ha='center', va='center',
                fontsize=6,
                color='#2F3E46' if not node.obstacle else 'white'
            )

    # Vẽ đường đi với cải tiến
    if path:
        x_coords = [x + path_finder.grid_size / 2 for x, y in path]
        y_coords = [y + path_finder.grid_size / 2 for x, y in path]
        ax.plot(
            x_coords, y_coords,
            color='#4ECDC4',
            linewidth=3,
            linestyle='-',
            marker='o',
            markersize=4,
            markerfacecolor='#292F36',
            markeredgecolor='none',
            alpha=0.9,
            label='Đường đi tối ưu'
        )

    # Vẽ điểm bắt đầu và kết thúc với cải tiến
    if path_finder.start_node:
        start_circle = Circle(
            (path_finder.start_node.x + path_finder.grid_size / 2,
             path_finder.start_node.y + path_finder.grid_size / 2),
            radius=path_finder.grid_size / 3,
            facecolor='#06D6A0',
            edgecolor='#073B4C',
            linewidth=1.5,
            alpha=1,
            label='Điểm bắt đầu'
        )
        ax.add_patch(start_circle)

    if path_finder.end_node:
        end_circle = Circle(
            (path_finder.end_node.x + path_finder.grid_size / 2,
             path_finder.end_node.y + path_finder.grid_size / 2),
            radius=path_finder.grid_size / 3,
            facecolor='#EF476F',
            edgecolor='#073B4C',
            linewidth=1.5,
            alpha=1,
            label='Điểm kết thúc'
        )
        ax.add_patch(end_circle)

    # Cải tiến phần chú thích và tiêu đề
    ax.set_title(
        'BẢN ĐỒ MÔ PHỎNG THUẬT TOÁN A*\nKích thước lưới: {}mm x {}mm'.format(
            path_finder.grid_width, path_finder.grid_depth),
        fontsize=14,
        pad=20,
        color='#2F3E46'
    )

    ax.set_xlabel('Chiều ngang (mm)', fontsize=10, labelpad=10)
    ax.set_ylabel('Chiều sâu (mm)', fontsize=10, labelpad=10)

    # Thêm chú thích được cải tiến
    legend = ax.legend(
        loc='upper right',
        framealpha=1,
        edgecolor='#354F52',
        facecolor='#84A98C',
        fontsize=9
    )

    # Tô màu nền cho đồ thị
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    # Thêm grid background
    ax.grid(True, which='major', color='#DEE2E6', linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', color='#E9ECEF', linestyle=':', linewidth=0.3)

    plt.tight_layout()
    plt.show()


# Sử dụng trong ví dụ trước:
if __name__ == "__main__":
    pf = PathFinder(grid_width=3800, grid_depth=3000, grid_size=100)
    obstacle_coords = [(-600,1000),(-550,1000),(-500,1000),(-450,1000),(-400,1000),(-350,1000),
                       (-300,1000),(-250,1000),(-200,1000),(-150,1000),(-100,1000),(-50,1000),
                       (0,1000),(50,1000),(100,1000),(-300,1900), (400,1900)]
    #obstacle_coords=[]
    pf.mark_obstacles_from_camera(obstacle_coords)

    pf.a_star(start_coord=(0, 0), end_coord=(-1000, 2900))

    # Hiển thị bản đồ
    visualize_map(pf, pf.path)