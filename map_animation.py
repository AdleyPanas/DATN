import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import  Polygon
import threading
import numpy as np
from Astar import *
class RealTimeVisualizer:
    def __init__(self, path_finder, obstacle_coords=None):
        self.path_finder = path_finder
        self.obstacle_coords = obstacle_coords if obstacle_coords else []
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self._first_draw = True
        self.lock = threading.Lock()
        self.updated = False
        self.latest_data = {'obstacle_coords': [],
            'path': [],
            'start': None,
            'end': None}
        # Tạo bảng màu chuyên nghiệp
        self.colors = {
            'background': '#F8F9FA',
            'node': '#CAD2C5',
            'obstacle': '#FF6B6B',
            'path': '#4ECDC4',
            'start': '#000000',
            'end': '#EF476F',
            'current': '#FFD166',
            'text': '#2F3E46',
            'border': '#354F52'
        }
        self.setup_plot()

    def update_data_from_thread(self):
        with self.lock:
            self.latest_data = {
                'obstacle_coords': self.obstacle_coords.copy(),
                'path': self.path_finder.path.copy() if hasattr(self.path_finder, 'path') else [],
                'start': (
                self.path_finder.start_node.x, self.path_finder.start_node.y) if self.path_finder.start_node else None,
                'end': (self.path_finder.end_node.x, self.path_finder.end_node.y) if self.path_finder.end_node else None
            }
            self.updated = True
    def setup_plot(self):
        """Khởi tạo đồ thị"""
        self.ax.clear()
        self.ax.set_xlim(-self.path_finder.grid_width/2 - 200,
                         self.path_finder.grid_width/2 + 200)
        self.ax.set_ylim(-200, self.path_finder.grid_depth + 200)
        self.ax.set_aspect('equal')
        self.ax.grid(True, which='both', linestyle=':', color='gray', alpha=0.7)
        self.ax.set_facecolor(self.colors['background'])
        # Vẽ tam giác bao
        triangle = Polygon(
            [(0, 0), (-self.path_finder.grid_width / 2, self.path_finder.grid_depth),
             (self.path_finder.grid_width / 2, self.path_finder.grid_depth)],
            closed=True, fill=False, edgecolor=self.colors['border'], linewidth=2, linestyle='--', alpha=0.7)
        self.ax.add_patch(triangle)

        # Khởi tạo các phần tử đồ họa
        self.nodes_plot = self.ax.scatter([], [], s=100, c=self.colors['node'],
                                          edgecolors=self.colors['border'],label='Node tự do')
        self.obstacles_plot = self.ax.scatter([], [], s=100, c=self.colors['obstacle'],
                                              edgecolors='red',label='Vùng va chạm')
        self.obs_cam_plot = self.ax.scatter([], [], s=100, c='#000000',
                                              edgecolors='yellow',label='Vật cản')
        self.path_plot, = self.ax.plot([], [], color=self.colors['path'],
                                       linewidth=3, marker='o',markersize=5,label='Đường đi')
        self.current_plot = self.ax.scatter([], [], s=200, c=self.colors['current'],
                                            marker='*',label='Node hiện tại')
        self.start_marker = self.ax.scatter([], [], s=200, c=self.colors['start'],
                                            edgecolors=self.colors['border'],linewidths=4,label='Điểm bắt đầu')
        self.end_marker = self.ax.scatter([], [], s=200, c=self.colors['end'],
                                          edgecolors=self.colors['border'],linewidths=4,label='Điểm kết thúc')
        # Cấu hình tiêu đề và chú thích
        self.ax.set_title(
            'MÔ PHỎNG THUẬT TOÁN A* TRÊN BẢN ĐỒ TAM GIÁC CÂN\n'
            f'Kích thước: {self.path_finder.grid_width}mm (đáy) × {self.path_finder.grid_depth}mm (cao) | '
            f'Kích thước lưới: {self.path_finder.grid_size}mm',fontsize=14, pad=20, color=self.colors['text'])
        self.ax.set_xlabel('Tọa độ X (mm)', fontsize=12, color=self.colors['text'])
        self.ax.set_ylabel('Tọa độ Z (mm)', fontsize=12, color=self.colors['text'])

        # Thêm lưới và chú thích
        self.ax.grid(True, linestyle=':', color='gray', alpha=0.5)
        legend = self.ax.legend(loc='lower right',facecolor='white',edgecolor=self.colors['border'],fontsize=10,
            title='Chú thích:',title_fontsize=11)
        legend.get_frame().set_alpha(0.8)

    def update_plot(self, frame=0):
        """Cập nhật đồ thị cho mỗi frame"""
        with self.lock:
            if not self.updated:
                return
            self.updated = False

            data = self.latest_data
        # Cập nhật trạng thái các node
        nodes = [(n.x, n.y) for n in self.path_finder.grid if not n.obstacle]
        obstacles = [(n.x, n.y) for n in self.path_finder.grid if n.obstacle]
        self.nodes_plot.set_offsets(nodes)
        if obstacles:
            self.obstacles_plot.set_offsets(obstacles)
        else:
            self.obstacles_plot.set_offsets(np.empty((0, 2)))
        self.obs_cam_plot.set_offsets(data['obstacle_coords'] if data['obstacle_coords'] else np.empty((0, 2)))
        if data['path']:
            xs, ys = zip(*data['path'])
            self.path_plot.set_data(xs, ys)
        else:
            self.path_plot.set_data([], [])

        if data['start']:
            self.start_marker.set_offsets([data['start']])
        if data['end']:
            self.end_marker.set_offsets([data['end']])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def start_animation(self):
        """Bắt đầu hiển thị thời gian thực"""

        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100,cache_frame_data=False)
        plt.ion()
        plt.show()

