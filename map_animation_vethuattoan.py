import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
import matplotlib.colors as mcolors
import numpy as np
from Astar_vethuattoan import *
class RealTimeVisualizer:
    def __init__(self, path_finder, obstacle_coords=None):
        self.path_finder = path_finder
        self.obstacle_coords = obstacle_coords if obstacle_coords else []
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self._first_draw = True
        # Tạo bảng màu chuyên nghiệp
        self.colors = {
            'background': '#F8F9FA',
            'node': '#CAD2C5',
            'obstacle': '#FF6B6B',
            'path': '#22f9e9',
            'start': '#000000',
            'end': '#EF476F',
            'current': '#FFD166',
            'text': '#2F3E46',
            'border': '#86adb1'
        }
        self.open_set_plot = self.ax.scatter([], [], s=100, c='lime', marker='o', label='Open set')
        self.closed_set_plot = self.ax.scatter([], [], s=100, c='gray', marker='x', label='Closed set')
        self.setup_plot()
    def setup_plot(self):
        """Khởi tạo đồ thị"""
        self.ax.clear()
        self.ax.set_xlim(-self.path_finder.grid_width/2 - 20,
                         self.path_finder.grid_width/2 + 20)
        self.ax.set_ylim(-20, self.path_finder.grid_depth + 20)
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
        self.nodes_plot = self.ax.scatter([], [], s=100, c=self.colors['node'],marker='s',
                                          edgecolors=self.colors['border'],label='Node tự do')
        self.obstacles_plot = self.ax.scatter([], [], s=100, c=self.colors['obstacle'],marker='s',
                                              edgecolors='red',label='Vùng va chạm')
        self.obs_cam_plot = self.ax.scatter([], [], s=100, c='#000000',marker='s',
                                              edgecolors='yellow',label='Vật cản')
        self.path_plot, = self.ax.plot([], [], color=self.colors['path'],
                                       linewidth=2, marker='o',markersize=3,label='Đường đi')
        self.current_plot = self.ax.scatter([], [], s=200, c=self.colors['current'],
                                            marker='*',label='Node hiện tại')
        self.start_marker = self.ax.scatter([], [], s=200, c=self.colors['start'],
                                            edgecolors=self.colors['border'],linewidths=4,label='Điểm bắt đầu')
        self.end_marker = self.ax.scatter([], [], s=200, c=self.colors['end'],
                                          edgecolors=self.colors['border'],linewidths=4,label='Điểm kết thúc')
        self.open_set_plot = self.ax.scatter([], [], s=60, c='lime', marker='o', label='Open set')
        self.closed_set_plot = self.ax.scatter([], [], s=60, c='gray', marker='x', label='Closed set')
        # Cấu hình tiêu đề và chú thích
        self.ax.set_title(
            'MÔ PHỎNG THUẬT TOÁN A* TRÊN BẢN ĐỒ TAM GIÁC CÂN\n'
            f'Kích thước: {self.path_finder.grid_width}mm (đáy) × {self.path_finder.grid_depth}mm (cao) | '
            f'Kích thước lưới: {self.path_finder.grid_size}mm',fontsize=14, pad=20, color=self.colors['text'])
        self.ax.set_xlabel('Tọa độ X (mm)', fontsize=12, color=self.colors['text'])
        self.ax.set_ylabel('Tọa độ Z (mm)', fontsize=12, color=self.colors['text'])
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles, labels, loc='lower right',
                       facecolor='white', edgecolor=self.colors['border'],
                       fontsize=10, title='Chú thích:', title_fontsize=11).get_frame().set_alpha(0.8)
        # Thêm lưới và chú thích
        self.ax.grid(True, linestyle=':', color='gray', alpha=0.5)
        legend = self.ax.legend(
            loc='lower right',
            facecolor='white',
            edgecolor=self.colors['border'],
            fontsize=10,
            title='Chú thích:',
            title_fontsize=11
        )
        legend.get_frame().set_alpha(0.8)

    def update_plot(self, frame=0):
        """Cập nhật đồ thị cho mỗi frame"""
        # Cập nhật trạng thái các node
        nodes = [(n.x, n.y) for n in self.path_finder.grid if not n.obstacle]
        obstacles = [(n.x, n.y) for n in self.path_finder.grid if n.obstacle]
        obs_cams = self.obstacle_coords
        if nodes:
            self.nodes_plot.set_offsets(nodes)
        if obstacles:
            self.obstacles_plot.set_offsets(obstacles)
        else:
            self.obstacles_plot.set_offsets(np.empty((0, 2)))
        if obs_cams:
            self.obs_cam_plot.set_offsets(obs_cams)
        else:
            self.obs_cam_plot.set_offsets(np.empty((0, 2)))
        # Cập nhật điểm bắt đầu/kết thúc
        if self.path_finder.start_node:
            self.start_marker.set_offsets([(self.path_finder.start_node.x,
                                            self.path_finder.start_node.y)])
        if self.path_finder.end_node:
            self.end_marker.set_offsets([(self.path_finder.end_node.x,
                                          self.path_finder.end_node.y)])

        if hasattr(self.path_finder, 'path') and len(self.path_finder.path) > 0:
            # Vẽ toàn bộ đường đi ngay lập tức
            self.path_plot.set_data(
                [p[0] for p in self.path_finder.path],
                [p[1] for p in self.path_finder.path]
            )
        else:
            self.path_plot.set_data([], [])
        for text in getattr(self, 'f_cost_texts', []):
            text.remove()  # Xóa các text cũ
        self.f_cost_texts = []

        for node in self.path_finder.grid:
            if not node.obstacle and node.f_cost != float('inf'):
                text = self.ax.text(node.x, node.y,  # Đặt hơi cao để không đè lên node
                                    f"{int(node.f_cost)}",
                                    fontsize=5, color='blue', ha='center')
                self.f_cost_texts.append(text)
        if self._first_draw:
            plt.ion()  # Bật chế độ tương tác
            #plt.show()
            self._first_draw = False
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def animate_path_finding(self):
        """Chạy animation từng bước mở rộng của A*"""
        self.step_index = 0

        def frame_update(frame):
            if self.step_index < len(self.path_finder.search_steps):
                step_data = self.path_finder.search_steps[self.step_index]
                current_node = step_data.get('current')
                open_set = step_data.get('open_set', [])
                closed_set = step_data.get('closed_set', [])

                # Cập nhật node hiện tại
                if current_node:
                    self.current_plot.set_offsets(np.array([[current_node.x, current_node.y]]))
                else:
                    self.current_plot.set_offsets(np.empty((0, 2)))

                # Cập nhật open set
                open_coords = [(node.x, node.y) for node in open_set]
                self.open_set_plot.set_offsets(open_coords if open_coords else np.empty((0, 2)))

                # Cập nhật closed set
                self.closed_set_plot.set_offsets(closed_set if closed_set else np.empty((0, 2)))

                self.update_plot(0)
                self.step_index += 1
            else:
                self.current_plot.set_offsets(np.empty((0, 2)))
                self.open_set_plot.set_offsets(np.empty((0, 2)))
                self.closed_set_plot.set_offsets(np.empty((0, 2)))
                self.update_plot(0)
                ani.event_source.stop()


        ani = FuncAnimation(self.fig, frame_update, interval=100, frames=len(self.path_finder.search_steps), cache_frame_data=False)
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    pf = PathFinder(grid_width=380, grid_depth=300, grid_size=10)
    visualizer = RealTimeVisualizer(pf)
    visualizer.obstacle_coords = [(0,90)]
    pf.mark_obstacles_from_camera(visualizer.obstacle_coords)
    pf.a_star(start_coord=(0, 0), end_coord=(0,290))
    #visualizer.update_plot(0)
    visualizer.animate_path_finding()
    plt.ioff()  # Tắt chế độ tương tác
    plt.show()