import math
from queue import PriorityQueue
from collections import defaultdict
import numpy as np
import time
class Node:
    def __init__(self, x, y, obstacle=False):
        self.x = x  # Tọa độ x (cm)
        self.y = y  # Tọa độ y (cm)
        self.g_cost = float('inf')  # Khoảng cách từ start, gán bằng vô cực
        self.h_cost = 0  # Khoảng cách heuristic đến end
        self.f_cost = float('inf')  # Tổng g_cost + h_cost
        self.parent = None  # Node cha
        self.obstacle = obstacle  # Mặc định không là vật cản
    def __lt__(self, other):
        # So sánh dựa trên f_cost, nếu bằng nhau so h_cost, nếu bằng nhau so x (nằm bên phải)
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        if self.h_cost != other.h_cost:
            return self.h_cost < other.h_cost
        return self.x > other.x

class PathFinder:
    def __init__(self, grid_width=3800, grid_depth=3000, grid_size=100):
        self.grid_width = grid_width  # Chiều ngang bản đồ (mm)
        self.grid_depth = grid_depth  # Chiều sâu bản đồ (mm)
        self.grid_size = grid_size  # Kích thước mỗi ô (mm)
        self.grid = self.create_grid()  # Tạo grid ban đầu không có vật cản
        self.open_list = PriorityQueue()
        self.closed_list = set()
        self.path = []
        self.expanded_coords = []
        self.start_node = None
        self.end_node = None
        self.current_node = None
    def create_grid(self):
        """Tạo grid gồm các node có thể đi qua, không có vật cản."""
        grid = []

        for y in range(0, self.grid_depth+1, self.grid_size):
            for x in range(-self.grid_width//2, self.grid_width//2+1, self.grid_size):
                # Kiểm tra node có nằm trong tam giác cân không
                if self.is_valid_node(x, y):
                    grid.append(Node(x, y, obstacle=False))
        return grid

    def is_valid_node(self, x, y):
        """Kiểm tra node có nằm trong tam giác cân không."""
        if y < 0 or y > self.grid_depth:
            return False
        valid_width = self.grid_width * (y / self.grid_depth)
        return abs(x) <= valid_width / 2

    def mark_obstacles_from_camera(self, obstacle_coords):
        """Đánh dấu vật cản dựa trên tọa độ từ camera depth."""
        self.expanded_coords.clear()
        for node in self.grid:
            node.obstacle = False
        for (x, y) in obstacle_coords:
            for xi in range(x - 20, x + 21, self.grid_size):
                for yi in range(y - 20, y + 21, self.grid_size):
                    self.expanded_coords.append((xi, yi))
        for (x, y) in self.expanded_coords:
            # Tìm node tương ứng trong grid và đánh dấu là vật cản
            node = next((n for n in self.grid if n.x == x and n.y == y), None)
            if node:
                node.obstacle = True

    def heuristic(self, node_a, node_b):
        """Tính khoảng cách Euclidean heuristic."""
        return math.sqrt((node_a.x - node_b.x) ** 2 + (node_a.y - node_b.y) ** 2)

    def get_neighbors(self, node):
        """Trả về 8 node lân cận hợp lệ (không phải vật cản)."""
        directions = [(-self.grid_size, 0), (self.grid_size, 0), (0, -self.grid_size), (0, self.grid_size),
                      (-self.grid_size, -self.grid_size), (-self.grid_size, self.grid_size),
                      (self.grid_size, -self.grid_size), (self.grid_size, self.grid_size)]
        neighbors = []
        for dx, dy in directions:
            x, y = node.x + dx, node.y + dy
            # Tìm node trong grid và kiểm tra không phải vật cản
            neighbor = next((n for n in self.grid if n.x == x and n.y == y and not n.obstacle), None)
            if neighbor:
                neighbors.append(neighbor)
        return neighbors

    def a_star(self, start_coord, end_coord):
        """Thuật toán A* với đầu vào là tọa độ (x, y)."""
        # Reset dữ liệu trước mỗi lần tìm đường
        self.open_list = PriorityQueue()
        self.closed_list = set()
        self.path = []
        self.start_node = None
        self.end_node = None
        self.current_node = None
        self.search_steps = []
        for node in self.grid:
            node.g_cost = float('inf')
            node.h_cost = 0
            node.f_cost = float('inf')
            node.parent = None

        # Tìm start_node và end_node trong
        print('điểm bắt đầu: ',start_coord)
        print('điểm kết thúc: ', end_coord)
        start_time = time.time()
        self.start_node = next((n for n in self.grid if n.x == start_coord[0] and n.y == start_coord[1]), None)
        self.end_node = next((n for n in self.grid if n.x == end_coord[0] and n.y == end_coord[1]), None)

        if not self.start_node or not self.end_node:
            print("Start hoặc end không hợp lệ!")
            return False

        # Khởi tạo thuật toán
        self.start_node.g_cost = 0
        self.start_node.h_cost = self.heuristic(self.start_node, self.end_node)
        self.start_node.f_cost = self.start_node.h_cost
        self.open_list.put(self.start_node)

        while not self.open_list.empty():
            self.current_node = self.open_list.get()
            print("chọn nút hiện tại: ",self.current_node.x,self.current_node.y,self.current_node.f_cost,self.current_node.h_cost)
            self.closed_list.add((self.current_node.x, self.current_node.y))
            self.search_steps.append({
                'current': self.current_node,
                'open_set': list(self.open_list.queue),  # Danh sách open hiện tại
                'closed_set': list(self.closed_list)  # Dạng (x, y)
            })
            if self.current_node.x == self.end_node.x and self.current_node.y == self.end_node.y:
                print("Tìm thấy đường đi!")
                current_time = time.time()
                elapsed = current_time - start_time
                print('Thời gian tính toán: ',elapsed*1000,'ms')
                self.reconstruct_path(self.current_node)
                return True

            neighbors = self.get_neighbors(self.current_node)
            for neighbor in neighbors:
                if (neighbor.x, neighbor.y) in self.closed_list:
                    continue

                tentative_g_cost = self.current_node.g_cost + self.heuristic(self.current_node, neighbor)
                if tentative_g_cost < neighbor.g_cost:
                    neighbor.parent = self.current_node
                    neighbor.g_cost = tentative_g_cost
                    neighbor.h_cost = self.heuristic(neighbor, self.end_node)
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                    # Nếu neighbor chưa có trong Open, thêm vào
                    if not any(n == neighbor for n in self.open_list.queue):
                        print('hàng xóm: ','(',neighbor.x,neighbor.y,neighbor.f_cost,neighbor.h_cost,')')
                        self.open_list.put(neighbor)

        print("Không tìm thấy đường đi!")
        return False

    def reconstruct_path(self, end_node):
        """Truy xuất đường đi từ end về start."""
        node = end_node
        while node:
            self.path.append((node.x, node.y))
            node = node.parent
        self.path.reverse()
        print("Đường đi:", self.path)

