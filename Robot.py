import pygame
import numpy as np
from Colors import *

class Robot:
    def __init__(self, x, y, cell_size, controller, vision, radius=8, env_padding=30):
        self.cell_size = cell_size
        self.env_padding = env_padding
        self.grid_x = int((x - env_padding) // cell_size)
        self.grid_y = int((y - env_padding) // cell_size)
        self.x = self.env_padding + (self.grid_x + 0.5) * self.cell_size
        self.y = self.env_padding + (self.grid_y + 0.5) * self.cell_size
        self.controller = controller
        self.vision = vision
        self.radius = radius
        self.direction = (0, 0)

    def move(self, obstacles, grid_width, grid_height):
        self.direction = self.controller.make_decision(self, obstacles)
        dx, dy = self.direction
        new_grid_x = self.grid_x + dx
        new_grid_y = self.grid_y + dy

        # Kiểm tra xem vị trí mới có nằm trong lưới hay không
        if not (0 <= new_grid_x < grid_width and 0 <= new_grid_y < grid_height):
            # Nếu vượt biên, coi như va chạm với chướng ngại vật
            return True  # Trả về True để báo hiệu va chạm

        # Lưu vị trí cũ để khôi phục nếu va chạm
        old_grid_x, old_grid_y = self.grid_x, self.grid_y
        old_x, old_y = self.x, self.y

        # Cập nhật tạm thời vị trí
        self.grid_x = new_grid_x
        self.grid_y = new_grid_y
        self.x = self.env_padding + (self.grid_x + 0.5) * self.cell_size
        self.y = self.env_padding + (self.grid_y + 0.5) * self.cell_size

        # Kiểm tra va chạm với chướng ngại vật
        if self.check_collision(obstacles):
            # Nếu va chạm, khôi phục vị trí cũ
            # self.grid_x, self.grid_y = old_grid_x, old_grid_y
            # self.x, self.y = old_x, old_y
            return True  # Báo hiệu va chạm

        return False  # Không có va chạm

    def draw(self, window):
        pygame.draw.circle(window, RED, (int(self.x), int(self.y)), self.vision, 1)
        pygame.draw.circle(window, RED, (int(self.x), int(self.y)), self.radius)

    def check_collision(self, obstacles):
        for obs in obstacles:
            x1, x2, y1, y2 = obs.get_bounding_box()
            closest_x = max(x1, min(self.x, x2))
            closest_y = max(y1, min(self.y, y2))
            distance = ((closest_x - self.x)**2 + (closest_y - self.y)**2) ** 0.5
            if distance < self.radius:
                return True
        return False

    def get_state(self, obstacles, grid_width, grid_height, goal):
        # Tạo ma trận trạng thái 5x5, khởi tạo tất cả là 0 (ô trống)
        state = np.zeros((5, 5), dtype=int)

        # Duyệt các ô trong phạm vi 5x5 xung quanh robot
        for i in range(-2, 3):
            for j in range(-2, 3):
                grid_x = self.grid_x + i
                grid_y = self.grid_y + j
                state_idx_x = i + 2
                state_idx_y = j + 2
                if not (0 <= grid_x < grid_width and 0 <= grid_y < grid_height):
                    state[state_idx_x, state_idx_y] = 1
                    continue

        # Duyệt các chướng ngại vật và gán giá trị 1 cho các ô mà chúng chiếm
        for obs in obstacles:
            x1, x2, y1, y2 = obs.get_bounding_box()
            grid_x1 = int((x1 - self.env_padding) // self.cell_size)
            grid_x2 = int((x2 - self.env_padding) // self.cell_size)
            grid_y1 = int((y1 - self.env_padding) // self.cell_size)
            grid_y2 = int((y2 - self.env_padding) // self.cell_size)
            for gx in range(max(0, grid_x1), min(grid_width, grid_x2 + 1)):
                for gy in range(max(0, grid_y1), min(grid_height, grid_y2 + 1)):
                    state_idx_x = gx - self.grid_x + 2
                    state_idx_y = gy - self.grid_y + 2
                    if 0 <= state_idx_x < 5 and 0 <= state_idx_y < 5:
                        state[state_idx_x, state_idx_y] = 1

        # Gán giá trị 2 cho vị trí robot (trung tâm ma trận)
        state[2, 2] = 2

        # Tính chỉ số ô lưới của đích
        goal_grid_x = int((goal[0] - self.env_padding) // self.cell_size)
        goal_grid_y = int((goal[1] - self.env_padding) // self.cell_size)

        # Tính khoảng cách Euclidean đến đích (tính bằng số ô)
        distance_to_goal = ((self.grid_x - goal_grid_x)**2 + (self.grid_y - goal_grid_y)**2)**0.5

        return state, distance_to_goal