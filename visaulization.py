import pygame
import os
from pygame.locals import *
from MapData import maps
from Colors import *
from datetime import datetime
from Obstacle import Obstacle

def draw_target(window, target):
    """Vẽ điểm mục tiêu (đỏ)."""
    pygame.draw.circle(window, RED, target, 8, 0)

def draw_start(window, start):
    """Vẽ điểm bắt đầu (xanh lá)."""
    pygame.draw.circle(window, GREEN, start, 6, 0)

def draw_path(window, path, color):
    """Vẽ đường đi với màu sắc chỉ định."""
    for i in range(1, len(path)):
        if path[i-1][0] == path[i][0] or path[i-1][1] == path[i][1]:
            pygame.draw.line(window, color, path[i-1], path[i], 4)
        else:
            pygame.draw.line(window, color, path[i-1], path[i], 6)

def draw_grid(window, cell_size, env_size, env_padding):
    """Vẽ lưới nền."""
    for i in range(1, int(env_size / cell_size)):
        pygame.draw.line(window, BLACK, (env_padding + i * cell_size, env_padding),
                         (env_padding + i * cell_size, env_padding + env_size), 1)
        pygame.draw.line(window, BLACK, (env_padding, env_padding + i * cell_size),
                         (env_padding + env_size, env_padding + i * cell_size), 1)

def select_map():
    """Chọn bản đồ từ người dùng."""
    print("Available maps:")
    for i, map_name in enumerate(maps.keys()):
        print(f"{i+1}. {map_name}")
    
    while True:
        try:
            choice = int(input(f"Select map (1-{len(maps)}): "))
            if 1 <= choice <= len(maps):
                return list(maps.keys())[choice-1]
            print(f"Please enter a number between 1 and {len(maps)}")
        except ValueError:
            print("Please enter a valid number")

def find_latest_file(map_name):
    """Tìm file mới nhất trong thư mục results/{map_name}/ bắt đầu bằng robot_paths_."""
    results_dir = f"results/{map_name}"
    if not os.path.exists(results_dir):
        return None
    
    files = [f for f in os.listdir(results_dir) if f.startswith("robot_paths_") and f.endswith(".txt")]
    if not files:
        return None
    
    def get_timestamp(file_name):
        timestamp_str = file_name.split('_')[2] + '_' + file_name.split('_')[3].split('.')[0]
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    
    latest_file = max(files, key=get_timestamp)
    return os.path.join(results_dir, latest_file)

def read_last_path(file_path, cell_size, env_padding):
    """Đọc đường đi cuối cùng từ file (loại bỏ điểm reset) và chuyển đổi sang tọa độ pixel."""
    episodes = []
    current_episode = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Episode"):
                if current_episode:
                    if len(current_episode) > 1:
                        episodes.append(current_episode[:-1])  # Loại bỏ điểm reset
                    current_episode = []
            elif line:
                try:
                    if line.endswith(','):
                        line = line[:-1]
                    coords_list = eval(f"[{line}]")
                    current_episode.extend(coords_list)
                except SyntaxError:
                    print(f"Error parsing line: {line}")
                    continue
    
    if current_episode and len(current_episode) > 1:
        episodes.append(current_episode[:-1])
    
    if not episodes:
        return []
    
    # Lấy episode cuối cùng
    last_path = episodes[-1]
    
    # Chuyển đổi tọa độ lưới sang tọa độ pixel
    pixel_path = [(env_padding + (x + 0.5) * cell_size, env_padding + (y + 0.5) * cell_size) for x, y in last_path]
    return pixel_path

if __name__ == "__main__":
    # Thiết lập thông số môi trường
    cell_size = 16
    env_size = 512
    env_padding = int(env_size * 0.06)
    
    # Chọn bản đồ
    selected_map = select_map()
    
    # Lấy thông tin bản đồ
    start_grid = maps[selected_map]["Start"]
    goal_grid = maps[selected_map]["Goal"]
    start = (env_padding + (start_grid[0] + 0.5) * cell_size, env_padding + (start_grid[1] + 0.5) * cell_size)
    goal = (env_padding + (goal_grid[0] + 0.5) * cell_size, env_padding + (goal_grid[1] + 0.5) * cell_size)
    
    # Tìm file đường đi mới nhất
    latest_file = find_latest_file(selected_map)
    if not latest_file:
        print(f"No path files found for map {selected_map} in results/{selected_map}/")
        exit()
    
    # Đọc đường đi cuối cùng
    path = read_last_path(latest_file, cell_size, env_padding)
    if not path:
        print("No valid paths found in the file.")
        exit()
    
    # Thiết lập Pygame
    pygame.init()
    screen = pygame.display.set_mode((env_size + 2 * env_padding, env_size + 3 * env_padding))
    pygame.display.set_caption("Path Visualization - Last Episode")
    
    # Khởi tạo chướng ngại vật
    obstacles_list = []
    if selected_map in maps:
        for obs_data in maps[selected_map]["Obstacles"]:
            obstacle = Obstacle(
                x=obs_data["x"],
                y=obs_data["y"],
                width=obs_data["width"],
                height=obs_data["height"],
                static=obs_data["static"],
                velocity=obs_data.get("velocity"),
                x_bound=obs_data.get("x_bound"),
                y_bound=obs_data.get("y_bound"),
                path=obs_data.get("path"),
                angle=obs_data.get("angle", 0)
            )
            obstacles_list.append(obstacle)
    
    running = True
    pause = False
    
    while running:
        screen.fill(WHITE)
        
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == MOUSEBUTTONDOWN:
                if button_pause.collidepoint(event.pos):
                    pause = not pause
        
        # Vẽ nút Pause
        button_pause = pygame.draw.rect(screen, BLACK, (env_padding + int(env_size * 0.4), env_padding * 2 + env_size,
                                                       int(env_size * 0.2), int(env_padding * 0.4)), 4)
        button_pause_text = pygame.font.SysFont("arial", env_padding // 2).render("Pause", True, BLACK)
        screen.blit(button_pause_text, button_pause_text.get_rect(center=button_pause.center))
        
        if not pause:
            # Di chuyển chướng ngại vật
            for obstacle in obstacles_list:
                obstacle.move()
        
        # Vẽ lưới
        draw_grid(screen, cell_size, env_size, env_padding)
        
        # Vẽ chướng ngại vật
        for obstacle in obstacles_list:
            obstacle.draw(screen)
        
        # Vẽ đường đi cuối cùng
        if path:
            draw_path(screen, path, RED)
        
        # Vẽ điểm bắt đầu và mục tiêu
        draw_start(screen, start)
        draw_target(screen, goal)
        
        # Vẽ viền
        pygame.draw.rect(screen, BLACK, (env_padding, env_padding, env_size, env_size), 3)
        
        pygame.display.update()
    
    pygame.quit()