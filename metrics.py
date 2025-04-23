import os
import numpy as np
import pandas as pd
from datetime import datetime
from MapData import maps

def calculateDistanceToGoal(start, goal):
    """Tính khoảng cách lý tưởng từ start đến goal (diagonal distance)."""
    x, y = goal[0] - start[0], goal[1] - start[1]
    return np.abs(x - y) + np.sqrt(2) * min(np.abs(x), np.abs(y))

def calculate_path_length(path):
    """Tính độ dài đường đi dựa trên khoảng cách Euclidean."""
    length = 0
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return length

def get_sum_turning_angle(path):
    """Tính tổng góc quay."""
    total_angle = 0
    for i in range(1, len(path) - 1):
        vector1 = [path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]]
        vector2 = [path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]]
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            continue
        angle = np.arccos(np.dot(vector1, vector2) / (norm1 * norm2 + 1e-6))
        total_angle += np.abs(angle)
    return total_angle

def get_average_turning_angle(path):
    """Tính góc quay trung bình."""
    total_angle = get_sum_turning_angle(path)
    num_turns = max(1, len(path) - 2)
    return total_angle / num_turns

def select_map():
    """Chọn bản đồ từ người dùng, tham khảo cách input của main.py."""
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

def read_paths_from_file(file_path):
    """Đọc các đường đi từ file và trả về danh sách các episode, loại bỏ điểm reset cuối."""
    episodes = []
    current_episode = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Episode"):
                if current_episode:
                    # Loại bỏ điểm reset cuối (thường là [1,16])
                    if len(current_episode) > 1:
                        episodes.append(current_episode[:-1])
                    current_episode = []
            elif line:
                # Xử lý dòng chứa nhiều tọa độ, ví dụ: [1,16], [2,15], ..., [30,4], [1,16],
                try:
                    # Loại bỏ dấu phẩy cuối nếu có
                    if line.endswith(','):
                        line = line[:-1]
                    # Chuyển chuỗi thành danh sách tọa độ
                    coords_list = eval(f"[{line}]")
                    current_episode.extend(coords_list)
                except SyntaxError:
                    print(f"Error parsing line: {line}")
                    continue
    
    if current_episode and len(current_episode) > 1:
        episodes.append(current_episode[:-1])
    
    return episodes

def calculate_metrics(episodes, start, goal, cell_size):
    """Tính toán các thông số cho tất cả episode."""
    oracle_length = calculateDistanceToGoal(start, goal)
    oracle_angle = np.pi / 4
    oracle_safety = 40.0
    
    path_lengths = []
    total_angles = []
    avg_angles = []
    success_lengths = []
    success_angles = []
    success_safeties = []
    fail_counter = 0
    
    for path in episodes:
        path_length = calculate_path_length(path)
        total_angle = get_sum_turning_angle(path)
        avg_angle = get_average_turning_angle(path)
        
        # Kiểm tra xem có đạt mục tiêu không (khoảng cách đến mục tiêu <= 1 ô lưới)
        success = ((path[-1][0] - goal[0])**2 + (path[-1][1] - goal[1])**2) ** 0.5 <= 1
        
        path_lengths.append(path_length)
        total_angles.append(total_angle)
        avg_angles.append(avg_angle)
        
        if not success:
            fail_counter += 1
            success_lengths.append(0)
            success_angles.append(0)
            success_safeties.append(0)
        else:
            success_lengths.append(oracle_length / max(oracle_length, path_length))
            success_angles.append(oracle_angle / max(oracle_angle, total_angle))
            success_safeties.append(1.0)  # Giả định an toàn tối đa nếu đạt mục tiêu
    
    success_rate = (1 - fail_counter / len(episodes)) * 100 if episodes else 0
    
    return {
        "avg_path_length": np.mean(path_lengths) if path_lengths else 0,
        "avg_total_angle": np.mean(total_angles) if total_angles else 0,
        "avg_turning_angle": np.mean(avg_angles) if avg_angles else 0,
        "success_rate": success_rate,
        "success_length": np.mean(success_lengths) if success_lengths else 0,
        "success_angle": np.mean(success_angles) if success_angles else 0,
        "success_safety": np.mean(success_safeties) if success_safeties else 0
    }

def main():
    # Chọn bản đồ
    selected_map = select_map()
    
    # Lấy thông tin start và goal
    start = maps[selected_map]["Start"]
    goal = maps[selected_map]["Goal"]
    
    # Tìm file mới nhất
    latest_file = find_latest_file(selected_map)
    if not latest_file:
        print(f"No path files found for map {selected_map} in results/{selected_map}/")
        return
    
    print(f"Processing latest file: {latest_file}")
    
    # Đọc các đường đi từ file
    episodes = read_paths_from_file(latest_file)
    
    if not episodes:
        print("No valid episodes found in the file.")
        return
    
    print(f"Found {len(episodes)} episodes")
    
    # Tính toán các thông số
    metrics = calculate_metrics(episodes, start, goal, cell_size=16)
    
    # In kết quả
    print("\nMetrics for all episodes:")
    print("-" * 60)
    print(f"Average Path Length: {metrics['avg_path_length']:.2f}")
    print(f"Average Total Turning Angle: {metrics['avg_total_angle']:.2f}")
    print(f"Average Turning Angle: {metrics['avg_turning_angle']:.2f}")
    print(f"Success Rate (%): {metrics['success_rate']:.2f}")
    print(f"Success Length: {metrics['success_length']:.4f}")
    print(f"Success Angle: {metrics['success_angle']:.4f}")
    print(f"Success Safety: {metrics['success_safety']:.4f}")
    
    # Lưu kết quả vào file xlsx nếu người dùng muốn
    if input("\nSave to xlsx? (y/n): ").lower() == "y":
        df = pd.DataFrame([[
            metrics['avg_path_length'],
            metrics['avg_total_angle'],
            metrics['avg_turning_angle'],
            metrics['success_rate'],
            metrics['success_length'],
            metrics['success_angle'],
            metrics['success_safety']
        ]], columns=[
            "Avg Path Length", "Avg Total Angle", "Avg Turning Angle",
            "Success Rate (%)", "Success Length", "Success Angle", "Success Safety"
        ])
        os.makedirs(f"results/{selected_map}", exist_ok=True)
        df.to_excel(f"results/{selected_map}/MetricsSummary.xlsx")
        print(f"Saved to results/{selected_map}/MetricsSummary.xlsx")

if __name__ == "__main__":
    main()