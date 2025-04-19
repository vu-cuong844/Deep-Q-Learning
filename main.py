import pygame
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from Robot import Robot
from Obstacle import Obstacle
from Controller import Controller
from Colors import *
from MapData import maps

def get_mode():
    """Get mode from user input with validation"""
    while True:
        mode = input("Enter mode (train/test): ").lower()
        if mode in ["train", "test"]:
            return mode == "train"
        print("Invalid mode. Please enter 'train' or 'test'")

# Khởi tạo Pygame
pygame.init()

# Thiết lập thông số môi trường
env_size = 512
cell_size = 16
env_padding = int(env_size * 0.06)  # ~30 pixel
GRID_WIDTH = env_size // cell_size  # 32 ô
GRID_HEIGHT = env_size // cell_size  # 32 ô
WINDOW_WIDTH = env_size + 2 * env_padding
WINDOW_HEIGHT = env_size + 2 * env_padding + 3 * env_padding  # Thêm không gian cho nút bấm và stats

# Thiết lập font
my_font = pygame.font.SysFont("arial", env_padding // 2)
stats_font = pygame.font.SysFont("arial", int(env_padding // 2.5))

# Chọn map
def select_map():
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

class Environment:
    def __init__(self, is_training=True, selected_map="map1", model_path="models/dqn_model.pth"):
        self.is_training = is_training
        self.model_path = model_path
        os.makedirs("models", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        
        # Initialize window
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Robot Navigation with DQN" + (" - Training" if is_training else " - Testing"))
        
        # Load map
        self.map_data = maps[selected_map]
        
        # Set up environment components
        self.setup_environment()
        
        # Training stats
        self.episode_count = 0
        self.step_count = 0
        self.reset_count = 0
        self.total_rewards = []
        self.episode_lengths = []
        self.best_reward = float('-inf')
        self.collision_count = 0
        self.goal_reached_count = 0
        self.prev_distance_to_goal = None
        self.episode_reward = 0
        self.episode_steps = 0
        self.last_save_time = time.time()
        
        # Clock for FPS control
        self.clock = pygame.time.Clock()
        self.FPS = 60 if not is_training else 240

    def setup_environment(self):
        # Get Start and Goal positions from map
        start_grid_x, start_grid_y = self.map_data["Start"]
        goal_grid_x, goal_grid_y = self.map_data["Goal"]
        
        # Convert grid positions to pixel coordinates
        self.start = (
            env_padding + (start_grid_x + 0.5) * cell_size,
            env_padding + (start_grid_y + 0.5) * cell_size
        )
        self.goal = (
            env_padding + (goal_grid_x + 0.5) * cell_size,
            env_padding + (goal_grid_y + 0.5) * cell_size
        )
        
        # Create obstacles from map data
        self.obstacles = []
        for obs_data in self.map_data["Obstacles"]:
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
            self.obstacles.append(obstacle)
        
        # Create controller and robot
        self.controller = Controller(self.goal, cell_size, env_padding, is_training=self.is_training, model_path=self.model_path)
        self.robot = Robot(self.start[0], self.start[1], cell_size, self.controller, vision=cell_size*2.5, radius=8, env_padding=env_padding)
        
        # Robot path history
        self.robot_path = [(self.robot.x, self.robot.y)]
        
        # UI elements
        self.button_start = pygame.Rect(env_padding + int(env_size * 0.7), env_padding * 2 + env_size,
                                int(env_size * 0.2), int(env_padding * 0.4))
        self.button_pause = pygame.Rect(env_padding + int(env_size * 0.4), env_padding * 2 + env_size,
                                int(env_size * 0.2), int(env_padding * 0.4))
        self.button_start_text = my_font.render("Start", True, BLACK)
        self.button_pause_text = my_font.render("Pause", True, BLACK)

    def reset_episode(self):
        """Reset the robot to the starting position and clear path history"""
        self.robot.x, self.robot.y = self.start
        self.robot.grid_x, self.robot.grid_y = self.map_data["Start"]
        self.robot_path.clear()
        self.robot_path.append((self.robot.x, self.robot.y))
        self.prev_distance_to_goal = None
        
        if self.is_training:
            self.total_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_steps)
            
            # Reset episode stats
            self.episode_count += 1
            self.episode_reward = 0
            self.episode_steps = 0
            
            # Update epsilon
            self.controller.update_epsilon()

    def draw_grid(self):
        """Draw the grid background"""
        self.window.fill(WHITE)
        for x in range(env_padding, WINDOW_WIDTH - env_padding, cell_size):
            pygame.draw.line(self.window, BLACK, (x, env_padding), (x, env_size + env_padding))
        for y in range(env_padding, env_size + env_padding, cell_size):
            pygame.draw.line(self.window, BLACK, (env_padding, y), (WINDOW_WIDTH - env_padding, y))

    def draw_stats(self):
        """Draw training/testing statistics"""
        stats_y = env_padding * 3 + env_size
        stats_x = env_padding
        line_height = env_padding // 3
        
        texts = []
        if self.is_training:
            texts = [
                f"Episodes: {self.episode_count}",
                f"Reset Count: {self.reset_count}",
                f"Epsilon: {self.controller.epsilon:.4f}",
                f"Goals: {self.goal_reached_count}",
                f"Collisions: {self.collision_count}"
            ]
            
            if len(self.total_rewards) > 0:
                avg_reward = sum(self.total_rewards[-100:]) / min(len(self.total_rewards), 100)
                texts.append(f"Avg Reward (100): {avg_reward:.2f}")
        else:
            texts = [
                f"Testing Mode",
                f"Goals: {self.goal_reached_count}",
                f"Collisions: {self.collision_count}"
            ]
        
        for i, text in enumerate(texts):
            text_surface = stats_font.render(text, True, BLACK)
            self.window.blit(text_surface, (stats_x, stats_y + i * line_height))

    def draw_environment(self):
        """Draw the complete environment"""
        # Draw grid
        self.draw_grid()
        
        # Draw obstacles
        for obstacle in self.obstacles:
            obstacle.draw(self.window)
        
        # Draw robot path
        if len(self.robot_path) > 1:
            pygame.draw.lines(self.window, RED, False, self.robot_path, 2)
        
        # Draw start and goal
        pygame.draw.rect(self.window, GREEN, (self.start[0]-5, self.start[1]-5, 10, 10))
        pygame.draw.circle(self.window, RED, self.goal, self.robot.radius)
        
        # Draw robot
        self.robot.draw(self.window)
        
        # Draw border
        pygame.draw.rect(self.window, BLACK, (env_padding, env_padding, env_size, env_size), 3)
        
        # Draw buttons
        pygame.draw.rect(self.window, BLACK, self.button_start, 4)
        pygame.draw.rect(self.window, BLACK, self.button_pause, 4)
        self.window.blit(self.button_start_text, self.button_start_text.get_rect(center=self.button_start.center))
        self.window.blit(self.button_pause_text, self.button_pause_text.get_rect(center=self.button_pause.center))
        
        # Draw stats
        self.draw_stats()

    def update(self):
        """Update environment state for one step"""
        # Move obstacles
        for obstacle in self.obstacles:
            obstacle.move()

        # Get current state
        state = self.robot.get_state(self.obstacles, GRID_WIDTH, GRID_HEIGHT, self.goal)
        _, current_distance_to_goal = state

        # Move robot
        direction = self.controller.make_decision(self.robot, self.obstacles)
        # Lưu lại hướng di chuyển để sử dụng trong training
        action_idx = self.controller.directions.index(direction)

        # Di chuyển robot và kiểm tra va chạm
        collided = self.robot.move(self.obstacles, GRID_WIDTH, GRID_HEIGHT)

        # Update path history
        if (self.robot.x, self.robot.y) != self.robot_path[-1]:
            self.robot_path.append((self.robot.x, self.robot.y))
            if len(self.robot_path) > 1000:  # Limit path length
                self.robot_path.pop(0)

        # Get next state
        next_state = self.robot.get_state(self.obstacles, GRID_WIDTH, GRID_HEIGHT, self.goal)
        _, next_distance_to_goal = next_state

        # Check for goal or collision
        reached_goal = ((self.robot.x - self.goal[0])**2 + (self.robot.y - self.goal[1])**2) ** 0.5 < 16
        done = reached_goal or collided

        if reached_goal:
            self.goal_reached_count += 1
        if collided:
            self.collision_count += 1

        # Update training information if in training mode
        if self.is_training:
            reward = self.controller.calculate_reward(
                self.robot, 
                self.obstacles, 
                done, 
                reached_goal, 
                next_distance_to_goal, 
                prev_distance=current_distance_to_goal
            )

            self.controller.store_experience(state, action_idx, reward, next_state, done)
            self.controller.train()

            # Update episode stats
            self.episode_reward += reward
            self.episode_steps += 1
            self.step_count += 1

        # Update previous distance
        self.prev_distance_to_goal = next_distance_to_goal

        # Reset if goal reached or collision
        if done:
            self.reset_count += 1
            self.reset_episode()

            # Save model periodically during training
            if self.is_training:
                current_time = time.time()
                if current_time - self.last_save_time > 60:  # Save every minute
                    self.save_model()
                    self.last_save_time = current_time

        return done

    def save_model(self):
        """Save model and training statistics"""
        # Save model
        self.controller.save_model()
        
        # Save metrics if we have enough data
        if len(self.total_rewards) > 1:
            self.plot_metrics()
        
        print(f"Model saved at episode {self.episode_count}")

    def plot_metrics(self):
        """Plot and save training metrics"""
        plt.figure(figsize=(12, 8))
        
        # Plot rewards
        plt.subplot(2, 1, 1)
        plt.plot(self.total_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        # Plot episode lengths
        plt.subplot(2, 1, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        plt.tight_layout()
        plt.savefig(f"plots/training_metrics_ep{self.episode_count}.png")
        plt.close()

    def train(self, max_episodes=1000):
        """Run training loop for specified number of episodes"""
        running = True
        started = False
        pause = False
        
        print(f"Training for {max_episodes} episodes...")
        
        while running and self.episode_count < max_episodes:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    if self.button_start.collidepoint(mouse_x, mouse_y):
                        started = True
                    elif self.button_pause.collidepoint(mouse_x, mouse_y):
                        pause = not pause
                        
            # Update environment if started and not paused
            if started and not pause:
                self.update()
            
            # Draw environment
            self.draw_environment()
            pygame.display.update()
            
            # Control FPS
            self.clock.tick(self.FPS)
        
        # Final save
        self.save_model()
        print("Training completed!")
        
        # Plot final metrics
        self.plot_metrics()
        
        return self.controller.model_path

    def save_path_to_file(self, episode_number):
        """Lưu đường đi hiện tại của robot vào file text"""
        # Tạo thư mục results nếu chưa tồn tại
        os.makedirs("results", exist_ok=True)

        # Tìm tên bản đồ bằng cách so sánh dữ liệu
        map_name = "unknown"
        for name, data in maps.items():
            if data == self.map_data:
                map_name = name
                break
            
        # Tạo tên file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results/robot_path_{map_name}_{timestamp}_ep{episode_number}.txt"

        # Lưu tọa độ đường đi
        with open(filename, "w") as f:
            for x, y in self.robot_path:
                # Chuyển đổi tọa độ pixel sang tọa độ lưới để dễ phân tích
                grid_x = (x - env_padding) // cell_size
                grid_y = (y - env_padding) // cell_size
                f.write(f"{grid_x},{grid_y}\n")

        print(f"Đã lưu đường đi vào {filename}")

    def test(self, episodes=10):
        """Chạy vòng lặp kiểm thử với số lượng episodes cụ thể"""
        running = True
        episode_count = 0
        steps_per_episode = []
        current_steps = 0

        # Tạo thư mục results nếu chưa tồn tại
        os.makedirs("results", exist_ok=True)

        # Tạo tên file chung cho tất cả các episode
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = f"results/robot_paths_{timestamp}.txt"

        # Tạo một danh sách để theo dõi đường đi của episode hiện tại
        current_episode_path = []

        print(f"Kiểm thử cho {episodes} episodes...")

        # Khởi tạo đường đi ban đầu
        current_episode_path.append((self.robot.x, self.robot.y))

        while running and episode_count < episodes:
            # Xử lý sự kiện
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Cập nhật vị trí robot trước khi gọi update()
            old_x, old_y = self.robot.x, self.robot.y

            # Cập nhật môi trường
            done = self.update()
            current_steps += 1

            # Kiểm tra nếu robot đã di chuyển, thêm vị trí mới vào đường đi
            if (self.robot.x, self.robot.y) != (old_x, old_y):
                current_episode_path.append((self.robot.x, self.robot.y))
            else:
                print(f"Bị kẹt tại: {self.robot.x}, {self.robot.y}.")

            if done:
                # Lưu đường đi của episode hiện tại vào file
                with open(result_file, "a") as f:
                    f.write(f"# Episode {episode_count + 1}\n")
                    for x, y in current_episode_path:
                        # Chuyển đổi tọa độ pixel sang tọa độ lưới
                        grid_x = int((x - env_padding) // cell_size)
                        grid_y = int((y - env_padding) // cell_size)
                        f.write(f"[{grid_x},{grid_y}], ")
                    f.write("\n")  # Dòng trống giữa các episode

                # Reset đường đi cho episode tiếp theo
                current_episode_path = [(self.start[0], self.start[1])]

                episode_count += 1
                steps_per_episode.append(current_steps)
                current_steps = 0
                print(f"Episode {episode_count}/{episodes} hoàn thành và đã lưu đường đi")

            # Vẽ môi trường
            self.draw_environment()
            pygame.display.update()

            # Kiểm soát FPS 
            self.clock.tick(self.FPS)

        # In kết quả kiểm thử
        success_rate = self.goal_reached_count / max(1, episode_count) * 100
        avg_steps = sum(steps_per_episode) / max(1, len(steps_per_episode))

        print("\nKết quả kiểm thử:")
        print(f"Episodes: {episode_count}")
        print(f"Đạt mục tiêu: {self.goal_reached_count} ({success_rate:.2f}%)")
        print(f"Va chạm: {self.collision_count}")
        print(f"Số bước trung bình: {avg_steps:.2f}")
        print(f"Tất cả đường đi đã được lưu vào file: {result_file}")

        return success_rate, avg_steps

def main():
    # Get mode from user
    is_training = get_mode()
    selected_map = select_map()
    
    if is_training:
        # Configure training
        model_name = f"dqn_{selected_map}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
        model_path = os.path.join("models", model_name)
        max_episodes = int(input("Enter number of training episodes (default 1000): ") or "1000")
        
        # Initialize and run training
        env = Environment(is_training=True, selected_map=selected_map, model_path=model_path)
        model_path = env.train(max_episodes=max_episodes)
        
        # Ask if user wants to test the trained model
        test_after = input("Test the trained model? (y/n): ").lower() == 'y'
        if test_after:
            env = Environment(is_training=False, selected_map=selected_map, model_path=model_path)
            env.test(episodes=5)
    else:
        # List available models
        models_dir = "models"
        if not os.path.exists(models_dir) or not os.listdir(models_dir):
            print("No trained models found. Please train a model first.")
            return
        
        print("\nAvailable models:")
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        for i, model_file in enumerate(model_files):
            print(f"{i+1}. {model_file}")
        
        # Select model
        while True:
            try:
                choice = int(input(f"Select model (1-{len(model_files)}): "))
                if 1 <= choice <= len(model_files):
                    model_path = os.path.join(models_dir, model_files[choice-1])
                    break
                print(f"Please enter a number between 1 and {len(model_files)}")
            except ValueError:
                print("Please enter a valid number")
        
        # Configure and run testing
        test_episodes = int(input("Enter number of test episodes (default 10): ") or "10")
        env = Environment(is_training=False, selected_map=selected_map, model_path=model_path)
        env.test(episodes=test_episodes)

    pygame.quit()

if __name__ == "__main__":
    main()