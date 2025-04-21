# Định nghĩa các map với thông tin về Start, Goal, và Obstacles
# Start và Goal giờ đây được định nghĩa bằng chỉ số ô lưới (grid_x, grid_y)
maps = {
    "map1": {
        "Start": (1, 16),  # Ô (1, 16) trên lưới 32x32
        "Goal": (30, 5),   # Ô (30, 5) trên lưới 32x32
        "Obstacles": [
            # Chướng ngại vật tĩnh (màu đen, xoay góc)
            {
                "x": 126,  # env_padding + cell_size * 8
                "y": 110,  # env_padding + cell_size * 5
                "width": 100,
                "height": 30,
                "static": True,
                "angle": 0
            },
            {
                "x": 430,  # env_padding + cell_size * 25
                "y": 270,  # env_padding + cell_size * 15
                "width": 30,
                "height": 150,
                "static": True,
                "angle": 0
            },
            {
                "x": 222,  # env_padding + cell_size * 12
                "y": 430,  # env_padding + cell_size * 25
                "width": 100,
                "height": 30,
                "static": True,
                "angle": 45
            },
            # Chướng ngại vật động (màu xanh nhạt)
            {
                "x": 270,  # env_padding + cell_size * 15
                "y": 190,  # env_padding + cell_size * 10
                "width": 30,
                "height": 30,
                "static": False,
                "velocity": [1.5, 0],  # Giảm từ 1.5 xuống 0.5
                "x_bound": (110, 430),  # env_padding + cell_size * 5, env_padding + cell_size * 25
                "y_bound": (190, 190)
            },
            {
                "x": 110,  # env_padding + cell_size * 5
                "y": 350,  # env_padding + cell_size * 20
                "width": 30,
                "height": 30,
                "static": False,
                "path": [
                    (110, 350),  # env_padding + cell_size * 5, env_padding + cell_size * 20
                    (190, 430),  # env_padding + cell_size * 10, env_padding + cell_size * 25
                    (270, 350),  # env_padding + cell_size * 15, env_padding + cell_size * 20
                    (190, 270),  # env_padding + cell_size * 10, env_padding + cell_size * 15
                    (110, 350)   # env_padding + cell_size * 5, env_padding + cell_size * 20
                ]
            }
        ]
    },
    "map2": {
        "Start": (1, 1),   # Góc trên trái
        "Goal": (30, 30),  # Góc dưới phải
        "Obstacles": [
            # Chướng ngại vật tĩnh
            {
                "x": 190,  # env_padding + cell_size * 10
                "y": 190,  # env_padding + cell_size * 10
                "width": 150,
                "height": 30,
                "static": True,
                "angle": 30
            },
            {
                "x": 350,  # env_padding + cell_size * 20
                "y": 350,  # env_padding + cell_size * 20
                "width": 30,
                "height": 150,
                "static": True,
                "angle": 0
            },
            # Chướng ngại vật động
            {
                "x": 270,  # env_padding + cell_size * 15
                "y": 270,  # env_padding + cell_size * 15
                "width": 30,
                "height": 30,
                "static": False,
                "velocity": [2.0, 1.0],  # Giảm từ 2.0, 1.0 xuống 0.5, 0.5
                "x_bound": (110, 430),
                "y_bound": (110, 430)
            }
        ]
    },

    "map3": {
        "Start": (1, 1),  # Green circle in top-left corner
        "Goal": (30, 30),  # Red circle in bottom-right corner
        "Obstacles": [
            # Black diamond obstacles arranged in a grid pattern
            # Row 1
            {
                "x": 110,  # cell 5
                "y": 110,  # cell 5
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45  # Diamond shape
            },
            {
                "x": 190,  # cell 10
                "y": 110,  # cell 5
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },
            {
                "x": 270,  # cell 15
                "y": 110,  # cell 5
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },
            {
                "x": 350,  # cell 20
                "y": 110,  # cell 5
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },

            # Row 2
            {
                "x": 110,  # cell 5
                "y": 190,  # cell 10
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },
            {
                "x": 190,  # cell 10
                "y": 190,  # cell 10
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },
            {
                "x": 270,  # cell 15
                "y": 190,  # cell 10
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },
            {
                "x": 350,  # cell 20
                "y": 190,  # cell 10
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },

            # Row 3
            {
                "x": 110,  # cell 5
                "y": 270,  # cell 15
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },
            {
                "x": 190,  # cell 10
                "y": 270,  # cell 15
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },
            {
                "x": 270,  # cell 15
                "y": 270,  # cell 15
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },
            {
                "x": 350,  # cell 20
                "y": 270,  # cell 15
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },

            # Row 4
            {
                "x": 110,  # cell 5
                "y": 350,  # cell 20
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },
            {
                "x": 190,  # cell 10
                "y": 350,  # cell 20
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },
            {
                "x": 270,  # cell 15
                "y": 350,  # cell 20
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            },
            {
                "x": 350,  # cell 20
                "y": 350,  # cell 20
                "width": 30,
                "height": 30,
                "static": True,
                "angle": 45
            }
        ]
    },
    "map4": {
        "Start": (1, 1),  # Điểm bắt đầu
        "Goal": (30, 30),  # Điểm kết thúc
        "Obstacles": [
            # Hàng 1 - các thanh ngang
            {
                "x": 150,
                "y": 110,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },
            {
                "x": 390,
                "y": 110,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },

            # Hàng 2
            {
                "x": 230,
                "y": 150,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },

            # Hàng 3
            {
                "x": 150,
                "y": 190,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },
            {
                "x": 390,
                "y": 190,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },

            # Hàng 4
            {
                "x": 230,
                "y": 230,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },

            # Hàng 5
            {
                "x": 150,
                "y": 270,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },
            {
                "x": 390,
                "y": 270,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },

            # Hàng 6
            {
                "x": 230,
                "y": 310,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },

            # Hàng 7
            {
                "x": 150,
                "y": 350,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },
            {
                "x": 390,
                "y": 350,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },

            # Hàng 8
            {
                "x": 230,
                "y": 390,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },

            # Hàng 9
            {
                "x": 150,
                "y": 430,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },
            {
                "x": 390,
                "y": 430,  # Giảm xuống
                "width": 150,
                "height": 16,
                "static": True,
                "angle": 0
            },

            # Chướng ngại vật động (tam giác xanh lam)
            {
                "x": 80,
                "y": 290,  # Điều chỉnh theo tọa độ y mới của hàng 5
                "width": 20,
                "height": 20,
                "static": False,
                "velocity": [1.0, 0],
                "x_bound": (80, 120),
                "y_bound": (290, 290)  # Điều chỉnh theo tọa độ y mới
            },
            {
                "x": 500,
                "y": 400,  # Điều chỉnh theo tọa độ y mới của hàng 3
                "width": 20,
                "height": 20,
                "static": False,
                "velocity": [-1.0, 0],
                "x_bound": (350, 500),
                "y_bound": (400, 400)  # Điều chỉnh theo tọa độ y mới
            },
        ]
    }
}       