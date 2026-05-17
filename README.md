# Reinforcement Learning Grid-world 8x8

Đồ án Reinforcement Learning from scratch cho bài toán Grid-world Navigation
kích thước 8 x 8. Agent bắt đầu từ Start, tìm đường đến Goal, và cần tránh
Trap/Wall trong một môi trường MDP hữu hạn.

## Thông Tin Nhóm

| STT | Họ và Tên | MSSV | Vai trò chính |
|:---:|:----------|:-----|:--------------|
| 1 | Trương Quang Phát | 2312xxxx | Tích hợp báo cáo, review mã nguồn, tổng quan và kết luận |
| 2 | [Tên Thành viên 2] | 2312xxxx | Lý thuyết RL, MDP, policy, chứng minh toán học |
| 3 | [Tên Thành viên 3] | 2312xxxx | Planning, learning, chứng minh toán học |
| 4 | [Tên Thành viên 4] | 2312xxxx | Grid-world, thuật toán NumPy, trực quan hóa |
| 5 | [Tên Thành viên 5] | 2312xxxx | Nghiên cứu mở rộng và phân tích |

## Mục Tiêu

- Xây dựng môi trường Grid-world 8x8 theo mô hình MDP.
- Cài đặt và so sánh hai nhóm thuật toán Reinforcement Learning cổ điển.
- Lưu metrics, system profiling, hình trực quan hóa và kết quả phục vụ báo cáo.
- Tất cả thuật toán được cài đặt from scratch, không dùng thư viện RL có sẵn.

## Thuật Toán

Planning algorithms:

- Policy Evaluation
- Policy Iteration
- Value Iteration
- Linear Programming

Learning algorithms:

- TD(0)
- SARSA
- Q-learning
- TD(lambda)

## Cấu Trúc Thư Mục

```text
Reinforcement-Learning/
├── README.md
├── requirements.txt
├── .gitignore
├── run_experiments.py
├── envs/
│   ├── __init__.py
│   ├── grid_world.py
│   ├── planning_grid_world.py
│   └── learning_grid_world.py
├── agents/
│   ├── __init__.py
│   ├── planning.py
│   └── learning.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── profiling.py
│   ├── experiment_io.py
│   └── visualization.py
├── notebooks/
│   ├── planning.ipynb
│   └── learning.ipynb
├── logs/
│   ├── planning/
│   │   ├── training_metrics.json
│   │   ├── system_metrics.json
│   │   └── experiment_summary.json
│   └── learning/
│       ├── training_metrics.json
│       ├── system_metrics.json
│       └── experiment_summary.json
└── report/
    ├── figures/
    │   ├── planning/
    │   ├── learning/
    │   └── comparison/
    └── README.md
```

## Cách Chạy Dự Kiến

```bash
python3 run_experiments.py
```
