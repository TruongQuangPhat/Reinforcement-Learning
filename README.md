# Reinforcement Learning Grid-world

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-array%20computing-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-LP%20solver-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-figures-11557C?style=flat-square)](https://matplotlib.org/)
[![RL](https://img.shields.io/badge/RL%20Algorithms-from%20scratch-2E7D32?style=flat-square)](#thuật-toán)
[![Tests](https://img.shields.io/badge/Tests-unittest-6A1B9A?style=flat-square)](#kiểm-thử)

Repository này là phần mã nguồn và thực nghiệm cho đồ án nghiên cứu
**Chương 14 - Reinforcement Learning** trong giáo trình **Foundations of
Machine Learning**. Dự án triển khai từ đầu bài toán **8x8 Grid-world
Navigation** nhằm minh họa mô hình Markov Decision Process (MDP), hàm giá trị,
chính sách, phương trình Bellman, các thuật toán Planning và các thuật toán
Learning.

Trong đồ án, nhóm so sánh các thuật toán planning model-based với các thuật
toán learning sample-based trên cùng một môi trường Grid-world. Project sử dụng
seed cố định, lưu kết quả thực nghiệm dưới dạng JSON và sinh hình ảnh trực quan
phục vụ phân tích trong báo cáo.

Mục tiêu của project là đảm bảo tính đúng đắn, khả năng tái lập và diễn giải
kết quả phù hợp với vai trò toán học của từng thuật toán, thay vì chỉ tập trung
vào việc chạy được chương trình.

## Thông Tin Nhóm

| Họ tên | MSSV |
| --- | --- |
| Phạm Quốc Khánh | 23120283 |
| Phạm Thành Nam | 23120301 |
| Trương Quang Phát | 23120318 |
| Châu Huỳnh Phúc | 23120329 |
| Huỳnh Tấn Phước | 23120334 |

## Nội Dung Đã Thực Hiện Và Mở Rộng

Nhóm đã triển khai từ đầu môi trường **Grid-world 8x8** và các thuật toán
Reinforcement Learning thuộc hai nhóm chính: Planning Algorithms và Learning
Algorithms. Phần Planning sử dụng mô hình chuyển trạng thái đầy đủ của MDP,
trong khi phần Learning chỉ học thông qua sample interaction với môi trường.

So với nội dung lý thuyết trong sách, project mở rộng theo hướng thực nghiệm:
tách riêng `PlanningGridWorld` và `LearningGridWorld`, bổ sung logging kết quả
dạng JSON, profiling runtime/CPU/memory, visualization cho value function và
policy, notebooks phân tích kết quả, sensitivity analysis cho một số siêu tham
số và test suite để kiểm tra tính đúng đắn của implementation.

## Phạm Vi Nghiên Cứu

Môi trường được mô hình hóa như một finite discounted Markov Decision Process:

$$
M = (S, A, P, r, \gamma)
$$

Cấu hình Grid-world mặc định:

- Kích thước lưới: `8 x 8`
- Trạng thái bắt đầu: `(0, 0)`
- Trạng thái đích: ô góc dưới bên phải
- Trap states: `(3, 3)`, `(4, 4)`
- Wall states: `(2, 2)`, `(2, 3)`, `(5, 5)`
- Tập hành động: `up`, `down`, `left`, `right`
- Reward: step `-1`, goal `+10`, trap `-10`
- Discount factor: `gamma = 0.99`
- Random seed: `42`

Dự án tách rõ hai cách nhìn môi trường:

- `PlanningGridWorld`: cho phép truy cập full transition model qua
  `get_transitions(state, action)`.
- `LearningGridWorld`: chỉ cho phép tương tác bằng sample qua `reset()` và
  `step(action)`.

Các learning algorithms không được dùng `get_transitions()` trong quá trình
training.

## Thuật Toán

### Planning Algorithms

Planning algorithms là nhóm model-based, được phép dùng transition model đầy đủ
của MDP.

- `PolicyEvaluation`: đánh giá value function của một fixed policy.
- `PolicyIteration`: lặp giữa policy evaluation và greedy policy improvement.
- `ValueIteration`: cập nhật trực tiếp bằng Bellman optimality backup.
- `LinearProgrammingPlanner`: giải discounted MDP bằng Bellman optimality
  inequalities.

Bellman optimality backup:

$$
V^{\ast}(s) = \max_a \sum_{s'} P(s' \mid s, a)
\left[r(s, a, s') + \gamma V^{\ast}(s')\right]
$$

Linear Programming objective:

$$
\min_V \sum_s V(s)
$$

Ràng buộc Bellman inequality:

$$
V(s) \ge r(s, a) + \gamma \sum_{s'} P(s' \mid s, a)V(s')
$$

### Learning Algorithms

Learning algorithms là nhóm sample-based, học từ các trajectory thu được khi
tương tác với môi trường.

- `TDZero`: one-step TD prediction cho state-value function.
- `TDLambda`: TD prediction với eligibility traces.
- `SARSA`: on-policy temporal-difference control.
- `QLearning`: off-policy temporal-difference control.

TD error cơ bản:

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

## Cấu Trúc Repository

```text
Reinforcement-Learning/
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_experiments.py
│   └── run_tests.py
├── agents/
│   ├── planning.py
│   └── learning.py
├── envs/
│   ├── grid_world.py
│   ├── planning_grid_world.py
│   └── learning_grid_world.py
├── utils/
│   ├── experiment_io.py
│   ├── logging_utils.py
│   ├── metrics.py
│   ├── profiling.py
│   └── visualization.py
├── notebooks/
│   ├── planning.ipynb
│   └── learning.ipynb
├── logs/
│   ├── planning/
│   └── learning/
├── report/
│   └── figures/
└── tests/
```

Vai trò chính:

- `envs/`: định nghĩa Grid-world interface và hai biến thể planning/learning.
- `agents/`: triển khai các thuật toán planning và learning từ đầu.
- `utils/`: metrics, profiling, JSON IO, logging và visualization.
- `scripts/`: chứa các entry point chạy experiment và test suite.
- `logs/`: kết quả thực nghiệm dạng JSON.
- `report/figures/`: hình ảnh dùng cho notebook và báo cáo.
- `tests/`: kiểm thử environment, algorithms, visualization và experiment
  runner.

## Cài Đặt

Tạo virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies chính:

- NumPy cho bảng giá trị, policy và tính toán số.
- SciPy cho Linear Programming planner.
- Matplotlib và Seaborn cho hình ảnh báo cáo.

Các thuật toán RL được cài đặt from scratch. Project không dùng Gymnasium,
Stable-Baselines, RLlib hoặc scikit-learn cho phần thuật toán.

## Tái Tạo Kết Quả Thực Nghiệm

Để tái tạo kết quả, cài dependencies từ `requirements.txt`, sau đó chạy:

```bash
python scripts/run_experiments.py --verbose 1
```

Lệnh trên sinh lại các file JSON trong `logs/planning/` và `logs/learning/`,
đồng thời tạo/cập nhật các hình trong `report/figures/`. Project dùng seed mặc
định `42`, vì vậy các kết quả chính có thể tái lập trên cùng cấu hình phần mềm.

Để tái tạo đầy đủ các experiment mở rộng được liệt kê trong phần Outputs, chạy:

```bash
python scripts/run_experiments.py --verbose 1 --run-td-lambda-sweep
python scripts/run_experiments.py --verbose 1 --run-control-sensitivity
python scripts/run_experiments.py --verbose 1 --run-gamma-sensitivity
python scripts/run_experiments.py --verbose 1 --run-multiseed-smoke
```

Sau khi chạy experiment, có thể kiểm tra nhanh tính đúng đắn bằng:

```bash
python scripts/run_tests.py
```

## Thiết Kế Đánh Giá

### Planning Metrics

- Bellman residuals
- Iterations
- Policy changes
- Bellman backups
- Value function
- Policy
- Runtime
- CPU time
- Current memory và peak memory
- Policy agreement so với Value Iteration khi phù hợp
- Value error so với Value Iteration khi phù hợp

### Learning Metrics

- Episode returns
- Moving average returns
- Episode steps
- Success rate
- Trap rate
- TD errors
- MSE so với planning baseline đúng
- Policy agreement so với Value Iteration cho control methods
- Runtime
- CPU time
- Current memory và peak memory
- Environment steps
- TD hoặc Q updates

### So Sánh Hợp Lệ

Project tách rõ prediction, control, planning và learning để tránh diễn giải
sai kết quả.

| Câu hỏi nghiên cứu | So sánh hợp lệ |
| --- | --- |
| Độ chính xác fixed-policy prediction | `PolicyEvaluation` vs `TDZero` / `TDLambda` |
| Agreement giữa optimal planning methods | `PolicyIteration` vs `ValueIteration` vs `LinearProgrammingPlanner` |
| Hành vi sample-based control | `SARSA` vs `QLearning` |
| Learned control so với optimal planning baseline | `SARSA` / `QLearning` vs `ValueIteration` |
| Ảnh hưởng của eligibility traces | `TDZero` vs `TDLambda` |

## Outputs

```text
logs/
├── planning/
│   ├── training_metrics.json
│   ├── system_metrics.json
│   ├── experiment_summary.json
│   └── gamma_sensitivity.json
└── learning/
    ├── training_metrics.json
    ├── system_metrics.json
    ├── experiment_summary.json
    ├── td_lambda_sweep.json
    ├── control_sensitivity.json
    └── multiseed_smoke.json
```

Figures được lưu tại:

```text
report/figures/
├── planning/
├── learning/
└── comparison/
```