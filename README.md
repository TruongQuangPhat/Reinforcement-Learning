# Reinforcement Learning Grid-world

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-array%20computing-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-LP%20solver-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-figures-11557C?style=flat-square)](https://matplotlib.org/)
[![RL](https://img.shields.io/badge/RL%20Algorithms-from%20scratch-2E7D32?style=flat-square)](#thuật-toán)
[![Tests](https://img.shields.io/badge/Tests-unittest-6A1B9A?style=flat-square)](#kiểm-thử)

Repository này triển khai một nghiên cứu Reinforcement Learning từ đầu trên
bài toán 8x8 Grid-world Navigation. Dự án so sánh các thuật toán planning
model-based với các thuật toán learning sample-based trong cùng một môi trường
MDP, có seed cố định, log JSON có cấu trúc và hình ảnh phục vụ báo cáo.

Mục tiêu của project là phục vụ báo cáo Machine Learning theo hướng học thuật:
ưu tiên tính đúng đắn, giả định rõ ràng, khả năng tái lập và diễn giải kết quả
đúng với vai trò toán học của từng thuật toán.

## Phạm Vi Nghiên Cứu

Môi trường được mô hình hóa như một finite discounted Markov Decision Process:

```text
M = (S, A, P, r, gamma)
```

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

```text
V*(s) = max_a sum_{s'} P(s' | s, a) [r(s, a, s') + gamma V*(s')]
```

Ràng buộc trong Linear Programming:

```text
V(s) >= r(s, a) + gamma sum_{s'} P(s' | s, a) V(s')
```

### Learning Algorithms

Learning algorithms là nhóm sample-based, học từ các trajectory thu được khi
tương tác với môi trường.

- `TDZero`: one-step TD prediction cho state-value function.
- `TDLambda`: TD prediction với eligibility traces.
- `SARSA`: on-policy temporal-difference control.
- `QLearning`: off-policy temporal-difference control.

TD error cơ bản:

```text
delta_t = R_{t+1} + gamma V(S_{t+1}) - V(S_t)
```

## Cấu Trúc Repository

```text
Reinforcement-Learning/
├── AGENTS.md
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
│   ├── README.md
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

## Chạy Thí Nghiệm

Chạy toàn bộ experiment suite mặc định:

```bash
python scripts/run_experiments.py --verbose 1
```

Chạy với progress logs định kỳ:

```bash
python scripts/run_experiments.py --verbose 2 --log-interval 100 --window-size 100
```

Chạy im lặng:

```bash
python scripts/run_experiments.py --verbose 0
```

Các CLI options:

| Option | Ý nghĩa |
| --- | --- |
| `--verbose {0,1,2}` | `0` silent, `1` summary, `2` periodic progress logs |
| `--log-interval N` | Số iteration hoặc episode giữa hai dòng progress |
| `--window-size N` | Window cho moving average và final-window learning metrics |
| `--run-td-lambda-sweep` | Optional sensitivity theo lambda của TD(lambda) |
| `--run-control-sensitivity` | Optional epsilon sensitivity cho SARSA/Q-learning |
| `--run-gamma-sensitivity` | Optional gamma sensitivity cho Value Iteration |
| `--run-multiseed-smoke` | Optional lightweight multi-seed smoke test |

Chạy các experiment mở rộng:

```bash
python scripts/run_experiments.py --verbose 1 --run-td-lambda-sweep
python scripts/run_experiments.py --verbose 1 --run-control-sensitivity
python scripts/run_experiments.py --verbose 1 --run-gamma-sensitivity
python scripts/run_experiments.py --verbose 1 --run-multiseed-smoke
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

Không nên kết luận một learning method "kém hơn" planning method chỉ vì runtime
lâu hơn hoặc cần nhiều environment steps hơn. Planning algorithms được biết
model; learning algorithms phải ước lượng từ sampled experience.

## Outputs

Experiment logs được lưu dưới dạng JSON:

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

Notebook nên đọc lại logs và figures đã sinh, không reimplement core algorithms
và không âm thầm chạy lại experiment với cấu hình khác.

## Kiểm Thử

Chạy full test suite:

```bash
python scripts/run_tests.py
```

Tắt ANSI color nếu cần:

```bash
python scripts/run_tests.py --no-color
```

Kiểm tra syntax/import:

```bash
python -m compileall envs agents utils scripts tests
```

Test suite bao phủ environment behavior, planning algorithms, learning
algorithms, visualization helpers, experiment outputs và deterministic seeding.

## Tái Lập Kết Quả

Dự án hỗ trợ tái lập thông qua:

- Random seed cố định trong experiment runner và algorithm constructors.
- JSON logs cho training metrics, system metrics và summaries.
- Figures lưu sẵn cho report integration.
- Unit tests cho core behavior.
- Optional sensitivity và multi-seed smoke checks.

Để đưa ra kết luận thống kê đầy đủ, nên mở rộng runner để lưu các independent
runs theo timestamp và aggregate trên nhiều seed hơn. Built-in multi-seed smoke
test chỉ là kiểm tra ổn định nhẹ, không phải full statistical evaluation.

## Diễn Giải Học Thuật

Khi viết báo cáo, nên diễn giải cẩn trọng:

- `PolicyEvaluation`, `TDZero`, `TDLambda` là prediction methods.
- `PolicyIteration`, `ValueIteration`, `LinearProgrammingPlanner` là
  model-based optimal-control methods.
- `SARSA` và `QLearning` học control policies từ samples.
- TD(lambda) không mặc định tốt hơn TD(0); kết quả phụ thuộc `lambda`,
  learning rate, trajectory distribution và environment dynamics.
- Policy agreement thấp hơn `1.0` không tự động đồng nghĩa policy tệ, vì
  Grid-world có thể có nhiều hành động tối ưu hoặc gần tối ưu.