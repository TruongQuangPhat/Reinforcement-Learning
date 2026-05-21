# Reinforcement Learning Grid-world

## 1. Tổng quan

Project này triển khai và so sánh các thuật toán Reinforcement Learning cổ điển
trên bài toán Grid-world Navigation kích thước 8x8. Agent bắt đầu từ trạng thái
Start, tương tác với môi trường, tránh Wall/Trap và tìm đường đến Goal.

Trong Reinforcement Learning, agent học cách ra quyết định thông qua quá trình
tương tác với environment. Grid-world là một môi trường nhỏ gọn nhưng đủ rõ để
minh họa các khái niệm cốt lõi như Markov Decision Process (MDP), Bellman
equation, policy, value function, Planning và Learning.

Project tập trung vào hai nhóm thuật toán:

- **Planning Algorithms**: biết đầy đủ model của MDP và dùng transition model để
  tính toán.
- **Learning Algorithms**: học từ các trajectory sample bằng `reset()` và
  `step(action)`, không truy cập trực tiếp transition model trong training.

Mục tiêu chính là xây dựng code từ đầu, lưu kết quả thực nghiệm có cấu trúc, và
chuẩn bị dữ liệu/hình ảnh phục vụ notebook và báo cáo học thuật.

## 2. Tính năng chính

- Môi trường Grid-world 8x8 với Start, Goal, Trap và Wall.
- Tách riêng `PlanningGridWorld` model-based và `LearningGridWorld`
  sample-based.
- Triển khai 8 thuật toán Reinforcement Learning cổ điển.
- Thu thập metrics, logging và profiling cho planning/learning experiments.
- Hỗ trợ verbose training logs:
  - `verbose=0`: silent.
  - `verbose=1`: summary.
  - `verbose=2`: progress table.
- Lưu experiment outputs dưới dạng JSON.
- Sinh figures phục vụ phân tích và báo cáo.
- Hỗ trợ optional sensitivity experiments.
- Có unit tests cho environment, algorithms và experiment runner.

## 3. Các thuật toán được triển khai

### 3.1. Planning Algorithms

Planning algorithms là nhóm thuật toán model-based. Các thuật toán này dùng
`get_transitions(state, action)` để truy cập transition model của MDP.

- **Policy Evaluation**: đánh giá value function của một policy cố định.
- **Policy Iteration**: lặp giữa policy evaluation và greedy policy improvement.
- **Value Iteration**: cập nhật trực tiếp bằng Bellman optimality backup.
- **Linear Programming Planner**: giải MDP thông qua Bellman optimality
  inequalities.

Bellman optimality backup có dạng:

```text
V*(s) = max_a sum_{s'} P(s'|s,a) [r(s,a,s') + gamma V*(s')]
```

Linear Programming Planner dùng ràng buộc:

```text
V(s) >= r(s,a) + gamma sum_{s'} P(s'|s,a)V(s')
```

### 3.2. Learning Algorithms

Learning algorithms là nhóm thuật toán sample-based/model-free. Trong training,
các thuật toán này chỉ dùng `reset()` và `step(action)`, không gọi
`get_transitions()`.

- **TD(0)**: học state-value function bằng one-step TD update.
- **TD(lambda)**: mở rộng TD với eligibility traces.
- **SARSA**: on-policy control, cập nhật dựa trên action thật sự được chọn ở
  next state.
- **Q-learning**: off-policy control, cập nhật dựa trên greedy target
  `max_a Q(S', a)`.

TD error cơ bản:

```text
delta_t = R_{t+1} + gamma V(S_{t+1}) - V(S_t)
```

## 4. Cấu trúc thư mục

```text
Reinforcement-Learning/
├── envs/
│   ├── grid_world.py
│   ├── planning_grid_world.py
│   └── learning_grid_world.py
├── agents/
│   ├── planning.py
│   └── learning.py
├── utils/
│   ├── metrics.py
│   ├── profiling.py
│   ├── experiment_io.py
│   ├── logging_utils.py
│   └── visualization.py
├── logs/
│   ├── planning/
│   └── learning/
├── notebooks/
│   ├── planning.ipynb
│   └── learning.ipynb
├── report/
│   └── figures/
├── tests/
├── run_experiments.py
├── run_tests.py
├── requirements.txt
└── README.md
```

Vai trò chính:

- `envs/`: định nghĩa môi trường Grid-world và hai interface planning/learning.
- `agents/`: chứa các thuật toán Planning và Learning.
- `utils/`: metrics, profiling, JSON IO, logging helper và visualization.
- `logs/`: kết quả thực nghiệm dạng JSON.
- `notebooks/`: phân tích, diễn giải thuật toán và trình bày kết quả.
- `report/figures/`: figures phục vụ báo cáo.
- `tests/`: kiểm thử environment, algorithms và experiment runner.
- `run_experiments.py`: entry point chạy experiment và sinh logs/figures.

## 5. Cài đặt

Project dùng Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Các dependency chính hiện tại:

- NumPy
- SciPy
- Matplotlib
- Seaborn

Các thuật toán RL được cài đặt from scratch; project không dùng Gymnasium,
Stable-Baselines, RLlib hoặc scikit-learn cho phần thuật toán.

## 6. Chạy thực nghiệm

Chạy chuẩn, có summary cho từng thuật toán:

```bash
python3 run_experiments.py --verbose 1
```

Chạy với progress chi tiết theo iteration/episode:

```bash
python3 run_experiments.py --verbose 2 --log-interval 100 --window-size 100
```

Chạy im lặng:

```bash
python3 run_experiments.py --verbose 0
```

Ý nghĩa các tham số:

- `verbose=0`: không in progress/summary thông thường.
- `verbose=1`: in start/done summary và final summary table.
- `verbose=2`: in progress table định kỳ.
- `log_interval`: số iteration hoặc episode giữa hai lần in progress log.
- `window_size`: kích thước cửa sổ dùng cho moving average và final-window
  metrics trong learning experiments.

Terminal output chỉ hiển thị summary/progress ngắn gọn. Value function,
Q-table, policy và raw arrays được lưu trong JSON logs thay vì in trực tiếp ra
terminal.

## 7. Các thực nghiệm mở rộng tùy chọn

Các experiment sau không chạy mặc định.

TD(lambda) sweep:

```bash
python3 run_experiments.py --verbose 1 --run-td-lambda-sweep
```

Control epsilon sensitivity cho SARSA/Q-learning:

```bash
python3 run_experiments.py --verbose 1 --run-control-sensitivity
```

Gamma sensitivity cho Value Iteration:

```bash
python3 run_experiments.py --verbose 1 --run-gamma-sensitivity
```

Multi-seed smoke test cho SARSA/Q-learning:

```bash
python3 run_experiments.py --verbose 1 --run-multiseed-smoke
```

Ý nghĩa:

- **TD(lambda) sweep**: kiểm tra ảnh hưởng của `lambda` lên TD(lambda).
- **Control sensitivity**: kiểm tra ảnh hưởng của `epsilon` lên SARSA và
  Q-learning.
- **Gamma sensitivity**: kiểm tra ảnh hưởng của discount factor `gamma` lên
  Value Iteration.
- **Multi-seed smoke**: kiểm tra ổn định nhẹ qua một vài random seeds. Đây không
  phải full statistical evaluation.

SARSA và Q-learning hỗ trợ optional epsilon decay ở class constructor:

- Fixed epsilon là mặc định.
- `epsilon_decay` chỉ có hiệu lực khi được cấu hình rõ.
- Nếu dùng epsilon decay, `epsilon_history` và `final_epsilon` được lưu trong
  metrics.

## 8. Metrics và phương pháp đánh giá

### 8.1. Planning metrics

- Bellman residual.
- Số iterations.
- Bellman backups.
- Runtime.
- CPU time và peak memory nếu có.
- Policy agreement.
- Value error so với Value Iteration.

Policy Iteration và Linear Programming Planner được so sánh với Value Iteration
vì đây là các thuật toán optimal-control trong nhóm planning. LP solver
iterations không được xem là Bellman iterations.

### 8.2. TD Prediction metrics

- TD errors.
- Mean absolute TD error theo episode.
- MSE so với Policy Evaluation.
- MSE checkpoint curve so với Policy Evaluation.
- Runtime, CPU time, peak memory.
- Environment steps.

Policy Evaluation là baseline phù hợp cho TD(0) và TD(lambda), vì cả ba đều là
prediction/evaluation methods dưới một policy.

### 8.3. Control Learning metrics

- Episode returns.
- Moving average returns.
- Success rate.
- Trap rate.
- Episode steps.
- TD errors.
- Q updates.
- MSE so với Value Iteration, với `V_learned(s) = max_a Q(s,a)`.
- Policy agreement so với Value Iteration.

Value Iteration là baseline optimal-control cho SARSA và Q-learning.

### 8.4. System profiling

Experiment logs lưu các system metrics như:

- Wall-clock runtime.
- CPU time nếu có.
- Current memory và peak memory nếu có.
- Bellman backups cho planning.
- Environment steps và Q/TD updates cho learning.

## 9. Logs, figures và outputs

Logs chính:

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

Các file sensitivity chỉ xuất hiện hoặc được cập nhật khi chạy flag tương ứng.

Figures được lưu dưới:

```text
report/figures/
├── planning/
├── learning/
└── comparison/
```

Những outputs này được thiết kế để notebook và báo cáo đọc lại trực tiếp, thay
vì phải chạy lại toàn bộ thuật toán trong notebook.

## 10. Chạy kiểm thử

Chạy toàn bộ test suite:

```bash
python3 run_tests.py
```

Nếu terminal không hỗ trợ ANSI color:

```bash
python3 run_tests.py --no-color
```

Kiểm tra syntax/import:

```bash
python3 -m compileall envs agents utils run_experiments.py tests
```

Test suite hiện bao phủ:

- Behavior của environment.
- Correctness cơ bản của Planning Algorithms.
- Correctness cơ bản của Learning Algorithms.
- Experiment runner, JSON logs và generated figures.
- Reproducibility theo seed.
- Optional epsilon decay metrics.

Ở thời điểm hiện tại, test suite gồm 24 tests và đang pass.

## 11. Notebooks

- `notebooks/planning.ipynb`: phân tích Planning Algorithms, Bellman residuals,
  policy changes, runtime/backups và so sánh PI/LP với VI.
- `notebooks/learning.ipynb`: phân tích TD methods, SARSA, Q-learning, learning
  curves, TD errors, MSE và policy agreement.

Notebook nên import code từ `envs/`, `agents/`, `utils/` và đọc JSON logs/figures
đã sinh sẵn. Không nên reimplement core algorithms trong notebook.

## 12. Lưu ý khi diễn giải kết quả

- Planning biết model của MDP; Learning học từ samples. Không nên kết luận
  learning "kém hơn" planning chỉ vì chạy chậm hơn hoặc dùng nhiều environment
  steps hơn.
- Không so TD(0) trực tiếp với Value Iteration như thể hai thuật toán giải cùng
  một bài toán.
- TDZero và TDLambda nên được so với PolicyEvaluation.
- SARSA và QLearning nên được so với ValueIteration.
- TD(lambda) không đảm bảo luôn tốt hơn TD(0); kết quả phụ thuộc `lambda`,
  `alpha` và trajectory.
- Success rate cao không đồng nghĩa policy agreement với Value Iteration bằng
  1.0.
- Policy agreement thấp hơn 1.0 không nhất thiết nghĩa là policy học được tệ, vì
  trong Grid-world có thể tồn tại nhiều hành động gần tối ưu hoặc cho rollout tốt.
- Multi-seed smoke chỉ là kiểm tra ổn định nhẹ, không thay thế full statistical
  evaluation.

## 13. Tái lập kết quả

Project hỗ trợ tái lập kết quả thông qua:

- Random seed cố định trong experiment runner.
- JSON logs cho training metrics, system metrics và experiment summaries.
- Unit tests có thể chạy lại độc lập.
- Optional multi-seed smoke để kiểm tra ổn định ở mức nhẹ.

Khi cần lưu nhiều batch experiment độc lập, nên mở rộng runner để ghi vào các
timestamped output folders thay vì ghi đè các file JSON hiện tại.

## 14. Hướng phát triển tiếp theo

- Timestamped experiment output folders.
- Larger multi-seed experiments cho đánh giá thống kê đầy đủ hơn.
- Risky Grid-world variant để làm rõ khác biệt giữa SARSA và Q-learning.
- Deep RL extension cho môi trường lớn hoặc quan sát phức tạp hơn.
- Tích hợp LaTeX report hoàn chỉnh.

## 15. License / Acknowledgement

License: Not specified yet.

Project được xây dựng cho mục đích học thuật, nhằm minh họa các thuật toán
Reinforcement Learning cổ điển trên một môi trường Grid-world hữu hạn.
