# Đồ Án 2: Một số chủ đề nền tảng trong học máy (CSC14005)

**Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM**  
**Khoa Công nghệ Thông tin - Bộ môn Khoa học máy tính**

---

## 1. Thông tin nhóm

| STT | Họ và Tên | MSSV | Vai trò chính |
|:---:|:----------|:-----|:--------------|
| 1 | Trương Quang Phát  | 2312xxxx | Tích hợp báo cáo, Review mã nguồn, Viết Tổng quan và Kết luận |
| 2 | [Tên Thành viên 2] | 2312xxxx | Lý thuyết RL (MDP, Policy), Chứng minh toán học |
| 3 | [Tên Thành viên 3] | 2312xxxx | Lý thuyết RL (Planning, Learning), Chứng minh toán học |
| 4 | [Tên Thành viên 4] | 2312xxxx | Cài đặt môi trường Grid-world, Thuật toán NumPy, Trực quan hóa |
| 5 | [Tên Thành viên 5] | 2312xxxx | Nghiên cứu bài báo tiên tiến, Phân tích mở rộng |

---

## 2. Giới thiệu

Dự án này tập trung vào nghiên cứu và cài đặt các thuật toán cơ bản trong Reinforcement Learning (RL), bao gồm các phương pháp Planning và Learning. Nội dung được xây dựng dựa trên Chương 14 của giáo trình "Foundations of Machine Learning" (2nd Edition) của Mehryar Mohri, Afshin Rostamizadeh và Ameet Talwalkar, xuất bản bởi MIT Press năm 2018.

---

## 3. Cấu trúc dự án

```
Reinforcement-Learning/
├── README.md                      
├── requirements.txt               
├── .gitignore                     
├── run_experiments.py             # File chính để chạy toàn bộ thí nghiệm
│
├── envs/                          # Thư mục môi trường
│   ├── __init__.py                # Khởi tạo module envs
│   └── grid_world.py              # Cài đặt môi trường Grid-world (state, action, reward)
│
├── agents/                        # Thư mục chứa các thuật toán RL
│   ├── __init__.py                # Khởi tạo module agents
│   ├── planning.py                # Cài đặt Value Iteration và Policy Iteration
│   └── learning.py                # Cài đặt Q-learning và SARSA
│
├── advanced/                      # Nội dung nâng cao và mở rộng
│   ├── __init__.py                # Khởi tạo module advanced
│   └── rlhf_demo.py               # Demo RLHF và DPO
│
├── utils/                         # Các hàm tiện ích
│   ├── __init__.py                # Khởi tạo module utils
│   └── visualization.py           # Vẽ heatmap, learning curve và các biểu đồ
│
├── logs/                          # Lưu kết quả thực nghiệm
│   ├── training_metrics.json      # Reward, loss, tốc độ hội tụ
│   ├── system_metrics.json        # Thời gian chạy, RAM, CPU
│   └── experiment_summary.json    # Tổng hợp và so sánh kết quả
│
└── report/                        # Báo cáo LaTeX
    ├── main.tex                   # File LaTeX chính
    ├── [chapter].tex              # Các chương nhỏ (modular)
    ├── references.bib             # Tài liệu tham khảo
    ├── figures/                   # Thư mục chứa hình ảnh và biểu đồ
    └── report.pdf                 # Báo cáo hoàn chỉnh
```

---

## 4. Chương sách thực hiện

**Chương:** Reinforcement Learning (Chương 14)

**Tài liệu tham khảo:**
- **Tiêu đề:** Foundations of Machine Learning (2nd Edition)
- **Tác giả:** Mehryar Mohri, Afshin Rostamizadeh, Ameet Talwalkar
- **Nhà xuất bản:** MIT Press, 2018

---

## 5. Mô tả công việc và Điểm mở rộng

### 5.1. Nội dung thực hiện cơ bản

**Báo cáo lý thuyết:**
- Markov Decision Process (MDP)
- Value Function và Policy
- Phương trình Bellman
- Các thuật toán Planning và Learning

**Cài đặt thuật toán:**
- Value Iteration
- Policy Iteration
- Q-learning
- SARSA

**Thực nghiệm (from scratch):**
- Xây dựng môi trường Grid-world bằng Python và NumPy
- Cài đặt các thuật toán Planning (Value Iteration, Policy Iteration)
- Cài đặt các thuật toán Learning (Q-learning, SARSA)

**Đánh giá:**
- Tốc độ hội tụ của các thuật toán
- Độ ổn định và hiệu suất
- So sánh giữa các phương pháp

**Trực quan hóa:**
- Heatmap thể hiện Value Function
- Learning curve thể hiện quá trình học
- So sánh hiệu suất giữa các thuật toán

### 5.2. Điểm mở rộng

**Toán học:**
- Chứng minh chi tiết phương trình Bellman
- Bổ sung các bước trung gian trong chứng minh
- Phân tích sự hội tụ của các thuật toán

**Nghiên cứu tiên tiến:**
- Reinforcement Learning from Human Feedback (RLHF)
- Direct Preference Optimization (DPO)
- Ứng dụng trong các hệ thống học hiện đại

**Ứng dụng thực tiễn:**
- Hệ thống hỏi đáp pháp luật Việt Nam
- Cải thiện độ chính xác và khả năng suy luận
- Tích hợp với các mô hình ngôn ngữ lớn

---

## 6. Hướng dẫn tái tạo kết quả thực nghiệm

### 6.1. Yêu cầu hệ thống

- Python >= 3.10
- Các thư viện được liệt kê trong `requirements.txt`

### 6.2. Cài đặt môi trường

```bash
cd Reinforcement-Learning
python3 -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 6.3. Thực thi mã nguồn

```bash
python run_experiments.py
```

### 6.4. Kết quả đầu ra

**Biểu đồ và hình ảnh:**
- Lưu tại: `report/figures/`
- Bao gồm: heatmap, learning curve, so sánh thuật toán

**Log thực nghiệm:**
- `logs/training_metrics.json`: Reward, loss, tốc độ hội tụ
- `logs/system_metrics.json`: Thời gian chạy, sử dụng RAM, CPU
- `logs/experiment_summary.json`: Tổng hợp và so sánh kết quả

---

## 7. Ghi chú

- Sử dụng `random.seed()` và `np.random.seed()` để đảm bảo tính tái lập (reproducibility)
- Dữ liệu log được thiết kế để phân tích hiệu năng mô hình và trích xuất số liệu phục vụ báo cáo
- Tất cả các thuật toán được cài đặt từ đầu (from scratch) sử dụng NumPy
- Môi trường Grid-world được thiết kế linh hoạt để thử nghiệm với các cấu hình khác nhau

---
