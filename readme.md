# ARSK-Clustering

[cite_start]ARSK-Clustering là đồ án môn học Khai thác dữ liệu và ứng dụng (CSC14004) - Đồ án 3: Phân cụm và Ứng dụng[cite: 58]. Dự án này tái hiện, phân tích và mở rộng bài báo **"Adaptively Robust and Sparse K-means Clustering"** (Transactions on Machine Learning Research, 11/2024).

Dự án hiện thực phương pháp phân cụm ARSK nhằm giải quyết đồng thời hai hạn chế lớn của K-means cổ điển:
- **Kháng nhiễu (Robustness):** Thêm ma trận lỗi (error matrix) để hấp thụ các điểm dị biệt, được kiểm soát bởi các hàm phạt nhóm (Group Lasso/SCAD).
- **Tính thưa (Sparsity):** Gán trọng số cho từng đặc trưng và ép các biến nhiễu về 0 thông qua hàm phạt $L_1$/SCAD để giảm chiều dữ liệu.
- **Tự động tinh chỉnh (Adaptive Tuning):** Đề xuất hàm *Robust Gap Statistics* để tự động chọn các siêu tham số phạt tối ưu.

Đồng thời lưu trữ bài làm của nhóm và các tài liệu liên quan tại repo này.

## 1. Thông tin nhóm

| STT | Họ và tên | MSSV |
| --- | --- | --- |
| 1 | Nguyễn Hữu Anh Trí | 23127130 |
| 2 | Cao Trần Bá Đạt | 23127168 |
| 3 | Tô Trần Hoàng Triệu | 23127133 |
| 4 | Cao Tấn Hoàng Huy | 23127051 |

## 2. Mục tiêu đồ án
- **Phần A (Phân tích bài báo):** Đọc hiểu, tóm tắt và phê bình phương pháp ARSK.
- **Phần B (Thực nghiệm):**
  - **Tái hiện thực nghiệm:** Lập trình lại thuật toán (re-implementation) và tái hiện Simulation 1 (so sánh 4 cấu hình hàm ngưỡng SCAD/Soft) từ bài báo gốc.
  - **Ablation Study tự thiết kế:** Cô lập và đánh giá độ quan trọng của module *Robust Gap Statistics* bằng cách so sánh đối đầu với *Standard Gap Statistics*.
  - [cite_start]**Thực nghiệm dữ liệu mới:** Áp dụng ARSK lên tập dữ liệu thực tế `Wine Quality` (không có trong bài báo gốc) để đánh giá khả năng tổng quát hóa, phân tích tính thưa và chỉ ra hiện tượng quá khớp (over-correction) trên dữ liệu sạch[cite: 58].

## 3. Cấu trúc thư mục

```text
.
├── docs/           # Lưu trữ kết quả thực nghiệm xuất ra (file CSV) và báo cáo
├── notebooks/      # Chứa các kịch bản chạy thực nghiệm và đánh giá
│   ├── 01_main_experiments.ipynb           # Tái hiện thực nghiệm gốc (Simulation 1)
│   ├── 02_ablation.ipynb                   # Thực nghiệm Ablation: Robust Gap vs Standard Gap
│   ├── 03_experiment_with_new_dataset.ipynb # Chạy ARSK trên tập dữ liệu mới
├── paper/          # Chứa file PDF đề bài lab và bài báo gốc TMLR
├── src/            # Mã nguồn lõi của thuật toán ARSK
│   ├── metrics.py      # Hàm tính toán độ đo (vd: Clustering Error Rate - CER)
│   ├── model.py        # Cài đặt lại mô hình của bài báo
│   └── utils.py        # Trình sinh dữ liệu mô phỏng phân phối nhiễu (Data Generator)
└── README.md       # Tài liệu hướng dẫn project
```

## 4. Cài đặt môi trường

Mã nguồn được phát triển trên Python 3.9+. Để chạy được các thực nghiệm, cần cài đặt các thư viện toán học và học máy cơ bản. Mở terminal và chạy lệnh:

```bash
pip install numpy pandas scikit-learn scipy matplotlib jupyter
```

## 5. Hướng dẫn chạy thực nghiệm

Dự án sử dụng Jupyter Notebook để dễ dàng tương tác và trực quan hóa kết quả. Mã nguồn lõi đã được đóng gói gọn gàng trong thư mục `src/`.

Để chạy các thực nghiệm, khởi động Jupyter và mở các file trong thư mục `notebooks/` theo thứ tự:

1. **Tái hiện Simulation 1:** Chạy file `notebooks/01_main_experiments.ipynb`. Script sẽ sinh dữ liệu mô phỏng với mức độ nhiễu $\pi \in \{0, 0.1, 0.2, 0.3\}$ và chạy ARSK qua 4 cấu hình hàm ngưỡng để báo cáo CER, số outlier, số feature.
2. **Ablation Study:** Chạy file `notebooks/02_ablation.ipynb`. Script này chạy song song 2 nhánh: sử dụng *Robust Gap* và *Standard Gap*, sau đó trích xuất ra các file tóm tắt `ablation_*.csv` vào thư mục `docs/`.
3. **Thử nghiệm dữ liệu mới (Wine Quality):** Chạy file `notebooks/03_experiment_with_new_dataset.ipynb` và `04_wine_new_dataset_10runs.ipynb`. Pipeline sẽ tải tập Wine, tự động tuning $\lambda_1, \lambda_2$, chạy ARSK, so sánh CER với thuật toán KMeans truyền thống, và in ra danh sách các trọng số biến (W) bị ép về 0.

## 6. Ghi chú thực thi và Phân tích nhanh

- Hàm `arsk` trong `src/model.py` đã được nhóm cài đặt lại hoàn chỉnh với các điều kiện dừng (`tol`, `max_iter`) và hàm kẹp (`np.clip`) để tránh lỗi phân kỳ hoặc tràn số từng tồn tại trong code gốc của tác giả.
- **Kết quả Ablation:** Khi tỷ lệ nhiễu $\ge 0.2$, Standard Gap bị sụp đổ hoàn toàn trong việc chọn $\lambda_1, \lambda_2$, trong khi Robust Gap duy trì độ chính xác cao. Điều này chứng minh khâu Tuning cũng cần phải được thiết kế kháng nhiễu đồng bộ với hàm mục tiêu.
- **Kết quả Dữ liệu mới:** Thuật toán ARSK tỏ ra yếu thế hơn KMeans truyền thống trên tập Wine Quality (dữ liệu sạch, tuyến tính). ARSK nhận diện nhầm tới ~72% dữ liệu là outlier (over-correction). Điều này cung cấp một góc nhìn phản biện: ARSK chỉ thực sự mạnh trên không gian dữ liệu cực kỳ đa chiều và nhiễu nặng, không nên lạm dụng cho các tập dữ liệu sạch, cơ bản.