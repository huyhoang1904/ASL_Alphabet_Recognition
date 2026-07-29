# 🤟 ASL Real-time Translation System

Dự án Nhận diện Ngôn ngữ Ký hiệu Mỹ (ASL) sử dụng MediaPipe và Mạng Neural (MLP) với PyTorch. Dự án hỗ trợ huấn luyện mô hình, kiểm thử trực tiếp qua Webcam và triển khai trên nền tảng Web thông qua FastAPI.

## 🛠 Yêu cầu hệ thống
- **Ngôn ngữ:** Python 3.11
- **Phần cứng:** Webcam (để test trực tiếp) và kết nối Internet (để tải thư viện Frontend).

## 📦 Cài đặt thư viện

Mở Terminal và chạy lệnh sau để cài đặt toàn bộ các package cần thiết cho dự án:

```bash
pip install numpy pandas opencv-python torch torchvision mediapipe joblib jinja2 pydantic fastapi uvicorn
```

## 📁 Cấu trúc thư mục & Ý nghĩa các file

### 1. Dữ liệu (Data Files)
- `train_data_asl (1).csv`: File dữ liệu gốc (chưa xử lý thay thế các nhãn bị lỗi/kém chất lượng như T, M, N, K, S).
- `new_data.csv`: File dữ liệu mới được trích xuất riêng cho các nhãn T, M, N, K, S để khắc phục.
- `train_data_asl_clone.csv`: File dữ liệu hoàn chỉnh sau khi đã tiến hành gộp và thay thế dữ liệu mới. (Đây là file dùng để train).

### 2. Mã nguồn chạy (Script Files)
- `replace_train_data.py`: Script dùng để chạy logic thay thế dữ liệu từ `new_data.csv` vào file gốc, xuất ra file `train_data_asl_clone.csv`.
- `MLP_train.py`: Chứa mã nguồn để huấn luyện mô hình mạng Nơ-ron (Neural Network) dựa trên tập dữ liệu đã làm sạch.
- `test.py`: Kịch bản kiểm thử mô hình dự đoán trực tiếp (Real-time) trên cửa sổ Webcam bằng thư viện OpenCV.

### 3. Mô hình đã huấn luyện (Model & Encoder)
- `asl_mlp_model_best_3.pth`: Trọng số của mô hình mạng Nơ-ron hoạt động tốt nhất.
- `label_encoder_nn.pkl`: File lưu trữ bộ chuyển đổi nhãn (Label Encoder) để ánh xạ kết quả số về dạng chữ cái (A-Z).

### 4. Triển khai Web (Web App Files)
- `app.py`: File Backend chính sử dụng FastAPI để xử lý API và render giao diện.
- `templates/`: Thư mục chứa file `index.html` (Frontend giao diện web).

---

## 🚀 Hướng dẫn sử dụng

### Chạy kiểm thử trên nền tảng Web (Khuyên dùng)
Giao diện Web cung cấp trải nghiệm mượt mà và đã được tích hợp bộ lọc ổn định khung hình.

1. Mở Terminal (trong PyCharm hoặc cmd) tại thư mục chứa dự án.
2. Khởi chạy Web Server bằng lệnh:
   ```bash
   uvicorn app:app --reload
   ```
3. Mở trình duyệt Web (Chrome, Edge...) và truy cập vào đường dẫn: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
4. Bấm **Cho phép (Allow)** khi trình duyệt yêu cầu quyền truy cập Camera. Đợi vài giây để mô hình tải và bạn có thể bắt đầu sử dụng!

