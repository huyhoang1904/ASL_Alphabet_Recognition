import torch
import torch.nn as nn
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
from collections import deque, Counter

class ASLNeuralNetwork(nn.Module):
    def __init__(self, input_size=42, num_classes=24): # Thay đổi num_classes thành số lượng nhãn thực tế
        super(ASLNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.network(x)


# ==========================================
# 1. THUẬT TOÁN LỌC NHIỄU
# ==========================================
class PredictionStabilizer:
    def __init__(self, window_size=20, threshold=15):
        self.window = deque(maxlen=window_size)
        self.threshold = threshold
        self.current_stable_prediction = "Waiting..."

    def reset(self):
        self.window.clear()
        self.current_stable_prediction = "Waiting..."

    def update(self, new_prediction):
        if new_prediction is None: return self.current_stable_prediction
        self.window.append(new_prediction)
        if len(self.window) < self.window.maxlen: return self.current_stable_prediction
        counts = Counter(self.window)
        most_common_label, count = counts.most_common(1)[0] # most_common(n): n tuple (label, count) có count lớn nhất
        if count >= self.threshold:
            self.current_stable_prediction = most_common_label
        return self.current_stable_prediction

# Theo dõi cử động tay để xử lý trường hợp ký hiệu động của J và Z
class DynamicGestureTracker:
    def __init__(self, buffer_size=25, movement_threshold=0.08):
        # Sliding window lưu vết 25 khung hình gần nhất
        self.pinky_buffer = deque(maxlen=buffer_size)  # Điểm 20 cho J
        self.index_buffer = deque(maxlen=buffer_size)  # Điểm 8 cho Z
        self.threshold = movement_threshold

    def update_and_check(self, landmarks, current_static_pred):
        # Lấy tọa độ tương đối (0-1) của đầu ngón út và ngón trỏ
        pinky_y = landmarks[20].y
        index_x, index_y = landmarks[8].x, landmarks[8].y

        self.pinky_buffer.append(pinky_y)
        self.index_buffer.append((index_x, index_y))

        # Đợi buffer đầy mới bắt đầu tính toán
        if len(self.pinky_buffer) < self.pinky_buffer.maxlen:
            return current_static_pred

        # ==================================
        # CHECK CHỮ J (Dáng I + Ngón út móc xuống)
        # ==================================
        if current_static_pred == 'I':
            # Tính độ lệch trục Y: Phần tử cuối - Phần tử đầu
            # Nếu tay di chuyển xuống, y sẽ tăng (trong OpenCV gốc tọa độ ở góc trên trái)
            dy = self.pinky_buffer[-1] - self.pinky_buffer[0]
            if dy > self.threshold:
                return 'J'

        # ==================================
        # CHECK CHỮ Z (Dáng D + Ngón trỏ vẽ zích zắc)
        # ==================================
        elif current_static_pred == 'D':
            # Chữ Z cơ bản: Điểm cuối nằm thấp hơn và lệch phải so với điểm đầu
            dy = self.index_buffer[-1][1] - self.index_buffer[0][1]
            dx = self.index_buffer[-1][0] - self.index_buffer[0][0]

            # Kiểm tra biên độ di chuyển (có thể tinh chỉnh threshold)
            if dy > self.threshold and abs(dx) > (self.threshold / 2):
                # Lưu ý: Nếu muốn cực kỳ chuẩn, bạn cần tính delta của 3 đoạn thẳng,
                # nhưng check delta tổng quát thế này là đủ tốt cho realtime
                return 'Z'

        return current_static_pred

# Xử lý ký tự và viết thành từ hoàn chỉnh
class TextBuilder:
    def __init__(self, space_timeout=45, cooldown_timeout=30):
        self.text = ""
        self.last_char = None
        self.empty_frames = 0 # biến đếm số lượng khung hình vắng mặt tay tối đa
        self.space_timeout = space_timeout # số lượng khung hình vắng mặt tay tối đa để hệ thống chèn 1 dấu cách
        self.space_added = False # biến lưu trạng thái thêm dấu cách

        # THỜI GIAN ĐÓNG BĂNG ---
        self.cooldown_timeout = cooldown_timeout  # Số frame bỏ qua sau khi gõ (thời gian chờ chuyển đổi cử chỉ)
        self.current_cooldown = 0 # biến đếm số frame trong thời gian chờ chuyển đổi cử chỉ

    def process(self, stable_char, hand_visible):
        # Trường hợp 1: Không có tay trong khung hình
        if not hand_visible:
            self.empty_frames += 1
            self.last_char = None

            # Vừa giấu tay là hủy đóng băng luôn để chuẩn bị gõ chữ mới
            self.current_cooldown = 0

            # Thêm dấu cách
            if self.empty_frames > self.space_timeout and not self.space_added:
                if len(self.text) > 0 and self.text[-1] != " ":
                    self.text += " "
                self.space_added = True

        # Trường hợp 2: Có tay trong khung hình
        else:
            self.empty_frames = 0
            self.space_added = False

            # --- KIỂM TRA ĐÓNG BĂNG ---
            if self.current_cooldown > 0:
                self.current_cooldown -= 1
                return self.text  # Bỏ qua mọi thao tác nếu đang bị đóng băng

            # Nếu hết đóng băng và có chữ mới ổn định
            if stable_char != "Waiting..." and stable_char != self.last_char:
                if stable_char == 'DEL':
                    self.text = self.text[:-1]
                else:
                    self.text += stable_char

                # Khóa chữ cái và BẬT ĐÓNG BĂNG
                self.last_char = stable_char
                self.current_cooldown = self.cooldown_timeout

        return self.text

# ==========================================
# 2. HÀM TỰ VẼ KHUNG XƯƠNG
# ==========================================
# Cấu trúc 21 điểm nối của bàn tay theo chuẩn MediaPipe
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),  # Ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12),  # Ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ngón áp út
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Ngón út
]


def draw_custom_landmarks(image, landmarks):
    h, w, _ = image.shape
    # Chuyển đổi từ tọa độ tỷ lệ (0-1) sang pixel thực tế
    pixel_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Vẽ các đường thẳng nối khớp xương (Màu xanh lá)
    for connection in HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        cv2.line(image, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (0, 255, 0), 2)

    # Vẽ các điểm nhấn tại từng khớp (Màu đỏ)
    for px, py in pixel_landmarks:
        cv2.circle(image, (px, py), 5, (0, 0, 255), -1)


# ==========================================
# 3. KHỞI TẠO TÀI NGUYÊN (AI & MODEL)
# ==========================================
# Load Label Encoder mới
le = joblib.load('label_encoder_nn.pkl')
num_classes = len(le.classes_)

# Khởi tạo khung mô hình và nạp trọng số
nn_model = ASLNeuralNetwork(input_size=42, num_classes=num_classes)
nn_model.load_state_dict(torch.load('asl_mlp_model_best_7.pth'))
nn_model.eval() # Bật chế độ suy luận (Tắt Dropout và BatchNorm)

stabilizer = PredictionStabilizer(window_size=25, threshold=20)
text_builder = TextBuilder(space_timeout=45, cooldown_timeout=30)
latest_prediction = "Waiting..."
latest_landmarks = None


# ==========================================
# 4. HÀM XỬ LÝ BẤT ĐỒNG BỘ (CHẠY NGẦM)
# ==========================================
def process_result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_prediction, latest_landmarks

    if not result.hand_landmarks:
        latest_landmarks = None
        stabilizer.reset()
        return

    latest_landmarks = result.hand_landmarks[0]

    # Chuẩn hóa toán học
    raw_coords = []
    for lm in latest_landmarks:
        raw_coords.extend([lm.x, lm.y])

    raw_coords = np.array(raw_coords)
    raw_coords[0::2] -= raw_coords[0]
    raw_coords[1::2] -= raw_coords[1]

    max_val = np.max(np.abs(raw_coords))
    if max_val != 0: raw_coords /= max_val

    # Dự đoán
    features = raw_coords.reshape(1, -1)
    input_tensor = torch.tensor(features, dtype=torch.float32)

    # Đưa vào mạng Nơ-ron
    with torch.no_grad():
        outputs = nn_model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)  # Lấy index có xác suất cao nhất
    # Giải mã số nguyên về chữ cái
    raw_letter = le.inverse_transform([predicted_idx.item()])[0]

    # Cập nhật prediction trực tiếp từ stabilizer
    stable_static = stabilizer.update(raw_letter)
    latest_prediction = stable_static

# ==========================================
# 5. CẤU HÌNH TASKS API (LIVE_STREAM)
# ==========================================
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    result_callback=process_result_callback
)

detector = vision.HandLandmarker.create_from_options(options)

# ==========================================
# 6. VÒNG LẶP CAMERA CHÍNH
# ==========================================
cap = cv2.VideoCapture(0)

print("=" * 50)
print("HỆ THỐNG ĐÃ SẴN SÀNG!")
print("Đưa tay vào camera để nhận diện. Bấm 'Q' để thoát.")
print("=" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    frame_timestamp_ms = int(time.time() * 1000)
    detector.detect_async(mp_image, frame_timestamp_ms)

    # --- CẬP NHẬT GIAO DIỆN ---
    hand_visible = latest_landmarks is not None
    current_sentence = text_builder.process(latest_prediction, latest_landmarks)

    if hand_visible:
        # Gọi hàm tự vẽ
        draw_custom_landmarks(frame, latest_landmarks)

        # Tính toán Bounding Box
        h, w, c = frame.shape
        x_min = int(min([lm.x for lm in latest_landmarks]) * w)
        y_min = int(min([lm.y for lm in latest_landmarks]) * h)
        x_max = int(max([lm.x for lm in latest_landmarks]) * w)
        y_max = int(max([lm.y for lm in latest_landmarks]) * h)

        # Vẽ viền và chữ dự đoán
        cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (255, 0, 0), 2)
        cv2.rectangle(frame, (x_min - 20, y_min - 60), (x_max + 20, y_min - 20), (255, 0, 0), -1)
        cv2.putText(frame, latest_prediction, (x_min, y_min - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    h, w, _ = frame.shape
    # Vẽ dải nền đen trong suốt ở dưới đáy
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # In chuỗi văn bản ra
    cv2.putText(frame, f"Text: {current_sentence}", (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('ASL Real-time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()