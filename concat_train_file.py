import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. CẤU HÌNH
DATA_ROOT = "custom_dataset"  # Thư mục cha chứa các thư mục con cần bổ sung
CSV_FILE = "train_data_asl (1).csv"
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

new_rows = []

# 2. QUÉT TOÀN BỘ THƯ MỤC ẢNH
for label in os.listdir(DATA_ROOT):
    label_path = os.path.join(DATA_ROOT, label)
    if not os.path.isdir(label_path): continue

    print(f"Đang xử lý chữ cái: {label}...")
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)
        if image is None: continue

        # Trích xuất landmarks (giống hệt logic lúc train)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            coords = []
            for lm in landmarks: coords.extend([lm.x, lm.y])

            # Chuẩn hóa
            coords = np.array(coords)
            coords[0::2] -= coords[0]
            coords[1::2] -= coords[1]
            max_val = np.max(np.abs(coords))
            if max_val != 0: coords /= max_val

            new_rows.append([label] + coords.tolist())

# 3. NỐI VÀO CSV GỐC
if new_rows:
    df_new = pd.DataFrame(new_rows)  # Giả sử file cũ đã có header, dùng header=False
    df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)
    print(f"Đã nối thành công {len(new_rows)} dòng dữ liệu mới vào {CSV_FILE}")