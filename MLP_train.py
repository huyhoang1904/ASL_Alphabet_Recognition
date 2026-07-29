import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


# ==========================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC MẠNG NƠ-RON (MLP)
# ==========================================
class ASLNeuralNetwork(nn.Module):
    def __init__(self, input_size=42, num_classes=24):
        super(ASLNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),  # Giảm overfitting

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, num_classes)
            # Không dùng Softmax ở đây vì hàm CrossEntropyLoss của PyTorch đã tự bao gồm
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 2. XỬ LÝ DỮ LIỆU (DATASET)
# ==========================================
class ASLDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ==========================================
# 3. QUY TRÌNH CHUẨN BỊ VÀ HUẤN LUYỆN
# ==========================================
if __name__ == "__main__":
    DATA_FILE = "train_data_asl_clone.csv"

    print("1. Đang tải và tiền xử lý dữ liệu...")
    df = pd.read_csv(DATA_FILE, header=None)

    # Tách nhãn (cột 0) và đặc trưng (các cột còn lại)
    X = df.iloc[:, 1:].values
    y_raw = df.iloc[:, 0].values

    # Mã hóa nhãn chữ cái thành số nguyên (0, 1, 2...)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    # Lưu bộ mã hóa để dùng khi test
    joblib.dump(le, 'label_encoder_nn.pkl')
    print(f"Đã lưu LabelEncoder. Nhận diện {num_classes} lớp chữ cái.\n")

    # Chia tập train/val (80% train, 20% validate)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = ASLDataset(X_train, y_train)
    val_dataset = ASLDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print("2. Đang khởi tạo mô hình PyTorch...")
    model = ASLNeuralNetwork(input_size=42, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("3. Bắt đầu huấn luyện...")
    EPOCHS = 30
    best_val_loss = float('inf')  # Khởi tạo mức loss tốt nhất là vô cực

    for epoch in range(EPOCHS):
        # --- QUÁ TRÌNH HUẤN LUYỆN ---
        model.train()
        train_loss = 0
        train_correct = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / len(train_dataset)

        # --- QUÁ TRÌNH ĐÁNH GIÁ (VALIDATION) ---
        model.eval()
        val_loss = 0
        val_correct = 0

        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)

                # Tính Loss cho tập validation
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / len(val_dataset)

        # In thông số của epoch hiện tại
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # --- LƯU MÔ HÌNH TỐT NHẤT ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'asl_mlp_model_best_4.pth')
            print(f"   -> [Cập nhật] Đã lưu mô hình tốt nhất với Val Loss: {best_val_loss:.4f}")

    print("\n4. Hoàn tất huấn luyện!")
    print("Mô hình hoạt động tốt nhất (chưa bị overfitting) đã được lưu tại 'asl_mlp_model_best_4.pth'")