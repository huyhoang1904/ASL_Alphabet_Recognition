from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib

app = FastAPI(title="ASL Dynamic Recognition")
templates = Jinja2Templates(directory="templates")


class ASLNeuralNetwork(nn.Module):
    def __init__(self, input_size=42, num_classes=24):
        super(ASLNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# Load Model
le = joblib.load('label_encoder_nn.pkl')
model = ASLNeuralNetwork(input_size=42, num_classes=len(le.classes_))
model.load_state_dict(torch.load('asl_mlp_model_best_3.pth', map_location='cpu'))
model.eval()


class HandData(BaseModel):
    features: list[float]


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict_sign(data: HandData):
    try:
        # 1. Nhận 42 tọa độ thô từ MediaPipe JS
        raw_coords = np.array(data.features, dtype=np.float32)

        # 2. Chuẩn hóa toán học y hệt bản Python của bạn
        raw_coords[0::2] -= raw_coords[0]
        raw_coords[1::2] -= raw_coords[1]
        max_val = np.max(np.abs(raw_coords))
        if max_val != 0:
            raw_coords /= max_val

        # 3. Đưa vào Neural Network
        input_tensor = torch.tensor(raw_coords).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            max_prob, predicted_idx = torch.max(torch.softmax(outputs, dim=1), 1)

        raw_letter = le.inverse_transform([predicted_idx.item()])[0]

        return {"prediction": raw_letter, "confidence": max_prob.item()}
    except Exception as e:
        return {"error": str(e)}