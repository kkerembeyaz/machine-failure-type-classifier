from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent / "models"

# Scaler, eğitimde DataFrame ile fit edildiği için aynı sütun isimlerini kullanıyoruz
NUMERIC_COLS = ['Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']


class FailureClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 6)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = FailureClassifier()
    model.load_state_dict(torch.load(MODEL_PATH / "machine-failure-classifier.pth",
                                     map_location="cpu", weights_only=True))
    model.eval()
    artifacts["model"] = model
    # scaler (sklearn objesi) ve dict'ler pickle olarak kaydedildi -> weights_only=False gerekli
    artifacts["scaler"] = torch.load(MODEL_PATH / "scaler.pth", weights_only=False)
    artifacts["type_order"] = torch.load(MODEL_PATH / "type_order.pth", weights_only=False)
    artifacts["label_map"] = torch.load(MODEL_PATH / "label_map.pth", weights_only=False)
    artifacts["label_map_inv"] = torch.load(MODEL_PATH / "label_map_inv.pth", weights_only=False)
    yield
    artifacts.clear()


app = FastAPI(title="Machine Failure Type Classifier", lifespan=lifespan)


class MachineInput(BaseModel):
    Type: Literal["L", "M", "H"]
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: float
    Torque: float
    Tool_wear: float

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Type": "M",
                "Air_temperature": 298.1,
                "Process_temperature": 308.6,
                "Rotational_speed": 1551,
                "Torque": 42.8,
                "Tool_wear": 0,
            }]
        }
    }


@app.post("/predict")
def predict(data: MachineInput):
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    type_order = artifacts["type_order"]
    label_map_inv = artifacts["label_map_inv"]

    # 1) Type encode
    type_encoded = type_order[data.Type]

    # 2) Sayısal alanları scale et
    numeric_df = pd.DataFrame([[data.Air_temperature, data.Process_temperature,
                                data.Rotational_speed, data.Torque, data.Tool_wear]],
                              columns=NUMERIC_COLS)
    numeric_scaled = scaler.transform(numeric_df)[0]

    # 3) Sütun sırası: [Type, Air temp, Process temp, Rot speed, Torque, Tool wear]
    features = [float(type_encoded), *numeric_scaled.tolist()]
    x = torch.tensor([features], dtype=torch.float32)

    # 4) Tahmin
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        class_idx = int(torch.argmax(logits, dim=1).item())

    return {
        "prediction": label_map_inv[class_idx],
        "class_index": class_idx,
        "probabilities": {label_map_inv[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in artifacts}


INDEX_HTML = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML.read_text(encoding="utf-8")
