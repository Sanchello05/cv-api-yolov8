from fastapi import FastAPI, UploadFile, File, HTTPException, Query

from ultralytics import YOLO

from app.schemas import PredictResponse
from app.services.inference import run_inference
from app.utils.image import load_image

# --------------------
# Инициализация приложения
# --------------------
app = FastAPI(
    title="CV Object Detection API",
    description="YOLO-based object detection service",
    version="1.0.0",
)

# --------------------
# Загрузка модели (1 раз при старте)
# --------------------
model = YOLO("models/best.pt")

# --------------------
# Константы
# --------------------
MAX_SIZE_MB = 5


# --------------------
# Health-check (обязателен в продакшене)
# --------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------
# Основной endpoint
# --------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(
        0.4,
        ge=0.0,
        le=1.0,
        description="Confidence threshold (0..1)"
    ),
):
    # 1. Читаем файл
    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(image_bytes) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    # 2. Загружаем изображение
    try:
        image = load_image(image_bytes)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image")

    # 3. Инференс
    try:
        detections = run_inference(model, image, conf)
    except Exception:
        raise HTTPException(status_code=500, detail="Inference failed")

    # 4. Ответ
    return {
        "image": {
            "width": image.width,
            "height": image.height
        },
        "detections": detections
    }

