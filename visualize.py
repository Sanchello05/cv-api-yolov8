import requests
from PIL import Image, ImageDraw

API_URL = "http://127.0.0.1:8000/predict"
IMAGE_PATH = "examples/demo_from_dataset.jpg"

# 1. Отправляем изображение
with open(IMAGE_PATH, "rb") as f:
    response = requests.post(
        API_URL,
        files={"file": f},
        params={"conf": 0.4}
    )

data = response.json()

# 2. Открываем изображение
img = Image.open(IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(img)

# 3. Рисуем bbox
for det in data["detections"]:
    bbox = det["bbox"]
    label = f"{det['class_name']} {det['confidence']:.2f}"

    draw.rectangle(
        [(bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"])],
        outline="red",
        width=3
    )
    draw.text((bbox["x1"], bbox["y1"] - 10), label, fill="red")

# 4. Показываем
img.show()
