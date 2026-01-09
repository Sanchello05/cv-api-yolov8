# CV API — Object Detection (YOLOv8 + FastAPI)

Inference-сервис для детекции объектов на изображениях.
Модель обучена на дорожных сценах малого и среднего потока автомобилей.

## Стек
- Python
- YOLOv8 (Ultralytics)
- FastAPI
- Docker

## Запуск через Docker
```bash
docker build -t cv-api .
docker run -p 8000:8000 --name cv-api-run cv-api

## API

### POST /predict
**Вход**
- Файл изображения (`jpg` / `png`)
- Query-параметр: `conf` — порог уверенности (float)

**Выход**
- JSON со списком детекций

Пример ответа:
```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "car",
      "confidence": 0.71,
      "bbox": {
        "x1": 120,
        "y1": 80,
        "x2": 300,
        "y2": 260
      }
    }
  ]
}
