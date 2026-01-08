def run_inference(model, image, conf: float):
    """
    Запускает модель и возвращает список детекций
    """
    results = model(image, conf=0.4)
    r = results[0]

    detections = []

    for box, score, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
        detections.append({
            "class_id": int(cls),
            "class_name": r.names[int(cls)],
            "confidence": float(score),
            "bbox": {
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
            }
        })

    return detections
