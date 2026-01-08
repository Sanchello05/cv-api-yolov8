import requests
import cv2
import json

API_URL = "http://127.0.0.1:8000/predict"
IMAGE_PATH = "client/demo_from_dataset.jpg"
OUTPUT_PATH = "client/result.jpg"

CONF = 0.4


def draw_boxes(image, detections):
    for det in detections:
        x1 = int(det["bbox"]["x1"])
        y1 = int(det["bbox"]["y1"])
        x2 = int(det["bbox"]["x2"])
        y2 = int(det["bbox"]["y2"])

        label = f'{det["class_name"]} {det["confidence"]:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    return image


def main():
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": f}
        params = {"conf": CONF}

        response = requests.post(API_URL, files=files, params=params)

    if response.status_code != 200:
        print("Request failed:", response.text)
        return

    data = response.json()
    print(json.dumps(data, indent=2))

    image = cv2.imread(IMAGE_PATH)
    image = draw_boxes(image, data["detections"])

    cv2.imwrite(OUTPUT_PATH, image)
    print("Result saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
