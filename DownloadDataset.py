import kagglehub

# Download latest version
path = kagglehub.dataset_download("boukraailyesali/traffic-road-object-detection-dataset-using-yolo")

print("Path to dataset files:", path)