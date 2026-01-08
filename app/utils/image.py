from PIL import Image, UnidentifiedImageError
from io import BytesIO


def load_image(image_bytes: bytes) -> Image.Image:
    """
    Загружает изображение из байтов и приводит к RGB
    """
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Invalid image")
