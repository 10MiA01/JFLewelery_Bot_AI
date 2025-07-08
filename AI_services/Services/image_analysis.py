from fastapi import UploadFile
from AI_services.DTO.ProductFilter import ImageResponse
from PIL import Image
import io

async def process_image(file: UploadFile) -> ImageResponse:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Заглушка — вместо неё будет запуск модели
    # TODO: внедрить CLIP/анализ
    styles = ["classic", "elegant"]
    metals = ["gold", "silver"]
    stones = ["diamond"]

    return ImageResponse(styles=styles, metals=metals, stones=stones)
