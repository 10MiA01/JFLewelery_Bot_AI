from fastapi import APIRouter, File, UploadFile
from AI_services.Services.image_analysis import process_image
from AI_services.DTO.ProductFilter import ImageResponse

router = APIRouter()

@router.post("/analyze-image", response_model=ImageResponse)
async def analyze_image(file: UploadFile = File(...)):
    result = await process_image(file)
    return result
