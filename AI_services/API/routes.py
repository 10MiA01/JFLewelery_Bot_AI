from fastapi import APIRouter, File, UploadFile
from AI_services.Services.image_analysis import process_image
from AI_services.DTO.ProductFilter import ImageResponse
import traceback
from fastapi import HTTPException


router = APIRouter()

@router.post("/analyze-image", response_model=ImageResponse)
async def analyze_image(file: UploadFile = File(...)):
     try:
        result = await process_image(file)
        print("[DEBUG] Result:")
        print(result)
        return result
     except Exception as e:
        print("[ERROR] Exception occurred in analyze_image")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
