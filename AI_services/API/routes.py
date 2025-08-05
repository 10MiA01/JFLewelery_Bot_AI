from fastapi import APIRouter, File, UploadFile, Form
from AI_services.Services.image_analysis import process_image_product_filter
from AI_services.Services.tryon import process_image_try_on
from AI_services.DTO.ProductFilter import ImageResponse
import traceback
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
import traceback


router = APIRouter()

@router.post("/analyze-image", response_model=ImageResponse)
async def analyze_image(file: UploadFile = File(...)):
     try:
        result = await process_image_product_filter(file)
        print("[DEBUG] Result:")
        print(result)
        return result
     except Exception as e:
        print("[ERROR] Exception occurred in analyze_image")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/virtual-fitting")
async def virtual_fitting(file: UploadFile = File(...), category: str = Form(...), id: int = Form(...)):
    try:
        result_image_bytes = await process_image_try_on(file, category, id)  
        return StreamingResponse(
            content=BytesIO(result_image_bytes),
            media_type="image/png"  
        )
    except Exception as e:
        print("[ERROR] Exception occurred in virtual_fitting")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
