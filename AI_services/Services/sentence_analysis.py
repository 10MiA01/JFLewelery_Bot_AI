from fastapi import UploadFile
from fastapi import HTTPException
from AI_services.DTO.ProductFilter import ImageResponse
import AI_services.DTO.CriteriaForFilter as criteria
from PIL import Image
import torch
import clip
import io
import json
from fastapi.encoders import jsonable_encoder



async def process_image_product_filter(sentence: str) -> ImageResponse:
    try:

        #Use CPU if GPU is not available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        # Get sentence

        # Make embading from sentence

        # make embedings for filter criterias

        # compare embedings and choose criterias
        

        #return JSON with criterias

        # result = ImageResponse(
        #     # General
        #     gender=gandersn_str,
        #     styles=top_styles,
        #     description=description_str,

        #     # For Metals
        #     metals=top_metals,
        #     metal_shapes=top_metal_shapes,
        #     metal_colors=top_metal_colors,
        #     metal_sizes=top_metal_sizes,
        #     metal_types=top_metal_types,

        #     # For Stones
        #     stones=top_stones,
        #     stone_shapes=top_stone_shapes,
        #     stone_colors=top_stone_colors,
        #     stone_sizes=top_stone_sizes,
        # )

        print("[DEBUG] ImageResponse:")
        print(json.dumps(jsonable_encoder(result), indent=2, ensure_ascii=False))

        return result

    except Exception as e:
        print("ERROR in process_image:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
