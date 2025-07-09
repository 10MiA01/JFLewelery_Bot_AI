
from pydantic import BaseModel
from typing import List, Optional

class ImageResponse(BaseModel):
    # General
    gender: Optional[str] = None
    styles: Optional[List[str]] = None
    description: Optional[str] = None

    # For Metals
    metals: Optional[List[str]] = None
    metal_shapes: Optional[List[str]] = None
    metal_colors: Optional[List[str]] = None
    metal_sizes: Optional[List[str]] = None
    metal_types: Optional[List[str]] = None


    # For Stones
    stones: Optional[List[str]] = None
    stone_shapes: Optional[List[str]] = None
    stone_colors: Optional[List[str]] = None
    stone_sizes: Optional[List[str]] = None

    

