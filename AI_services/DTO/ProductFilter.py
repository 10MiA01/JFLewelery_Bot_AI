
from pydantic import BaseModel
from typing import List

class ImageResponse(BaseModel):
    styles: List[str]
    metals: List[str]
    stones: List[str]

