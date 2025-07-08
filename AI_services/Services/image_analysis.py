from fastapi import UploadFile
from AI_services.DTO.ProductFilter import ImageResponse
from PIL import Image
import torch
import clip
import io


#Use CPU if GPU is not available
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#Criterias 
#General
genders = [
    "men", "women", "unisex", "male", "female",
    "boy", "girl", "child", "adult", "teen"
]


styles = [
    "classic", "elegant", "modern", "vintage", "minimalist",
    "boho", "baroque", "romantic", "futuristic", "casual"
]

description_candidates = [
    "luxurious", "handcrafted", "delicate", "bold", "unique",
    "minimalistic", "vintage-inspired", "modern design", "elegant finish", "high quality"
]

#For metals
metals = [
    "gold", "silver", "platinum", "rose gold", "white gold",
    "titanium", "steel", "bronze", "copper", "palladium"
]

metal_shapes = [
    "round", "oval", "square", "rectangle", "heart",
    "twisted", "geometric", "triangular", "hexagonal", "octagonal"
]

metal_colors = [
    "yellow", "white", "rose", "black", "blue",
    "green", "purple", "red", "pink", "orange"
]

metal_sizes = [
    "Small", "Medium", "Large"
]

metal_types = [
    "solid", "plated", "mixed", "brushed", "polished",
    "matte", "textured", "engraved", "hammered", "oxidized"
]

#For stones
stones = [
    "diamond", "emerald", "sapphire", "ruby", "amethyst",
    "topaz", "pearl", "turquoise", "garnet", "opal"
]

stone_shapes = [
    "round", "princess", "emerald", "oval", "marquise",
    "pear", "heart", "cushion", "asscher", "brilliant"
]

stone_colors = [
    "white", "blue", "red", "green", "purple",
    "pink", "yellow", "black", "brown", "orange"
]

stone_sizes = [
    "Tiny", "Regular", "Big"
]

stone_types = [
    "precious", "semi-precious", "synthetic", "natural", "treated",
    "untreated", "raw", "cut", "faceted", "cabochon"
]


#Categories in CLIP tokens
#General
genders_tokens = clip.tokenize(genders).to(device)
styles_tokens = clip.tokenize(styles).to(device)
description_tokens = clip.tokenize(genders).to(device)
#Metals
metals_tokens = clip.tokenize(metals).to(device)
metal_shapes_tokens = clip.tokenize(metal_shapes).to(device)
metal_colors_tokens = clip.tokenize(metal_colors).to(device)
metal_sizes_tokens = clip.tokenize(metal_sizes).to(device)
metal_types_tokens = clip.tokenize(metal_types).to(device)
#Stones
stones_tokens = clip.tokenize(stones).to(device)
stone_shapes_tokens = clip.tokenize(stone_shapes).to(device)
stone_colors_tokens = clip.tokenize(stone_colors).to(device)
stone_sizes_tokens = clip.tokenize(stone_sizes).to(device)
stone_types_tokens = clip.tokenize(stone_types).to(device)


# TO DO-do-do do do~
# Comparison of image into criterias
image_input = preprocess(image).unsqueeze(0).to(device)
image_features = model.encode_image(image_input)

# Probabilities for each criteria
style_probs = (image_features @ model.encode_text(style_tokens).T).softmax(dim=-1)
metal_probs = (image_features @ model.encode_text(metal_tokens).T).softmax(dim=-1)
stone_probs = (image_features @ model.encode_text(stone_tokens).T).softmax(dim=-1)

# Select top results
top_styles = [styles[i] for i in style_probs[0].topk(2).indices]
top_metals = [metals[i] for i in metal_probs[0].topk(2).indices]
top_stones = [stones[i] for i in stone_probs[0].topk(2).indices]



async def process_image(file: UploadFile) -> ImageResponse:
    contents = await file.read()                #Get bytes from request
    image = Image.open(io.BytesIO(contents))    #Convert bytes in pillow image

    # Заглушка 
    # TODO: внедрить CLIP/анализ
    styles = ["classic", "elegant"]
    metals = ["gold", "silver"]
    stones = ["diamond"]

    return ImageResponse(styles=styles, metals=metals, stones=stones)
