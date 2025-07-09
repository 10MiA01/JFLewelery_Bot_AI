from fastapi import UploadFile
from fastapi import HTTPException
from AI_services.DTO.ProductFilter import ImageResponse
from PIL import Image
import torch
import clip
import io






async def process_image(file: UploadFile) -> ImageResponse:
    try:
        contents = await file.read()                #Get bytes from request
        image = Image.open(io.BytesIO(contents))    #Convert bytes in pillow image

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


        # Comparison of image into criterias
        image_input = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image_input)

        # Probabilities for each criteria
        #General 
        genders_probs = (image_features @ model.encode_text(genders_tokens).T).softmax(dim=-1)
        styles_probs = (image_features @ model.encode_text(styles_tokens).T).softmax(dim=-1)
        description_probs = (image_features @ model.encode_text(description_tokens).T).softmax(dim=-1)

        #Metals
        metals_probs = (image_features @ model.encode_text(metals_tokens).T).softmax(dim=-1)
        metal_shapes_probs = (image_features @ model.encode_text(metal_shapes_tokens).T).softmax(dim=-1)
        metal_colors_probs = (image_features @ model.encode_text(metal_colors_tokens).T).softmax(dim=-1)
        metal_sizes_probs = (image_features @ model.encode_text(metal_sizes_tokens).T).softmax(dim=-1)
        metal_types_probs = (image_features @ model.encode_text(metal_types_tokens).T).softmax(dim=-1)

        #Stones
        stones_probs = (image_features @ model.encode_text(stones_tokens).T).softmax(dim=-1)
        stone_shapes_probs = (image_features @ model.encode_text(stone_shapes_tokens).T).softmax(dim=-1)
        stone_colors_probs = (image_features @ model.encode_text(stone_colors_tokens).T).softmax(dim=-1)
        stone_sizes_probs = (image_features @ model.encode_text(stone_sizes_tokens).T).softmax(dim=-1)
        stone_types_probs = (image_features @ model.encode_text(stone_types_tokens).T).softmax(dim=-1)


        # Select top results
        #General
        top_ganders = [genders[i] for i in genders_probs[0].topk(1).indices]
        gandersn_str = ", ".join(top_ganders)

        top_styles = [styles[i] for i in styles_probs[0].topk(2).indices]

        top_description = [description_candidates[i] for i in description_probs[0].topk(2).indices]
        description_str = ", ".join(top_description)

        #Metals
        top_metals = [metals[i] for i in metals_probs[0].topk(2).indices]
        top_metal_shapes = [metal_shapes[i] for i in metal_shapes_probs[0].topk(2).indices]
        top_metal_colors = [metal_colors[i] for i in metal_colors_probs[0].topk(2).indices]
        top_metal_sizes = [metal_sizes[i] for i in metal_sizes_probs[0].topk(2).indices]
        top_metal_types = [metal_types[i] for i in metal_types_probs[0].topk(2).indices]

        #Stones
        top_stones = [stones[i] for i in stones_probs[0].topk(2).indices]
        top_stone_shapes = [stone_shapes[i] for i in stone_shapes_probs[0].topk(2).indices]
        top_stone_colors = [stone_colors[i] for i in stone_colors_probs[0].topk(2).indices]
        top_stone_sizes = [stone_sizes[i] for i in stone_sizes_probs[0].topk(2).indices]
        top_stone_types = [stone_types[i] for i in stone_types_probs[0].topk(2).indices]


       #return JSON with criterias

        return ImageResponse(
            # General
            gender = gandersn_str,
            styles = top_styles,
            description = description_str,

            # For Metals
            metals = top_metals,
            metal_shapes = top_metal_shapes,
            metal_colors = top_metal_colors,
            metal_sizes = top_metal_sizes,
            metal_types = top_metal_types,


            # For Stones
            stones = top_stones,
            stone_shapes = top_stone_shapes,
            stone_colors = top_stone_colors,
            stone_sizes = top_stone_sizes,
            )
    except Exception as e:
        print("ERROR in process_image:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
