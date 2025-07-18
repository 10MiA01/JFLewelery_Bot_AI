﻿from fastapi import APIRouter, File, UploadFile, Form
from PIL import Image
from io import BytesIO
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import math
from pathlib import Path
from Helpers import CategoryTryOnDict

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    # PIL (RGB) -> NumPy array (BGR)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def find_group_by_category(category):
    for group, categories in CategoryTryOnDict.items():
        if category in categories:
            return group
        return "Not found a group"

def analyze_image_parts(cv2_image):
    results = {}

    # Pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        pose_results = pose.process(cv2_image)
        results['pose'] = pose_results.pose_landmarks

    # Hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        hands_results = hands.process(cv2_image)
        results['hands'] = hands_results.multi_hand_landmarks

    # Face mesh
    with mp_face.FaceMesh(static_image_mode=True) as face:
        face_results = face.process(cv2_image)
        results['face'] = face_results.multi_face_landmarks

    return results

def is_part_present(results, category_group):
    # body
    if category_group == 'body' :
        return results['pose'] is not None
    # hands
    if category_group == 'hands':
        return results['hands'] is not None
    # face
    if category_group == 'body':
        return results['pose'] is not None

def get_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def get_scale(p1, p2, base_length):
    """
    base_length — ref size of img in pixels
    """
    dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    scale = dist / base_length
    return scale


def rotate_and_scale_image(img, angle, scale):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)    # Matrix for 2D rotate and scale
    rotated_scaled = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)  # Apply the matrix
    return rotated_scaled

def overlay_image_alpha(background, overlay, x, y):
    """
    background — base image (OpenCV, BGR)
    overlay — PNG image (BGRA) with transparency
    x, y — coordinates of centre for paste - counted from landmarks
    """
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    # Paste coordinates
    x1 = max(x - ow // 2, 0)
    y1 = max(y - oh // 2, 0)
    x2 = min(x1 + ow, bw)
    y2 = min(y1 + oh, bh)

    # Crop for neded area
    overlay_crop = overlay[0:(y2 - y1), 0:(x2 - x1)] # Slice of numpy 2D array

    # if nothing to paste => return just a photo
    if overlay_crop.shape[0] == 0 or overlay_crop.shape[1] == 0:
        return background

    # Get channels from pixels arrays => splits a multi-channel image (in this case 4-channel) into 2D arrays
    b, g, r, a = cv2.split(overlay_crop)

    # Normalize alpha channel to range from 0 to 1
    alpha = a / 255.0

    # apply overlay pixels to bachground 
    for c, color in enumerate([b, g, r]):
        background[y1:y2, x1:x2, c] = (
            alpha * color + (1 - alpha) * background[y1:y2, x1:x2, c]
        )

    return background



# To Do sketch
async def process_image(file: UploadFile, category: str, id: int):

    # Get the image from client
    image = Image.open(file.file).convert("RGB")
    cv2_image = pil_to_cv2(image)

    #Get the inamge of product by id
    product_path = Path(__file__).parent.parent / "JFJewelery" / "Media" / "images" / "products" / str(id) / "tryon.png"
    overlay_image = cv2.imread(str(product_path), cv2.IMREAD_UNCHANGED)

    # Declare the output
    output_image = None

    # Get the category
    selected_group = find_group_by_category(category)

    # General pose analysis
    results = analyze_image_parts(cv2_image)

    if not is_part_present(results, selected_group):
        raise ValueError(f"Photo nit suitable for a category {category}, часть тела не найдена")

    # face
    if selected_group == "face":
        with mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
            face_results = face_mesh.process(cv2_image)

            if face_results.multi_face_landmarks:
                raise ValueError("Face not recognized")

            landmarks = face_results.multi_face_landmarks[0].landmark # relative points
            h, w, _ = cv2_image.shape # Get real parametrs of image

            if category == "Earrings":              # Landmark-234 left, 454 right
                x_left = int(landmarks[234].x * w)       # Left ear
                y_left = int(landmarks[234].y * h)
                x_left_neck = int(landmarks[200].x * w)     # Left neck
                y_left_neck = int(landmarks[200].y * h)

                x_right = int(landmarks[454].x * w)       # Right ear
                y_right = int(landmarks[454].y * h)
                x_right_neck = int(landmarks[430].x * w)     # Right neck
                y_right_neck = int(landmarks[430].y * h)
                
                # coordinates
                pl1 = (x_left, y_left)                  # Left ear
                pl2 = (x_left_neck, y_left_neck)        # Left neck
                pr1 = (x_right, y_right)                # Right ear
                pr2 = (x_right_neck, y_right_neck)      # Right neck

                # angles
                angle_left = get_angle(pl1, pl2)
                angle_right = get_angle(pr1, pr2)

                # scale
                scale_left = get_scale(pl1, pl2, w)
                scale_right = get_scale(pr1, pr2, w)

                # transform
                transform_left = rotate_and_scale_image(overlay_image, angle_left, scale_left)
                transform_right = rotate_and_scale_image(overlay_image, angle_right, scale_right)

                # paste the image
                output_left = overlay_image_alpha(cv2_image, transform_left, x_left, y_left)
                output_right = overlay_image_alpha(output_left, transform_right, x_right, y_right)

                output_image = output_right

            elif category == "Necklaces": 
                x1 = int(landmarks[152].x * w)        # Landmark chin-152
                y1 = int(landmarks[152].y * h)
                x2 = int(landmarks[200].x * w)       # Landmark chin-152
                y2 = int(landmarks[200].y * h)

                # coordinates
                p1 = (x1, y1)
                p2 = (x2, y2)

                # angles
                angle = get_angle(p1, p2)

                # scale
                scale = get_scale(p1, p2, w)
                
                # transform
                transform = rotate_and_scale_image(overlay_image, angle, scale)

                # paste the image
                output = overlay_image_alpha(cv2_image, transform, x2, y2)

                output_image = output

            elif category == "Chokers":            
                x1 = int(landmarks[200].x * w)  # Left neck point
                y1 = int(landmarks[200].y * h)
                x2 = int(landmarks[430].x * w)  # Right neck point
                y2 = int(landmarks[430].y * h)
                x_center = (x1 + x2) // 2       # Center of the neck
                y_center = (y1 + y2) // 2

                # coordinates
                p1 = (x1, y1)
                p2 = (x2, y2)

                # angles
                angle = get_angle(p1, p2)
                
                # scale
                scale = get_scale(p1, p2, w)
                
                # transform
                transform = rotate_and_scale_image(overlay_image, angle, scale)

                # paste the image
                output = overlay_image_alpha(cv2_image, transform, x_center, y_center)

                output_image = output

            elif category == "Pendants":            # Landmark-152
                x1 = int(landmarks[152].x * w)
                y1 = int(landmarks[152].y * h)
                x2 = x1
                y2 = int((landmarks[152].y + 0.05) * h)  # Little lower on the neck

                # coordinates
                p1 = (x1, y1)
                p2 = (x2, y2)

                # angles
                angle = get_angle(p1, p2)
                
                # scale
                scale = get_scale(p1, p2, w)
                
                # transform
                transform = rotate_and_scale_image(overlay_image, angle, scale)

                # paste the image
                output = overlay_image_alpha(cv2_image, transform, x2, y2)

                output_image = output

            elif category == "Hair_Accessories":            # Landmark-10
                x1 = int(landmarks[10].x * w)
                y1 = int((landmarks[10].y - 0.05) * h)  # little up, in hair
                x2 = x1
                y2 = int(landmarks[152].y * h)

                # coordinates
                p1 = (x1, y1)
                p2 = (x2, y2)

                # angles
                angle = get_angle(p1, p2)
                
                # scale
                scale = get_scale(p1, p2, w)
                
                # transform
                transform = rotate_and_scale_image(overlay_image, angle, scale)

                # paste the image
                output = overlay_image_alpha(cv2_image, transform, x1, y1)
                
                output_image = output

            elif category == "Ear_Cuffs":                # Landmark-234 left, 454 right
                # Left ear
                x_left = int(landmarks[234].x * w)
                y_left = int(landmarks[234].y * h)
                x_left_up = int(landmarks[127].x * w)
                y_left_up = int(landmarks[127].y * h)
                x_left_center = (x_left + x_left_up) // 2       # Center of the ear
                y_left_center = (y_left + y_left_up) // 2

                # Right ear
                x_right = int(landmarks[454].x * w)
                y_right = int(landmarks[454].y * h)
                x_right_up = int(landmarks[356].x * w)
                y_right_up = int(landmarks[356].y * h)
                x_right_center = (x_right + x_right_up) // 2
                y_right_center = (y_right + y_right_up) // 2


                # coordinates
                p1_left = (x_left_up, y_left_up)
                p2_left = (x_left, y_left)
                p1_right = (x_right_up, y_right_up)
                p2_right = (x_right, y_right)

                # angles
                angle_left = (p1_left, p2_left)
                angle_right = (p1_right, p2_right)

                # scale
                scale_left =(p1_left, p2_left, w)
                scale_right = (p1_right, p2_right, w)

                # transform
                transform_left = rotate_and_scale_image(overlay_image, angle_left, scale_left)
                transform_right = rotate_and_scale_image(overlay_image, angle_right, scale_right)

                # paste the image
                output_left = overlay_image_alpha(cv2_image, transform_left, x_left_center, y_left_center)
                output_right = overlay_image_alpha(output_left, transform_right, x_right_center, y_right_center)

                output_image = output_right
        
        


    # hands
    # body

    output_buffer = BytesIO()
    image.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    return output_buffer.getvalue()




