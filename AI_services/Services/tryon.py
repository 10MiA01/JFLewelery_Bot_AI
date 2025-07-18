from fastapi import APIRouter, File, UploadFile, Form
from PIL import Image
from io import BytesIO
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
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



# To Do sketch
async def process_image(file: UploadFile, category: str, id: int):

    # Get the image
    image = Image.open(file.file).convert("RGB")
    cv2_image = pil_to_cv2(image)

    # Get the category
    selected_group = find_group_by_category(category)

    # General pose analysis
    results = analyze_image_parts(cv2_image)

    if not is_part_present(results, selected_group):
        raise ValueError(f"Photo nit suitable for a category {category}, ????? ???? ?? ???????")

    # face
    if selected_group == "face":
        with mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
            face_results = face_mesh.process(cv2_image)

            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark # relative points
                h, w, _ = cv2_image.shape # Get real parametrs of image

            if category == "Earrings":              # Landmark-234 left, 454 right
                x_left = int(landmarks[234].x * w)       # Left ear
                y_left = int(landmarks[234].y * h)
                x_right = int(landmarks[454].x * w)       # Right ear
                y_right = int(landmarks[454].y * h)

                # paste the image

            elif category == "Necklaces":           # Landmark-152
                x = int(landmarks[152].x * w)
                y = int(landmarks[152].y * h)

                # image.paste(necklace_img, (x, y), necklace_img)

            elif category == "Chokers":             # Landmark-17
                x = int((landmarks[17].x + landmarks[152].x) / 2 * w)
                y = int((landmarks[17].y + landmarks[152].y) / 2 * h)

                # image.paste(choker_img, (x, y), choker_img)


            elif category == "Pendants":            # Landmark-152
                x = int(landmarks[152].x * w)
                y = int((landmarks[152].y + 0.05) * h)  # Little lower on the neck

                # image.paste(pendant_img, (x, y), pendant_img)


            elif category == "Hair_Accessories":            # Landmark-10
                x = int(landmarks[10].x * w)
                y = int((landmarks[10].y - 0.05) * h)  # little up, in hair

                # image.paste(hair_img, (x, y), hair_img)

            elif category == "Ear_Cuffs":                # Landmark-234 left, 454 right
                x_left = int(landmarks[234].x * w)       # Left ear
                y_left = int(landmarks[234].y * h)
                x_right = int(landmarks[454].x * w)       # Right ear
                y_right = int(landmarks[454].y * h)

                # paste the image


        #analyze face 
        


    # hands
    # body

    output_buffer = BytesIO()
    image.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    return output_buffer.getvalue()




