from fastapi import APIRouter, File, UploadFile, Form
from PIL import Image
from io import BytesIO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision  
import cv2
import numpy as np
from PIL import Image
import math
from pathlib import Path
from AI_services.Helpers.CategoryTryOnDict import CategoryTryOnDict


mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

# new model for hands
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# path for task 
task_path = Path(__file__).resolve().parents[2] / "ai_models" / "hand_landmarker.task"

print("Resolved model path:", task_path)

if not task_path.exists():
    raise FileNotFoundError(f"Model file not found at: {task_path}")

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2)

detector = vision.HandLandmarker.create_from_options(options)


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
        pose_landmarks = results.get('pose')
        first_pose = pose_landmarks if pose_landmarks else None  # pose всегда один результат
        print("Pose landmarks:", first_pose)

    # Hands
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    with detector.detect(mp_image) as hands:
        hands_results = hands.process(cv2_image)
        results['hands'] = hands_results.multi_hand_landmarks
        hand_landmarks = results.get('hands')
        first_hand = hand_landmarks[0] if hand_landmarks and len(hand_landmarks) > 0 else None
        print("First hand landmarks:", first_hand)

    # Face mesh
    with mp_face.FaceMesh(static_image_mode=True) as face:
        face_results = face.process(cv2_image)
        results['face'] = face_results.multi_face_landmarks
        face_landmarks = results.get('face')
        first_face = face_landmarks[0] if face_landmarks and len(face_landmarks) > 0 else None
        print("First face landmarks:", first_face)

    return results

def is_part_present(results, category_group):
    # body
    if category_group == 'body' :
        return results['pose'] is not None
    # hands
    if category_group == 'hands':
        return results['hands'] is not None
    # face
    if category_group == 'face':
        return results['face'] is not None

def resize_image(image, target_width=640):
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    new_dim = (target_width, int(h * scale))
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)


def get_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def shift_along_angle(x, y, angle_deg, distance):
    angle_rad = math.radians(angle_deg)
    dx = int(math.cos(angle_rad) * distance)
    dy = int(math.sin(angle_rad) * distance)
    return x + dx, y + dy

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

def get_hand_score(landmarks, image_shape):
    h, w, _ = image_shape
    # coordinates from landmarks
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    # How much landmarks are on the picture
    visible_points = sum(0 <= lm.x <= 1 and 0 <= lm.y <= 1 for lm in landmarks)

    # How much area covered with a hand
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    area = (max_x - min_x) * (max_y - min_y)

    # get the score
    score = visible_points + area * 100

    return score

# To Do sketch
async def process_image(file: UploadFile, category: str, id: int):

    # Get the image from client
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to read image: {str(e)}")

    cv2_image = pil_to_cv2(image)
    if cv2_image is None:
        raise ValueError("Failed to convert image to OpenCV format")

    # scaling 
    cv2_image = resize_image(cv2_image, target_width=640)

    # Get the inamge of product by id
    project_root = Path(__file__).resolve().parents[3]  # go to the root reposetory
    product_path = project_root / "JFjewelery" / "Media" / "images" / "products" / str(id) / "tryon.png"

    if not product_path.exists():
        raise FileNotFoundError(f"Overlay image not found at: {product_path}")

    overlay_image = cv2.imread(str(product_path), cv2.IMREAD_UNCHANGED)
    if overlay_image is None:
        raise ValueError(f"Failed to load image at path: {product_path}")

    print("Resolved product path:", product_path)

    # Declare the output
    output_image = None

    # Get the category
    selected_group = find_group_by_category(category)

    # General pose analysis
    results = analyze_image_parts(cv2_image)

    if not is_part_present(results, selected_group):
        raise ValueError(f"Photo not suitable for a category {category}, body part not found")

    # face
    if selected_group == "face":
        with mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
            face_results = face_mesh.process(cv2_image)

            if not face_results.multi_face_landmarks:
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
    if selected_group == "hands":
        with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
            hand_results = hands.process(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

        if not hand_results.multi_hand_landmarks:
            raise ValueError("Hand not recognized")
            
        best_hand = None
        best_score = -1

        for hand_landmarks in hand_results.multi_hand_landmarks:
            score = get_hand_score(hand_landmarks.landmark, cv2_image.shape)
            if score > best_score:
                best_score = score
                best_hand = hand_landmarks
            
            landmarks_ = best_hand.landmark # relative points
            h, w, _ = cv2_image.shape # Get real parametrs of image

            if category == "Rings":              
                x1 = int(landmarks[10].x * w)        # Landmark middle finger-pip-10
                y1 = int(landmarks[10].y * h)
                x2 = int(landmarks[9].x * w)       # Landmark middle finger-mcp-9
                y2 = int(landmarks[9].y * h)
                x_center = (x1 + x2) // 2       # Center 
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

            
            # To Do
            elif category == "Bracelets": 
                x1 = int(landmarks[9].x * w)       # Landmark middle finger-mcp-9
                y2 = int(landmarks[9].y * h)
                x2 = int(landmarks[0].x * w)       # Landmark wrist-0
                y2 = int(landmarks[0].y * h)

                # coordinates
                p1 = (x1, y1)
                p2 = (x2, y2)

                # angles
                angle = get_angle(p1, p2)

                # scale
                scale = get_scale(p1, p2, w)
                
                # transform
                transform = rotate_and_scale_image(overlay_image, angle, scale)
                
                # Get point for paste
                x_shift, y_shift = shift_along_angle(x1, y1, angle, 30)

                # paste the image
                output = overlay_image_alpha(cv2_image, transform, x_shift, y_shift)

                output_image = output

    # body
    if selected_group == "body":
        with mp_pose.Pose(static_image_mode=True) as pose:
            pose_results = pose.process(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

        if not pose_results.pose_landmarks:
            raise ValueError("Body not recognized")

        landmarks = pose_results.pose_landmarks.landmark # relative points
        h, w, _ = cv2_image.shape # Get real parametrs of image

        # Common calculations for pose
        x_left = int(landmarks[12].x * w)       # Left shoulder
        y_left = int(landmarks[12].y * h)
        x_left_pelvis = int(landmarks[24].x * w)     # Left pelvis
        y_left_pelvis  = int(landmarks[24].y * h)

        x_right = int(landmarks[11].x * w)      # Right shoulder
        y_right = int(landmarks[11].y * h)
        x_right_pelvis  = int(landmarks[23].x * w)     # Right pelvis
        y_right_pelvis  = int(landmarks[23].y * h)

        # Axis shoulder
        x_axis_shoulder = int((x_left + x_right) / 2)
        y_axis_shoulder = int((y_left + y_right) / 2)

        # Axis pelvis
        x_axis_pelvis = int((x_left_pelvis + x_right_pelvis) / 2)
        y_axis_pelvis = int((y_left_pelvis + y_right_pelvis) / 2)

        # coordinates
        p1 = (x_axis_shoulder, y_axis_shoulder)
        p2 = (x_axis_pelvis, y_axis_pelvis)

        # angles
        angle = get_angle(p1, p2)

        # scale
        scale = get_scale(p1, p2, w)

        # transform
        transform = rotate_and_scale_image(overlay_image, angle, scale)
                
 

        # To Do
        if category in ["Brooches", "Pins"]:              
            # 3/4 of the chest
            x_chest = int(x_right - (x_right - x_left) * 0.25)
            y_chest = int(y_right + (y_right - y_right_pelvis)* 0.4)

            # paste the image
            output = overlay_image_alpha(cv2_image, transform, x_chest, y_chest)

            output_image = output

            
        # To Do
        elif category == "Chains": 
            # paste points
            x_base = int((landmarks[11].x + landmarks[12].x) / 2 * w)
            y_base = int((landmarks[11].y + landmarks[12].y) / 2 * h)
            y_top = y_base - int(0.05 * h)     

            # paste the image
            output = overlay_image_alpha(cv2_image, transform, x_base, y_top)

            output_image = output  

    # BGR to RBG
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Outpun cv => PIL
    pil_image = Image.fromarray(output_image_rgb)

    # save in bytes
    output_buffer = BytesIO()
    pil_image.save(output_buffer, format="PNG")
    # set on the begining of buffer
    output_buffer.seek(0)

    # get bytes
    image_bytes = output_buffer.getvalue()

    return image_bytes




 