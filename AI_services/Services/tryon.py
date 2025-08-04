from fastapi import APIRouter, File, UploadFile, Form
from PIL import Image
from io import BytesIO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
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

# Use an absolute path
task_path = Path(__file__).parent / "hand_landmarker.task"

print("Resolved model path:", task_path)

if not task_path.exists():
    raise FileNotFoundError(f"Model file not found at: {task_path}")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(task_path.resolve())),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    # PIL (RGB) -> NumPy array (BGR)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def pil_to_cv2_alpha(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV format, preserving alpha channel if present."""
    pil_image = pil_image.convert("RGBA")  # ensure 4 channels
    rgba = np.array(pil_image)
    # Convert RGBA (PIL) → BGRA (OpenCV)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

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
    hands_results = detector.detect(mp_image)
    results['hands'] = hands_results.hand_landmarks
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

def flip_image_horizontal(pil_image: Image.Image) -> Image.Image:
    return pil_image.transpose(Image.FLIP_LEFT_RIGHT)

def get_angle_y(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dx, dy))
    return angle

def get_angle_x(p1, p2):
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

def autocrop_image(image, border = 0):
    # Get the bounding box
    bbox = image.getbbox()

    # Crop the image to the contents of the bounding box
    image = image.crop(bbox)

    # Determine the width and height of the cropped image
    (width, height) = image.size

    # Add border
    width += border * 2
    height += border * 2
    
    # Create a new image object for the output image
    cropped_image = Image.new("RGBA", (width, height), (0,0,0,0))

    # Paste the cropped image onto the new image
    cropped_image.paste(image, (border, border))

    return cropped_image

def rotate_and_scale_image(image, angle, scale):
    # image — PIL Image with alpha channel
    w, h = image.size
    image = image.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)
    image = image.rotate(angle, expand=True, resample=Image.BICUBIC)
    
    # Cut transparent borders
    cropped = autocrop_image(image, 0)
    return cropped
    
def get_center_of_image(image):
    h, w = image.size
    hc = h // 2
    wc = w // 2
    return hc, wc

def get_x_axis_of_image (image):
    w, _ = image.size
    return w // 2

def get_y_axis_of_image (image):
    _, h = image.size
    return h // 2

def get_forearm_direction(landmarks, w, h):
    wrist = landmarks[0]
    middle_finger_base = landmarks[9]

    wx = int(wrist.x * w)
    wy = int(wrist.y * h)
    mx = int(middle_finger_base.x * w)
    my = int(middle_finger_base.y * h)

    dx = mx - wx
    dy = my - wy

    # vector inversion
    fx = -dx
    fy = -dy

    # normilize vector
    length = math.hypot(fx, fy)
    if length == 0:
        return (0, 0)
    fx /= length
    fy /= length

    return fx, fy



def overlay_image_alpha(background, overlay, x, y):
    """
    background — base image (OpenCV, BGR)
    overlay — PNG image (BGRA) with transparency
    x, y — coordinates of centre for paste - counted from landmarks
    image center will paste in x, y
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

    # working with as pil image and convert  to cv2 right before paste
    overlay_image = Image.open(product_path).convert("RGBA")    

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

            if category == "Earrings":             
                #check if ear is visible
                x_center = landmarks[1].x
                x_left_ear = landmarks[234].x
                x_right_ear = landmarks[454].x

                # distances
                dist_left = abs(x_left_ear - x_center)
                dist_right = abs(x_right_ear - x_center)

                threshold = 0.09

                # Flags of visibility
                LEFT_VISIBLE = dist_left + threshold > dist_right 
                RIGHT_VISIBLE = dist_right + threshold  > dist_left 

                # angles = 0, usual earring go down because of gravity
                angle = 0

                x_offset = int(0.015 * w)

                # left ear
                if LEFT_VISIBLE:
                    x_left = int(landmarks[162].x * w)       # Left ear up
                    y_left = int(landmarks[162].y * h)
                    x_left_down = int(landmarks[132].x * w)     # Left ear down
                    y_left_down = int(landmarks[132].y * h)
                    # coordinates for angle and scale
                    pl1 = (x_left, y_left)                  # Left ear
                    pl2 = (x_left_down, y_left_down)        # Left down
                    # scale
                    scale_left = get_scale(pl1, pl2, w)
                    # transform
                    transform_left = rotate_and_scale_image(overlay_image, angle, scale_left)
                    # check
                    print("Left transform size:", transform_left.size)
                    # coorditates to paste                
                    hl, wl = get_center_of_image(transform_left)
                    xp_left_up = int(landmarks[93].x * w)       # Up
                    yp_left_up = int(landmarks[93].y * h)
                    xp_left_down = int(landmarks[132].x * w)     # Down
                    yp_left_down = int(landmarks[132].y * h)

                    xp_left = xp_left_down - x_offset
                    yp_left = yp_left_down - abs(yp_left_up - yp_left_down) // 4 + hl
                    # convert product in cv2
                    cv2_left =  pil_to_cv2_alpha(transform_left)
                    # paste the image
                    cv2_image = overlay_image_alpha(cv2_image, cv2_left, xp_left, yp_left)

                if RIGHT_VISIBLE:
                    x_right = int(landmarks[389].x * w)       # Right ear up
                    y_right = int(landmarks[389].y * h)
                    x_right_down = int(landmarks[361].x * w)     # Right ear down 
                    y_right_down = int(landmarks[361].y * h)
                    # coordinates for angle and scale
                    pr1 = (x_right, y_right)                # Right ear
                    pr2 = (x_right_down, y_right_down)      # Right down
                    # scale
                    scale_right = get_scale(pr1, pr2, w)
                    # transform
                    transform_right = rotate_and_scale_image(overlay_image, angle, scale_right)
                    # check
                    print("Right transform size:", transform_right.size)
                    # coorditates to paste
                    hr, wr = get_center_of_image(transform_right)
                    xp_right_up = int(landmarks[323].x * w)       # Up
                    yp_right_up = int(landmarks[323].y * h)
                    xp_right_down = int(landmarks[361].x * w)     # Down
                    yp_right_down = int(landmarks[361].y * h)

                    xp_right = xp_right_down + x_offset 
                    yp_right = yp_right_down - abs(yp_right_up - yp_right_down) // 4 + hr
                    # convert product in cv2
                    cv2_right =  pil_to_cv2_alpha(transform_right)
                    # paste the image
                    cv2_image = overlay_image_alpha(cv2_image, cv2_right, xp_right, yp_right)
         
                output_image = cv2_image

            elif category == "Ear_Cuffs": 
                #check what ear is visible
                x_center = landmarks[1].x
                x_left_ear = landmarks[234].x
                x_right_ear = landmarks[454].x

                # distances
                dist_left = abs(x_left_ear - x_center)
                dist_right = abs(x_right_ear - x_center)

                threshold = 0.09

                # Flags of visibility
                visibility_flag = None
                if dist_left >= dist_right:
                    visibility_flag = "left_ear"
                else:
                    visibility_flag = "right_ear" 
                
                if visibility_flag == "left_ear":
                    x_left = int(landmarks[162].x * w)       # Left ear up
                    y_left = int(landmarks[162].y * h)
                    x_left_down = int(landmarks[132].x * w)     # Left ear down
                    y_left_down = int(landmarks[132].y * h)
                    # coordinates for angle and scale
                    pl1 = (x_left, y_left)                  # Left ear
                    pl2 = (x_left_down, y_left_down)        # Left down
                    # scale
                    scale_left = get_scale(pl1, pl2, w)
                    # angles
                    angle_left = get_angle_y(pl1, pl2)
                    # transform
                    transform_left = rotate_and_scale_image(overlay_image, angle_left, scale_left)
                    # check
                    print("Left transform size:", transform_left.size)
                    # coorditates to paste                
                    wl = get_x_axis_of_image(transform_left)
                    xp_left_center = int(x_left_down - wl)              # center
                    yp_left_center = int((y_left + y_left_down) / 2)
                    # convert product in cv2
                    cv2_left =  pil_to_cv2_alpha(transform_left)
                    # paste the image
                    cv2_image = overlay_image_alpha(cv2_image, cv2_left, xp_left_center, yp_left_center)

                    output_image = cv2_image

                elif visibility_flag == "right_ear":
                    x_right = int(landmarks[389].x * w)       # Right ear up
                    y_right = int(landmarks[389].y * h)
                    x_right_down = int(landmarks[361].x * w)     # Right ear down 
                    y_right_down = int(landmarks[361].y * h)
                    # coordinates for angle and scale
                    pr1 = (x_right, y_right)                # Right ear
                    pr2 = (x_right_down, y_right_down)      # Right down
                    # mirror lrft caff
                    mirrored_image = flip_image_horizontal(overlay_image)
                    # scale
                    scale_right = get_scale(pr1, pr2, w)
                    # angles
                    angle_right = get_angle_y(pr1, pr2)
                    # transform
                    transform_right = rotate_and_scale_image(mirrored_image, angle_right, scale_right)
                    # check
                    print("Right transform size:", transform_right.size)
                    # coorditates to paste                
                    wl = get_x_axis_of_image(transform_right)
                    xp_right_center = int(x_right_down + wl)              # center
                    yp_right_center = int((y_right + y_right_down) / 2)
                    # convert product in cv2
                    cv2_right =  pil_to_cv2_alpha(transform_right)
                    # paste the image
                    cv2_image = overlay_image_alpha(cv2_image, cv2_right, xp_right_center, yp_right_center)

                    output_image = cv2_image

                else:
                    output_image = cv2_image 

            elif category in [ "Necklaces", "Chokers", "Pendants"]: 
                # angles = 0, usual necklace go down because of gravity
                angle = 0

                # y offset
                if category == "Necklaces":
                    y_offset = int(0.1 * w)
                else:
                    y_offset = int(0.1 * w)

                # x offset
                x_nose_bridge = int(landmarks[1].x * w)
                x_chin = int(landmarks[152].x * w)
                head_tilt = x_nose_bridge - x_chin

                # neck left
                x1 = int(landmarks[58].x * w)        
                y1 = int(landmarks[58].y * h)
                # neck right
                x2 = int(landmarks[288].x * w)       
                y2 = int(landmarks[288].y * h)
                # coordinates
                p1 = (x1, y1)
                p2 = (x2, y2)
                # scale
                scale = get_scale(p1, p2, w) * 1.1 # necklace should be a little wider than neck
                # transform
                transform = rotate_and_scale_image(overlay_image, angle, scale)
                # coorditates to paste    
                hc = get_y_axis_of_image(transform)       
                y_chin = int(landmarks[152].y * h) 
                xp_center = (x1 + x2) // 2 - int(head_tilt * 0.9)
                yp_center = abs((y1 + y2) // 2) + hc + y_offset
                if y_chin + hc  > yp_center:
                    yp_center = y_chin + hc 
          
                # convert product in cv2
                cv2_transform =  pil_to_cv2_alpha(transform)
                # paste the image
                output = overlay_image_alpha(cv2_image, cv2_transform, xp_center, yp_center)

                output_image = output

            elif category == "Hair_Accessories":            
                x1 = int(landmarks[10].x * w)
                y1 = int(landmarks[10].y * h)  # little up, in hair
                x2 = int(landmarks[164].x * w)
                y2 = int(landmarks[164].y * h)
                # coordinates
                p1 = (x1, y1)
                p2 = (x2, y2)
                # angles
                angle = get_angle_y(p1, p2)                
                # scale
                scale = get_scale(p1, p2, w)                
                # transform
                transform = rotate_and_scale_image(overlay_image, angle, scale)
                # coorditates to paste
                xp = x1
                yp = int((landmarks[10].y - 0.05) * h)
                # convert product in cv2
                cv2_transform =  pil_to_cv2_alpha(transform)
                # paste the image
                output = overlay_image_alpha(cv2_image, cv2_transform, xp, yp)
                
                output_image = output

            
        
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
            
            landmarks = best_hand.landmark # relative points
            h, w, _ = cv2_image.shape # Get real parametrs of image

            if category == "Rings":              
                x1 = int(landmarks[11].x * w)        # Landmark middle finger-pip-10
                y1 = int(landmarks[11].y * h)
                x2 = int(landmarks[9].x * w)       # Landmark middle finger-mcp-9
                y2 = int(landmarks[9].y * h)                
                # coordinates
                p1 = (x1, y1)
                p2 = (x2, y2)
                # angles
                angle = get_angle_y(p1, p2)
                # scale
                scale = get_scale(p1, p2, w)                
                # transform
                transform = rotate_and_scale_image(overlay_image, angle, scale)
                # convert product in cv2
                cv2_transform =  pil_to_cv2_alpha(transform)
                # coorditates to paste
                x3 = int(landmarks[10].x * w)        # Landmark middle finger-pip-10
                y3 = int(landmarks[10].y * h)
                x_center = (x3 + x2) // 2       # Center 
                y_center = (y3 + y2) // 2
                # paste the image
                output = overlay_image_alpha(cv2_image, cv2_transform, x_center, y_center)

                output_image = output

            
            # To Do
            elif category == "Bracelets": 
                x1 = int(landmarks[9].x * w)       # Landmark middle finger-mcp-9
                y1 = int(landmarks[9].y * h)
                x2 = int(landmarks[0].x * w)       # Landmark wrist-0
                y2 = int(landmarks[0].y * h)
                x3 = int(landmarks[5].x * w)       # Landmark middle finger-mcp-9
                y3 = int(landmarks[5].y * h)
                x4 = int(landmarks[17].x * w)       # Landmark middle finger-mcp-9
                y4 = int(landmarks[17].y * h)

                # coordinates
                p1 = (x1, y1)
                p2 = (x2, y2)
                p3 = (x3, y3)
                p4 = (x4, y4)

                # angles
                angle = get_angle_y(p1, p2)

                # scale
                scale = get_scale(p3, p4, w)

                # vector
                fx, fy = get_forearm_direction(landmarks, w, h)
                
                # transform
                transform = rotate_and_scale_image(overlay_image, angle, scale)
                # convert product in cv2
                cv2_transform =  pil_to_cv2_alpha(transform)
                # coorditates to paste
                
                # Get point for paste
                x_shift = int(x2 + fx * 30)
                y_shift = int(y2 + fy * 30)

                # paste the image
                output = overlay_image_alpha(cv2_image, cv2_transform, x_shift, y_shift)

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
        angle = get_angle_y(p1, p2)

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




 