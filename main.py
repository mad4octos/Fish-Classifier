import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO
import os
from PIL import Image
import time
from torchvision import transforms
from classifier import BinaryClassificationCNN  # Your classifier module
from underwater_enhancement import LACE, LACC


# Load models
sam2_checkpoint = "/projects/maha7624/software/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
yolo_model = YOLO('/scratch/alpine/maha7624/classifier/deliver/yolov8s.pt')

# Load and setup classifier
class_model = BinaryClassificationCNN()
class_model.load_state_dict(torch.load(
    "/scratch/alpine/maha7624/classifier/deliver/model_weights.pth"))
class_model.eval()

# Preprocessing function for the classifier
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Initialize inference state
video_dir = "/scratch/alpine/maha7624/classifier/deliver/sub"
frame_names = sorted(
    [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
    key=lambda p: int(os.path.splitext(p)[0])
)


# Tracking variables
tracking_initialized = False
paused = False
tracked_objects = {}
object_states = {}
next_obj_id = 1

# Display settings
DISPLAY_WIDTH = 800
scale_factor = None


def get_box_from_mask(mask):
    """Calculate bounding box from mask"""
    if mask is None or not mask.any():
        return None

    y_indices, x_indices = np.nonzero(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    x1, x2 = np.min(x_indices), np.max(x_indices)
    y1, y2 = np.min(y_indices), np.max(y_indices)
    return np.array([x1, y1, x2, y2])


def resize_frame(frame):
    """Resize frame while maintaining aspect ratio"""
    global scale_factor
    height, width = frame.shape[:2]
    scale_factor = DISPLAY_WIDTH / width
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(frame, (new_width, new_height)), scale_factor


def classify_detection(frame, box):
    """Classify a detection using the binary classifier"""
    x1, y1, x2, y2 = map(int, box)
    cropped = frame[y1:y2, x1:x2]

    if cropped.size == 0:
        return False

    try:
        input_tensor = preprocess(cropped).unsqueeze(0)
        with torch.no_grad():
            class_output = class_model(input_tensor)
            prob = torch.sigmoid(class_output).item()
            return prob >= 0.6  # Return True if it's the target class
    except Exception as e:
        print(f"Classification failed: {e}")
        return False

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def merge_new_detections(new_detections):
    global next_obj_id

    for new_box, new_cls, new_conf in new_detections:
        # Check if this new detection overlaps with an existing object
        is_new_object = True
        for obj_id, obj_info in tracked_objects.items():
            iou = calculate_iou(new_box, obj_info['box'])
            if iou > 0.3:  # Overlapping with an existing tracked object
                is_new_object = False
                break

        if is_new_object:
            # Add the new object
            object_states[next_obj_id] = predictor.init_state(video_path=video_dir)
            tracked_objects[next_obj_id] = {
                'box': new_box,
                'class': new_cls,
                'confidence': new_conf,
                'color': np.random.randint(0, 255, size=3).tolist()
            }
            print(f"New object detected: ID {next_obj_id}")
            next_obj_id += 1

def initialize_tracking(frame):
    """Initialize tracking using YOLOv8 detections and classification"""
    global tracking_initialized, next_obj_id, tracked_objects, object_states

    # Enhance the frame
    beta = 1.5
    frame = frame / 255.0
    frame, _ = LACC(frame)
    frame = LACE(frame * 255.0, beta=beta)
    frame = frame.astype(np.uint8)

    # Run YOLOv8 detection
    results = yolo_model(frame)

    # Clear existing tracking data
    tracked_objects.clear()
    object_states.clear()

    # Process each detection
    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result
        if conf > 0.5:  # Confidence threshold
            # Convert to integer coordinates
            box = np.array([int(x1), int(y1), int(x2), int(y2)])

            # Only process if classification passes
            if classify_detection(frame, box):
                # Initialize SAM state for this object
                object_states[next_obj_id] = predictor.init_state(video_path=video_dir)

                # Create new tracking entry
                tracked_objects[next_obj_id] = {
                    'box': box,
                    'class': int(cls),
                    'confidence': float(conf),
                    'color': np.random.randint(0, 255, size=3).tolist()
                }
                next_obj_id += 1

    if tracked_objects:
        tracking_initialized = True
        print(f"Initialized tracking for {len(tracked_objects)} objects")


def process_frame(frame_idx):
    global tracking_initialized, tracked_objects, object_states, next_obj_id

    frame_path = os.path.join(video_dir, frame_names[frame_idx])
    frame = np.array(Image.open(frame_path))
    original_frame = frame.copy()

    try:
        # If not initialized, start tracking with initial YOLO detections
        if not tracking_initialized:
            initialize_tracking(frame)

        # Always check for new detections in each frame
        results = yolo_model(frame)

        new_detections = []

        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result
            if conf > 0.5:  # Confidence threshold
                new_box = np.array([int(x1), int(y1), int(x2), int(y2)])
                if classify_detection(frame, new_box):  # Ensure it's a valid target
                    new_detections.append((new_box, int(cls), float(conf)))

        # Merge new detections with tracked objects
        merge_new_detections(new_detections)

        # Process each tracked object
        objects_to_remove = []
        vis_frame = frame.copy()

        for obj_id, obj_info in tracked_objects.items():
            try:
                # Get object's SAM state
                current_state = object_states[obj_id]

                # Get segmentation prediction
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=current_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=obj_info['box']
                )

                if out_mask_logits is not None:
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                    if len(mask.shape) == 3 and mask.shape[0] == 1:
                        mask = np.squeeze(mask, axis=0)

                    # Update tracking box
                    new_box = get_box_from_mask(mask)
                    if new_box is not None:
                        if classify_detection(frame, new_box):
                            obj_info['box'] = new_box
                        else:
                            objects_to_remove.append(obj_id)
                            continue
                    else:
                        objects_to_remove.append(obj_id)
                        continue

                    # Visualize
                    color = obj_info['color']
                    overlay = np.zeros_like(frame)
                    overlay[mask > 0] = color
                    vis_frame = cv2.addWeighted(vis_frame, 1.0, overlay, 0.3, 0)

                    # Draw bounding box
                    x1, y1, x2, y2 = obj_info['box'].astype(int)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID: {obj_id} (Class: {obj_info['class']})"
                    cv2.putText(vis_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            except Exception as e:
                print(f"Error processing object {obj_id}: {str(e)}")
                objects_to_remove.append(obj_id)

        # Remove lost objects
        for obj_id in objects_to_remove:
            tracked_objects.pop(obj_id, None)
            object_states.pop(obj_id, None)

        frame = vis_frame

        # Add status text
        status = f"Tracking {len(tracked_objects)} objects" if tracking_initialized else "Initializing tracking"
        if paused:
            status += " (Paused)"
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{len(frame_names)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if tracking_initialized else (0, 0, 255), 2)

        # Resize frame for display
        resized_frame, _ = resize_frame(frame)
        return resized_frame

    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        resized_frame, _ = resize_frame(original_frame)
        return resized_frame


# Setup window
#cv2.namedWindow("SAM Multi-Object Tracking")

# Main loop
frame_idx = 0
frame_delay = 1 / 30  # 30 FPS
total_frames = len(frame_names)

print("\nControls:")
print("Space - Pause/Resume")
print("R - Reinitialize tracking")
print("Q - Quit")

while frame_idx < total_frames:
    current_time = time.time()

    if not paused:
        frame = process_frame(frame_idx)

        #if frame is not None:
            # Convert to BGR for display
           # frame_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #cv2.imshow("SAM Multi-Object Tracking", frame_display)

        # Advance frame
        frame_idx += 1

    # Handle keyboard input
    #key = cv2.waitKey(1) & 0xFF
    #if key == ord('q'):  # Quit
    #    break
    #elif key == ord('r'):  # Reset tracking
    #    tracking_initialized = False
    #    next_obj_id = 1
    #elif key == ord(' '):  # Space to pause/resume
    #    paused = not paused

    # Control frame rate
    elapsed = time.time() - current_time
    if elapsed < frame_delay:
        time.sleep(frame_delay - elapsed)

print("\nVideo processing completed!")
#cv2.destroyAllWindows()