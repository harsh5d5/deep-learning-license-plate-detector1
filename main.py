import os

from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# Determine the absolute directory of the currently running script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# global results and persistence
results = {}
persistent_info = {} # stores the latest detection for each car_id

mot_tracker = Sort()

# Load Models
coco_model = YOLO(os.path.join(SCRIPT_DIR, 'yolov8n.pt'))
license_plate_detector = YOLO(os.path.join(SCRIPT_DIR, 'models', 'license_plate_detector.pt'))

# load video using absolute path
video_path = os.path.abspath(os.path.join(SCRIPT_DIR, 'number_detection', '2103099-uhd_3840_2160_30fps.mp4'))
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # --- PERFORMANCE OPTIMIZATION: Process at 1080p instead of 4K ---
        # 4K is too heavy for real-time CPU processing. Downscaling once.
        target_h, target_w = 1080, 1920
        frame = cv2.resize(frame, (target_w, target_h))
        
        if frame_nmr % 10 == 0:
            print(f"Processing frame {frame_nmr}...")
        results[frame_nmr] = {}
        
        # detect vehicles
        detections = coco_model(frame, verbose=False)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame, verbose=False)[0]
        for license_plate in license_plates.boxes.data.tolist():
            lp_x1, lp_y1, lp_x2, lp_y2, lp_score, lp_class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Calculate the current relative offset regardless of OCR
                car_w, car_h = xcar2 - xcar1, ycar2 - ycar1
                new_rel_offset = [(lp_x1 - xcar1) / car_w, (lp_y1 - ycar1) / car_h, 
                                 (lp_x2 - xcar1) / car_w, (lp_y2 - ycar1) / car_h]

                # Initialize or update the persistent info
                if car_id not in persistent_info:
                    persistent_info[car_id] = {
                        'text': 'None',
                        'score': 0,
                        'crop': None,
                        'rel_offset': new_rel_offset
                    }
                else:
                    # SMOOTHING: Blend the new offset with the old one (0.2 alpha) to prevent "jumping"
                    alpha = 0.2
                    old_off = persistent_info[car_id]['rel_offset']
                    persistent_info[car_id]['rel_offset'] = [
                        old_off[i] * (1 - alpha) + new_rel_offset[i] * alpha for i in range(4)
                    ]

                # Skip OCR if high-confidence result exists
                if persistent_info[car_id]['score'] > 0.9:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {'bbox': [lp_x1, lp_y1, lp_x2, lp_y2],
                                          'text': persistent_info[car_id]['text'],
                                          'bbox_score': lp_score,
                                          'text_score': persistent_info[car_id]['score']}
                    }
                    continue

                # crop license plate
                license_plate_crop = frame[int(lp_y1):int(lp_y2), int(lp_x1): int(lp_x2), :]

                # process license plate (Lightweight)
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # read license plate number (Heavyweight - CPU OCR)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [lp_x1, lp_y1, lp_x2, lp_y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': lp_score,
                                                                    'text_score': license_plate_text_score}}
                    
                    # --- PRECISION UPDATE: Only update text/crop if score is better ---
                    if license_plate_text_score > persistent_info[car_id]['score']:
                        persistent_info[car_id].update({
                            'text': license_plate_text,
                            'score': license_plate_text_score,
                            'crop': license_plate_crop.copy()
                        })

        # --- ADVANCED VISUALIZATION WITH PERSISTENCE ---
        # Only draw for vehicles that have a detected license plate and are not exiting
        for track in track_ids:
            xcar1, ycar1, xcar2, ycar2, car_id = track
            
            # --- EXIT ZONE: Remove annotations when car reaches bottom of frame ---
            if ycar2 > (target_h - 10):
                if car_id in persistent_info:
                    del persistent_info[car_id]
                continue

            # 1. Only show information if we have a detected plate for this car
            if car_id in persistent_info:
                # Draw car "corner" border (Green)
                draw_border(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 5, 
                            line_length_x=40, line_length_y=40)
                
                info = persistent_info[car_id]
                lp_text = info['text']
                lp_crop = info['crop']
                rel_offset = info['rel_offset']

                # Calculate current plate bbox based on car's current position (Relative Locking)
                cur_car_w, cur_car_h = xcar2 - xcar1, ycar2 - ycar1
                lp_x1 = xcar1 + (rel_offset[0] * cur_car_w)
                lp_y1 = ycar1 + (rel_offset[1] * cur_car_h)
                lp_x2 = xcar1 + (rel_offset[2] * cur_car_w)
                lp_y2 = ycar1 + (rel_offset[3] * cur_car_h)
                
                # Draw plate bounding box (Now locked to the car)
                cv2.rectangle(frame, (int(lp_x1), int(lp_y1)), (int(lp_x2), int(lp_y2)), (0, 0, 255), 2)

                # 2. Add White Info Box (Crop + Text) above car
                try:
                    # Prepare plate crop for display
                    display_crop = cv2.resize(lp_crop, (int((lp_x2 - lp_x1) * 100 / (lp_y2 - lp_y1)), 100))
                    H, W, _ = display_crop.shape
                    
                    # Background area (White)
                    bx1, by1 = int((xcar2 + xcar1 - W) / 2), int(ycar1 - H - 80)
                    bx2, by2 = int((xcar2 + xcar1 + W) / 2), int(ycar1 - 10)
                    
                    # Ensure within frame bounds
                    if by1 > 0 and bx1 > 0 and bx2 < frame.shape[1]:
                        # Draw White box for text background
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2 - H - 5), (255, 255, 255), -1)
                        
                        # Put Cropped Plate Image
                        frame[by2 - H:by2, bx1:bx2, :] = display_crop
                        
                        # Draw Plate Number Text
                        (text_w, text_h), _ = cv2.getTextSize(lp_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                        cv2.putText(frame, lp_text, 
                                    (int((bx1 + bx2 - text_w) / 2), int(by1 + (by2 - by1 - H) / 2 + text_h / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                except Exception as e:
                    pass

        # Show the frame - Already at 1080p, so minimal scaling overhead
        cv2.imshow('ANPR Live Detection', cv2.resize(frame, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close window
cap.release()
cv2.destroyAllWindows()

# write results to local directory
output_csv = os.path.join(SCRIPT_DIR, 'test.csv')
write_csv(results, output_csv)