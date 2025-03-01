from ultralytics import YOLO
import cv2
import numpy as np
from sort import *
from util import get_car, read_license_plate, write_csv

# Initialize the results dictionary
results = {}

# Initialize the tracker
mot_tracker = Sort()

# Load YOLO models
coco_model = YOLO('../Yolo_Weights/yolov8n.pt')
license_plate_detector = YOLO('./models/best.pt')

# Load the video
cap = cv2.VideoCapture('../Videos/anpr_6.mp4')
vehicles = [2, 3, 5, 7]

# Read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break
    
    # Initialize the frame in the dictionary
    results[frame_nmr] = {}
    
    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if class_id in vehicles:
            detections_.append([x1, y1, x2, y2, score])
    
    # Check if detections exist before updating the tracker
    if len(detections_) > 0:
        track_ids = mot_tracker.update(np.asarray(detections_))
    else:
        track_ids = np.empty((0, 5))  # Empty array to avoid error
    
    # Detect license plates
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Assign a license plate to a car
        if track_ids.shape[0] == 0:
            continue  # Avoid error if no vehicle is tracked
        
        car_data = get_car(license_plate, track_ids)
        if car_data is None:
            continue  # Avoid error if no car is found
        xcar1, ycar1, xcar2, ycar2, car_id = car_data
        
        # Initialize the key car_id if it does not exist
        if car_id not in results[frame_nmr]:
            results[frame_nmr][car_id] = {}
        
        # Crop the license plate
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        
        # Check if the crop is valid
        if license_plate_crop.size == 0:
            continue
        
        # Process license plate
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

        # Read license plate number
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
        
        if license_plate_text is not None:
            results[frame_nmr][car_id] = {
                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                'license_plate': {
                    'bbox': [x1, y1, x2, y2],
                    'text': license_plate_text,
                    'bbox_score': score,
                    'text_score': license_plate_text_score
                }
            }

# Write the results to a CSV file
write_csv(results, './result_1.csv')
