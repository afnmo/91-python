import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from tracker import*
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Load YOLO model
model = YOLO('yolov8n.pt')


# Video capture from device for stream1
stream1 = cv2.VideoCapture(r"C:\flutterapps\vidoe1.mp4")

# Video capture from device for stream2
stream2 = cv2.VideoCapture(r"C:\flutterapps\video5.mp4")

# Define entering and exiting areas
entering_area = [(120, 260), (313, 258), (690, 360), (500, 430)]
exiting_area = [(343, 176), (620, 140), (830, 240), (640, 370)]

# Mouse callback function
def RGB1(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print("Stream 1:", colorsBGR)

def RGB2(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print("Stream 2:", colorsBGR)

# Create windows for both streams
cv2.namedWindow('Stream 1')
cv2.namedWindow('Stream 2')

# Set mouse callback function for each window
cv2.setMouseCallback('Stream 1', RGB1)
cv2.setMouseCallback('Stream 2', RGB2)



# Initialize variables and objects for tracking and counting
tracker1 = Tracker()
tracker2 = Tracker()
entering_cars = {}
entering_car_counter = []
exiting_cars = {}
exiting_car_counter = []
current = 0
# database connection 
cred = credentials.Certificate(r"C:\flutterapps\app-be149-firebase-adminsdk-q5m3j-38fa27ada9.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
stationId = '8W5t2M4sCP5tTww53Xlp'
# Main loop for processing frames from both streams
while True:
    # Read frames from both streams
    ret1, frame1 = stream1.read()
    ret2, frame2 = stream2.read()

    # Check if frames were read successfully
    if not ret1 or not ret2:
        break

    # Resize frames if necessary
    frame1 = cv2.resize(frame1, (1020, 500))
    frame2 = cv2.resize(frame2, (1020, 500))




    # Detect cars and update tracker for stream1
    detections1 = model(frame1)[0]
    if detections1:
        detections1_ = []
        for detection1 in detections1.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection1
            if int(class_id) in [2, 3, 5, 7]:  # Filtering vehicles
                detections1_.append([x1, y1, x2, y2, score])
        # print(detections1_)
        tracked_cars1 = tracker1.update(detections1_)
    else:
        tracked_cars1 = []

    # Detect cars and update tracker for stream2
    detections2 = model(frame2)[0]
    if detections2:
        detections2_ = []
        for detection2 in detections2.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection2
            if int(class_id) in [2, 3, 5, 7]:  # Filtering vehicles
                detections2_.append([x1, y1, x2, y2, score])

        tracked_cars2 = tracker2.update(detections2_)
    else:
        tracked_cars2 = []

    # Process entering and exiting cars for both streams
    for stream_index, (tracked_cars, area, car_counter, cars_dict) in enumerate([(tracked_cars1, entering_area, entering_car_counter, entering_cars), 
                                                                                (tracked_cars2, exiting_area, exiting_car_counter, exiting_cars)], start=1):
        for bbox in tracked_cars:
            x1, y1, x2, y2, car_id = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Check if the car is within the entering or exiting area
            if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                cars_dict[car_id] = (cx, cy)
                if car_id not in car_counter:
                    car_counter.append(car_id)

    current = len(entering_car_counter) - len(exiting_car_counter)
    doc_ref = db.collection('Station').document(stationId)
    doc_ref.update({'current': current})
    # Display entering and exiting car counts on frames
    cvzone.putTextRect(frame1, f'Entering: {len(entering_car_counter)}', (50, 60), 1, 1)
    cvzone.putTextRect(frame2, f'Exiting: {len(exiting_car_counter)}', (846, 59), 1, 1)
    
    # Draw entering area on stream1
    cv2.polylines(frame1, [np.array(entering_area, np.int32)], True, (0, 255, 0), 2)

    # Draw exiting area on stream2
    cv2.polylines(frame2, [np.array(exiting_area, np.int32)], True, (0, 0, 255), 2)
    # Display frames
    cv2.imshow("Stream 1", frame1)
    cv2.imshow("Stream 2", frame2)

    # Check for exit condition
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video streams and close windows
stream1.release()
stream2.release()
cv2.destroyAllWindows()
