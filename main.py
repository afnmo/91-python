import cv2
from sort.sort import *
from ultralytics import YOLO
import csv 
import numpy as np
from database import CarSearch

# Initialize SORT tracker
mot_tracker = Sort()

# Load YOLO models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('python/locate_plate.pt')
read_plate = YOLO('python/extract_plate.pt')

area = [(300, 1040), (1175, 870), (1902, 994), (1590, 1575)]

        # for detection in detections.boxes.data.tolist():
        #     x1, y1, x2, y2, score, class_id = detection
        #     if int(class_id) in [2, 3, 5, 7]:  # Filtering vehicles
        #         detections_.append([x1, y1, x2, y2, score])

# Mapping numbers to substitute characters
number_to_substitute = {
    10: 'A',
    11: 'B',
    12: 'D',
    13: 'E',
    14: 'G',
    15: 'H',
    16: 'J',
    17: 'K',
    18: 'L',
    19: 'N',
    20: 'R',
    21: 'S',
    22: 'T',
    23: 'U',
    24: 'V',
    25: 'X',
    26: 'Z'
}

# Load video
cap = cv2.VideoCapture('python/video9.mov')

def RGB1(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('Stream 1')

cv2.setMouseCallback('Stream 1', RGB1)
# Container to store the predicted characters and their confidence scores for each license plate
plate_characters = {}

# Read frames
frame_nmr = -1
ret = True
while ret:
    if frame_nmr > 30:
        break
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret or frame.size == 0:
        print("Error: Empty frame or invalid dimensions")
        break
    if ret:
        print(frame.size)
        # Detect vehicles
        cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in [2, 3, 5, 7]:  # Filtering vehicles
                # Check if the center of the detected vehicle is within area1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                    detections_.append([x1, y1, x2, y2, score])
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        if detections_:
            track_ids = mot_tracker.update(np.asarray(detections_))
        

        for detection in track_ids:
            x1, y1, x2, y2, track_id = detection  # Extract coordinates and track ID
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2) 

            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
            # Display track ID
            cv2.putText(frame, f'Track ID: {track_id}', (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Cut out the car image
            car_image = frame[int(y1):int(y2), int(x1):int(x2)]

            # cv2.imshow("Car Image", car_image)



            # Detect license plates within the car image
            license_plates = license_plate_detector(car_image)[0]

            # Process license plates detected in the car image
            for plate_idx, license_plate in enumerate(license_plates.boxes.data.tolist()):
                x1_lp, y1_lp, x2_lp, y2_lp, score_lp, _ = license_plate  # Unpack license plate variables
                    
                # Convert license plate coordinates to match the frame coordinates
                x1_lp += x1
                x2_lp += x1
                y1_lp += y1
                y2_lp += y1

                cv2.rectangle(frame, (int(x1_lp), int(y1_lp)), (int(x2_lp), int(y2_lp)), (0, 0, 255), 2)

                # Crop license plate
                license_plate_crop = frame[int(y1_lp):int(y2_lp), int(x1_lp):int(x2_lp), :]
                # cv2.imshow("license_plate_crop", license_plate_crop)

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Predict characters in license plate crop using a model named extract_plate.pt
                p2 = read_plate.predict(license_plate_crop)

                # Extract the predicted bounding boxes for characters
                res2 = p2[0].boxes.data.cpu().numpy().astype(np.float64)
                sorted_res2 = res2[res2[:, 0].argsort()]

                # Extract predicted characters and apply necessary substitutions
                predicted_characters = sorted_res2[:, -1]
                substituted_characters = [number_to_substitute.get(char, str(int(char))) for char in predicted_characters]
                print(f"frame_nmr: {frame_nmr}, substituted_characters: {substituted_characters}")

                # Extract confidence scores
                confidence_scores = sorted_res2[:, 4]

                # Group characters by license plate
                plate_key = (frame_nmr, track_id)
                if plate_key not in plate_characters:
                    plate_characters[plate_key] = {'characters': substituted_characters, 'confidence_scores': confidence_scores}
                else:
                    # Retrieve the previously stored confidence scores
                    existing_confidence_scores = plate_characters[plate_key]['confidence_scores']
                    
                    # Determine the length of the existing confidence scores
                    existing_length = len(existing_confidence_scores)
                    
                    # Ensure that the length of confidence_scores matches the length of existing_confidence_scores
                    if len(confidence_scores) != existing_length:
                        if len(confidence_scores) < existing_length:
                            confidence_scores = np.pad(confidence_scores, (0, existing_length - len(confidence_scores)), mode='constant', constant_values=-1)
                        else:
                            confidence_scores = confidence_scores[:existing_length]
                    
                    # Update the confidence scores in plate_characters using np.maximum()
                    plate_characters[plate_key]['confidence_scores'] = np.maximum(existing_confidence_scores, confidence_scores)
    
    # Display frame
    cv2.imshow('Stream 1', frame)       
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):     
        break


# Initialize a dictionary to store the consensus characters for each license plate
consensus_characters = {}

all_confidence = {}

# Iterate over the detected characters for each license plate
for plate_idx, predictions in plate_characters.items():
    char_confidence = {}

    # Concatenate predicted characters
    predicted_characters = ''.join(predictions['characters'])

    # Store confidence scores for each character
    for position, char in enumerate(predicted_characters):
        char_confidence[position] = {'char': char, 'score': predictions['confidence_scores'][position]}

    all_confidence[plate_idx] = char_confidence

# Iterate over all_confidence to find the most confident characters
for plate_idx, predictions in all_confidence.items():
    max_score = -1
    max_char = None

    # Concatenate predicted characters
    predicted_characters = ''.join([prediction['char'] for position, prediction in sorted(predictions.items())])

    # Find the character with the maximum confidence score
    for position, prediction in predictions.items():
        if prediction['score'] > max_score:
            max_score = prediction['score']
            max_char = prediction['char']

    # Store the most confident character and its confidence score
    consensus_characters[plate_idx] = (predicted_characters, max_score)

# Sort the consensus characters based on confidence score (optional)
sorted_consensus_characters = sorted(consensus_characters.items(), key=lambda x: x[1][1], reverse=True)



# Create an instance of the CarSearch class
car_search = CarSearch()

# Write the consensus characters and perform car search and plate addition
with open('python/results/final_results.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Frame Number', 'Track ID', 'Top Predicted Characters', 'Top Confidence Score'])

    # Iterate over each track ID
    for track_id in sorted(set(track_id for frame_nmr, track_id in plate_characters.keys())):
        # Filter consensus characters by track ID
        consensus_characters_track = {plate_idx: (predicted_characters, score) 
                                      for plate_idx, (predicted_characters, score) in consensus_characters.items() if plate_idx[1] == track_id}
        
        # Sort the consensus characters for the current track ID based on confidence score (optional)
        sorted_consensus_characters_track = sorted(consensus_characters_track.items(), key=lambda x: x[1][1], reverse=True)
        
        # Check if there are predictions for the current track ID
        if sorted_consensus_characters_track:
            # Get the top 1 most confident characters for the current track ID
            plate_idx, (predicted_characters, score) = sorted_consensus_characters_track[0]
            frame_nmr, track_id = plate_idx
            
            # Write the data to the CSV file
            csv_writer.writerow([frame_nmr, track_id, predicted_characters, score])
            
            # Search for the car by plate
            user_id, car_id = car_search.search_cars_by_plate(predicted_characters)
            
            if user_id is not None:
                # If the car is found, add the plate
                station_id = '8W5t2M4sCP5tTww53Xlp'  # Replace 'YourStationIDHere' with the actual station ID
                car_search.add_plate(user_id, station_id, predicted_characters, car_id)
            else:
                print(f"Car not found with plate {predicted_characters}.")
        else:
            # If no predictions found for the current track ID, write default values to the CSV file
            csv_writer.writerow(['N/A', track_id, 'N/A', -1])

print("Final results stored in 'final_results.csv'")



# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()



