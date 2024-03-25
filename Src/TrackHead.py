import cv2
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n-pose.pt')

# Load input video
cap = cv2.VideoCapture("race_trimmed.mp4")

# Create tracker
tracker = cv2.TrackerGOTURN()  # You can use different trackers like cv2.TrackerMIL_create() or cv2.TrackerTLD_create()

# Dictionary to store trackers for each person
trackers = {}

# Flag to control pause state
paused = False

# Variable to keep track of frame number
frame_number = 0

while True:
    # Read a frame from the video if not paused
    if not paused:
        ret, frame = cap.read()
        frame_number += 1
    else:
        # If paused, wait for keypress to resume
        key = cv2.waitKey(0) & 0xFF
        if key == ord('p'):  # Resume playback on 'p' key press
            paused = False
        elif key == ord('q'):  # Quit on 'q' key press
            break

    if not ret:
        break

    # Run YOLO model to detect people and estimate poses
    results = model(frame)

    # Update trackers
    for result in results:
        keypoints = result.keypoints

        # Iterate over keypoints for each person detected
        for i, person_keypoints in enumerate(keypoints.xy):
            head_x, head_y = person_keypoints[0]
            head_confidence = keypoints.conf[i][0]
            
            person_id = f"Person_{i+1}"
            
            if person_id not in trackers:
                # Create a new tracker if person not tracked yet
                trackers[person_id] = cv2.TrackerGOTURN()
                bbox = (int(head_x) - 20, int(head_y) - 20, 40, 40)  #bounding box around head
                trackers[person_id].init(frame, bbox)
            else:
                # Update existing tracker
                success, bbox = trackers[person_id].update(frame)
                if success:
                    # Draw bounding box around the tracked head
                    bbox = tuple(map(int, bbox))
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                else:
                    # If tracking fails, delete the tracker
                    del trackers[person_id]

            print(f"Frame {frame_number} - {person_id} - Head Coordinates: ({head_x}, {head_y}), Confidence: {head_confidence}")

            # Draw a circle at the head position
            cv2.circle(frame, (int(head_x), int(head_y)), radius=5, color=(0, 255, 0), thickness=-1)

    # Display the result
    cv2.imshow("YOLO Person Detection", frame)

    # Check for key press to pause or quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):  # Pause on 'p' key press
        paused = not paused
    elif key == ord('q'):  # Quit on 'q' key press
        break

cap.release()
cv2.destroyAllWindows()
