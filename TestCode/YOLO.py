import cv2
import numpy as np
import os

# Load YOLO
current_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(current_dir, "yolov3.weights")
config_path = os.path.join(current_dir, "yolov3.cfg")

# Load YOLO
net = cv2.dnn.readNet(weights_path,config_path)
output_layers = net.getUnconnectedOutLayersNames()

cap = cv2.VideoCapture("race_trimmed.mp4")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
skip_frames = int(total_frames / 1.2)
cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame is successfully read
    if not ret:
        break

    # Do something with the frame (e.g., display or process)

    # Display the frame (optional)
    cv2.imshow('Frame', frame)
    # Load image
    ##image = cv2.imread("runnerobj.jpg")  # Replace with your image path
    height, width, channels = frame.shape

    # Preprocess image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)


    # Get bounding boxes, confidence scores, and class IDs
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            ##print(scores)
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0 corresponds to person class
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                ##
                ##h = int(h * 0.8)  # Decrease height by 20%


                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes on the image
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = "Athlete"
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("YOLO Person Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(outs[0][0])
