import cv2
import numpy as np
from openpose import pyopenpose as op

# Set up OpenPose parameters
params = {
    "model_folder": "path/to/openpose/models/",
    "hand": False,
    "face": False,
}

# Initialize OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Read input video
cap = cv2.VideoCapture("path/to/your/video.mp4")

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame with OpenPose
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    # Extract keypoints and draw skeleton
    keypoints = datum.poseKeypoints
    frame = datum.cvOutputData

    if keypoints is not None and len(keypoints) > 0:
        # Draw keypoints and skeleton on the frame
        for person_keypoints in keypoints:
            for point in person_keypoints:
                cv2.circle(frame, tuple(point[:2].astype(int)), 5, (0, 255, 0), -1)
            for limb in op.PosePairs:
                if person_keypoints[limb[0].value].all() and person_keypoints[limb[1].value].all():
                    cv2.line(frame, tuple(person_keypoints[limb[0].value][:2].astype(int)),
                             tuple(person_keypoints[limb[1].value][:2].astype(int)), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("OpenPose Example", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
