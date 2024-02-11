import cv2
import numpy as np

##PLAN
##
##Take each frame from video, detect track lanes and set finish line.
##


def filter_horizontal_lines(lines):
    horizontal_lines = []
    for line in lines:
        print("_________")
        x1, y1, x2, y2 = line[0]

        # Calculate the slope of the line
        slope = (y2 - y1) / (x2 - x1 + 1e-5)  # Adding a small number to avoid division by zero

        # Check if the slope is closeish to 0 (horizontal line)
        if abs(slope) < 0.1:
            print(line)
            horizontal_lines.append(line)

    return horizontal_lines


path = 'track.jpg'

image = cv2.imread(path)

cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
##cv2.imshow('Grayscale Image', gray)
##cv2.waitKey(0)

# Apply GaussianBlur to reduce noise and help the algorithm
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Display the blurred image
##cv2.imshow('Blurred Image', blur)
##cv2.waitKey(0)

# Use Canny edge detector to find edges in the image
edges = cv2.Canny(blur, 20, 100)

# Display the edges
cv2.imshow('Edges Detected', edges)
cv2.waitKey(0)

# Use Hough Line Transform to detect lines in the image
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=60, maxLineGap=15)

# Draw the lines on the original image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the final result
cv2.imshow('Lines Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(lines)
horizontal = filter_horizontal_lines(lines)
#print("Horizontal:", filter_horizontal_lines(lines))
