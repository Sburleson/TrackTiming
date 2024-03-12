# TrackTiming
raspberrypi track and field timing using opencv

2/12/24
Added some testing code into the TestCode folder along with a few pictures from google that I was using to test on. The code takes in the picture and outputs the detected lines, now working on detecting people.

2/19
Testing the person detection using YOLO. Added test code and depended files in the TestCode folder.

2/26
Now combining function created in Opencv.py with YOLO.py to output both Athlete and line detection. Aswell as wrote PiCamera.py to be able to use the raspberry pi camera moudule. Still working on finding best filter values and camera angle. 

3/11
Trying out OpenPose instead of Yolo bounding box to hopefully get a more accurate reading on athleat positioning. Test Code folder updated to reflect this.
