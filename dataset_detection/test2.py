import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


addr = "/home/joker/computerVision/final_project/vision_final_prj/dataset/9527533_1a.jpg"
frame = cv2.imread(addr)
cv2.imshow("",frame)
cv2.waitKey(0)

# Load the dictionary that was used to generate the markers.
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters_create()
# Detect the markers in the image
markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
arr = np.concatenate(markerCorners, axis=0)
for x in range(4):
    if markerIds[x] == 30:
        ul = arr[x, 0, 0], arr[x, 0, 1]
    if markerIds[x] == 32:
        dl = arr[x, 3, 0], arr[x, 3, 1]
    if markerIds[x] == 31:
        ur = arr[x, 1, 0], arr[x, 1, 1]
    if markerIds[x] == 33:
        dr = arr[x, 2, 0], arr[x, 2, 1]
points1 = np.array([ur,ul,dr,dl], dtype=np.float32)


