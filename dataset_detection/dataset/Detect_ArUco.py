import cv2

addr = "9527533_2c.jpg"

frame = cv2.imread(addr)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

# Load the dictionary that was used to generate the markers.
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters_create()
# Detect the markers in the image
markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

print("{} ArUco marker were detected.".format(len(markerCorners)))
assert len(markerCorners) == 4, "You have detected {} instead of 4 ArUco markers!".format(len(markerCorners))

p1 = markerCorners[0].mean(1).squeeze()[0], markerCorners[0].mean(1).squeeze()[1]
p2 = markerCorners[1].mean(1).squeeze()[0], markerCorners[1].mean(1).squeeze()[1]
p3 = markerCorners[2].mean(1).squeeze()[0], markerCorners[2].mean(1).squeeze()[1]
p4 = markerCorners[3].mean(1).squeeze()[0], markerCorners[3].mean(1).squeeze()[1]

cv2.circle(frame, p1, 20, (255, 0, 0), -1)
cv2.circle(frame, p2, 20, (0, 255, 0), -1)
cv2.circle(frame, p3, 20, (0, 0, 255), -1)
cv2.circle(frame, p4, 20, (255, 0, 255), -1)

cv2.imshow("frame", frame)
#cv2.imwrite("frame.png", frame)
print('framed was saved as "frame.png" on the disk.')
cv2.waitKey()

cv2.destroyAllWindows()
