import  cv2
import numpy as np
frame=cv2.imread("/home/joker/computerVision/final_project/vision_final_prj/dataset/9527533_1d.jpg")
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
arr=np.concatenate(markerCorners, axis=0 )

# arr=np.array([[markerIds[0],markerIds[1],markerIds[2],markerIds[3]]])
# print(arr[1,0,0])
# for i in range (4):
#     if markerIds[i]==30:
#         p1 = markerCorners[i].mean(1).squeeze()[0], markerCorners[0].mean(1).squeeze()[1]
#     if markerIds[i]==32:
#         p2 = markerCorners[i].mean(1).squeeze()[0], markerCorners[1].mean(1).squeeze()[1]
#     if markerIds[i] == 31:
#         p3 = markerCorners[i].mean(1).squeeze()[0], markerCorners[2].mean(1).squeeze()[1]
#     if markerIds[i] == 33:
#         p4 = markerCorners[i].mean(1).squeeze()[0], markerCorners[3].mean(1).squeeze()[1]
output_size = (480,320)

print(markerIds)


for x in range(4):
    if markerIds[x]==30:
        ul=arr[x,0,0],arr[x,0,1]
    if markerIds[x]==32:
        dl=arr[x,3,0],arr[x,3,1]
    if markerIds[x]==31:
        ur=arr[x,1,0],arr[x,1,1]
    if markerIds[x]==33:
        dr=arr[x,2,0],arr[x,2,1]

points1 = np.array([ur,ul,dr,dl], dtype=np.float32)
points2 = np.array([(480,0),(0,0),(480,320),(0,320)], dtype=np.float32)
H = cv2.getPerspectiveTransform(points1, points2)
J = cv2.warpPerspective(frame,H,  output_size)

cv2.imshow('frame', J);
cv2.waitKey(0)

cv2.circle(frame,ul, 20, (255, 0, 0), -1)
cv2.circle(frame,ur, 20, (255, 0, 0), -1)
cv2.circle(frame,dr, 20, (255, 0, 0), -1)
cv2.circle(frame,dl, 20, (255, 0, 0), -1)
cv2.imshow("frame", frame)
cv2.waitKey(0)
# cv2.waitKey(1000)
# p1 = markerCorners[0]
# print(p1[0])

