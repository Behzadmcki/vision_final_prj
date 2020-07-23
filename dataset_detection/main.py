import  cv2
import numpy as np

fnames1 = glob.glob("/home/joker/computerVision/final_project/vision_final_prj/dataset/Final project dataset/*_1*")
fnames2 = glob.glob("/home/joker/computerVision/final_project/vision_final_prj/dataset/Final project dataset/*_2*")

fnames1.sort()
fnames2.sort()

frame=cv2.imread("/home/joker/computerVision/final_project/vision_final_prj/dataset/9527533_2d.jpg")
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
output_size = (320,480)
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
arr=np.concatenate(markerCorners, axis=0 )
tx = 10
ty = 10


# print(markerIds)
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

# use an affine transformation matrix (2x3)
M1 = np.array([[1, 0, tx],
              [0, 1, ty]]).astype(np.float32)
M2= np.array([[1, 0, -tx],
              [0, 1, -ty]]).astype(np.float32)
output_size = (330,490) # output image size

pts = np.float32([[330, 0],
                  [0, 0],
                  [330, 490],
                  [0, 490]]).reshape(-1, 1, 2)
print(pts)
print(points1)
print(points1)
points2 = np.array([(325,5),(5,5),(325,485),(5,485)], dtype=np.float32)

H1 = cv2.getPerspectiveTransform(points1, points2)


dst1 = cv2.perspectiveTransform(pts,np.linalg.inv(H1)).reshape(4,2)

H3= cv2.getPerspectiveTransform(dst1, points2)

J1 = cv2.warpPerspective(frame,H3,  output_size)

cv2.imshow('frame', J1)
cv2.waitKey(0)


cv2.circle(frame,ul, 20, (255, 0, 0), -1)
cv2.circle(frame,ur, 20, (255, 0, 0), -1)
cv2.circle(frame,dr, 20, (255, 0, 0), -1)
cv2.circle(frame,dl, 20, (255, 0, 0), -1)
cv2.imshow("frame", frame)
cv2.imwrite("test.jpg", J1)
cv2.waitKey(0)



