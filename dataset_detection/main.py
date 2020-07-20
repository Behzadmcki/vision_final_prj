import  cv2
import numpy as np
frame=cv2.imread("/home/joker/computerVision/final_project/vision_final_prj/dataset/9527533_2a.jpg")
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
arr=np.concatenate(markerCorners, axis=0 )

output_size = (320,480)


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



# if ur[0]-ul[0]>0:
#     if ur[1]-dr[1]>0:
#         UP_RIGHT = int(ur[0]+10),int( ur[1] + 10 )
#         DOWN_RIGHT= int(dr[0]+10),int(dr[1] - 10)
#         UP_LEFT = int(ul[0]-10), int(ul[1] + 10)
#         DOWN_LEFT = int(dl[0]-10), int(dl[1] - 10)
#
#     else:
#         UP_RIGHT = int(ur[0]+10),int( ur[1] - 10 )
#         DOWN_RIGHT= int(dr[0]+10),int(dr[1] + 10)
#         UP_LEFT = int(ul[0]-10), int(ul[1] - 10)
#         DOWN_LEFT = int(dl[0]-10), int(dl[1] + 10)
# else:
#     if ur[1]-dr[1]>0:
#         UP_RIGHT = int(ur[0]-10),int( ur[1] + 10 )
#         DOWN_RIGHT= int(dr[0]-10),int(dr[1] - 10)
#         UP_LEFT = int(ul[0]+10), int(ul[1] + 10)
#         DOWN_LEFT = int(dl[0]+10), int(dl[1] - 10)
#     else:
#         UP_RIGHT = int(ur[0]-10),int( ur[1] - 10 )
#         DOWN_RIGHT= int(dr[0]-10),int(dr[1] + 10)
#         UP_LEFT = int(ul[0]+10), int(ul[1] - 10)
#         DOWN_LEFT = int(dl[0]+10), int(dl[1] + 10)

# if ul[0]-ur[0]>0:
#     if ul[1]-dl[1]>0:
#         UP_LEFT = int(ul[0]+10),int(ul[1] + 10)
#     else:
#         UP_LEFT = int(ul[0]+10),int(ul[1] - 10)
# else:
#     if ul[1]-dl[1]>0:
#         UP_LEFT = int(ul[0]-10),int(ul[1] + 10)
#     else:
#         UP_LEFT = int(ul[0]-10),int(ul[1] - 10)
#
# if dl[0]-dr[0]>0:
#     if dl[1]-ul[1]>0:
#         DOWN_LEFT = int(dl[0]+10),int(dl[1] + 10)
#     else:
#         DOWN_LEFT = int(dl[0]+10),int(dl[1] - 10)
# else:
#     if dl[1]-ul[1]>0:
#         DOWN_LEFT = int(dl[0]-10),int(dl[1] + 10)
#     else:
#         DOWN_LEFT = int(dl[0]-0),int(dl[1] - 10)
#
# if dr[0]-dl[0]>0:
#     if dr[1]-ur[1]>0:
#         DOWN_RIGHT= int(dr[0]+10),int(dr[1] + 10)
#     else:
#         DOWN_RIGHT = int(dr[0]+10),int(dr[1] +10)
# else:
#     if dr[1]-ur[1]>0:
#         DOWN_RIGHT = int(dr[0]-10),int(dr[1] + 10)
#     else:
#         DOWN_RIGHT = int(dr[0]-10),int(dr[1] - 10)
tx = 10
ty = 10

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
# points1 = np.array([UP_RIGHT,UP_LEFT,DOWN_RIGHT,DOWN_LEFT], dtype=np.float32)
print(points1)
points2 = np.array([(325,5),(5,5),(325,485),(5,485)], dtype=np.float32)
points3 = np.array([(320,0),(0,0),(320,480),(0,480)], dtype=np.float32)

H1 = cv2.getPerspectiveTransform(points1, points2)
H2 = cv2.getPerspectiveTransform(points1, points3)

dst1 = cv2.perspectiveTransform(pts,np.linalg.inv(H1)).reshape(4,2)
dst2 = cv2.perspectiveTransform(pts,np.linalg.inv(H2)).reshape(4,2)

H3= cv2.getPerspectiveTransform(dst1, points2)
H4= cv2.getPerspectiveTransform(dst2, points2)

# print(dst)

J1 = cv2.warpPerspective(frame,H3,  output_size)
J2 = cv2.warpPerspective(frame,H4,  output_size)
cv2.imshow('frame', J2)
cv2.waitKey(0)
cv2.imshow('frame', J1)
cv2.waitKey(0)

# J1= cv2.warpAffine(J2,M2,output_size)
# J2= cv2.warpAffine(J2,M1,output_size)

# cv2.imshow('frame', J1)
# cv2.waitKey(0)
#
# cv2.imshow('frame', J2)
# cv2.waitKey(0)

# J=J1+J2
cv2.imshow('frame', J1);
cv2.imwrite("test.jpg", J1)
cv2.waitKey(0)
cv2.circle(frame,ul, 20, (255, 0, 0), -1)
cv2.circle(frame,ur, 20, (255, 0, 0), -1)
cv2.circle(frame,dr, 20, (255, 0, 0), -1)
cv2.circle(frame,dl, 20, (255, 0, 0), -1)
cv2.imshow("frame", frame)
cv2.waitKey(0)



