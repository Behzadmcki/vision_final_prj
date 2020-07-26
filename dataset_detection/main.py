from builtins import type

import  cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from copy import copy, deepcopy

i=hs=ws=0
output_size = (320,480)
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
tx = 10
ty = 10
index=0
dsize=(28,28)
fnames = glob.glob("/home/joker/computerVision/final_project/Final project dataset/*")

for file in fnames:
    path = file
    # print(path[-6])
    if path[-6]=="1":type=1
    elif path[-6]=="2":type=2

    frame=cv2.imread(file)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    arr=np.concatenate(markerCorners, axis=0 )

    # baraye dar avordane char gooshe aruco bar asase arocu ha

    # print(file)
    for x in range(4):
        if markerIds[x]==30:
            ul=arr[x,0,0],arr[x,0,1]
        if markerIds[x]==32:
            dl=arr[x,3,0],arr[x,3,1]
        if markerIds[x]==31:
            ur=arr[x,1,0],arr[x,1,1]
        if markerIds[x]==33:
            dr=arr[x,2,0],arr[x,2,1]
    # chargooshe tasvir
    points1 = np.array([ur,ul,dr,dl], dtype=np.float32)

    # use an affine transformation matrix (2x3)
    M1 = np.array([[1, 0, tx],
                  [0, 1, ty]]).astype(np.float32)

    output_size = (330,490) # output image size

    pts = np.float32([[330, 0],
                      [0, 0],
                      [330, 490],
                      [0, 490]]).reshape(-1, 1, 2)

    points2 = np.array([(325,5),(5,5),(325,485),(5,485)], dtype=np.float32)

    H1 = cv2.getPerspectiveTransform(points1, points2)

    dst1 = cv2.perspectiveTransform(pts,np.linalg.inv(H1)).reshape(4,2)

    H3= cv2.getPerspectiveTransform(dst1, points2)

    J1 = cv2.warpPerspective(frame,H3,  output_size)

    # cv2.imshow('frame', J1)
    # cv2.waitKey(0)
    im  = J1.copy()
    im1 = J1.copy()
    im2 = J1.copy()

    cv2.circle(frame,ul, 20, (255, 0, 0), -1)
    cv2.circle(frame,ur, 20, (255, 0, 0), -1)
    cv2.circle(frame,dr, 20, (255, 0, 0), -1)
    cv2.circle(frame,dl, 20, (255, 0, 0), -1)

    # cv2.imshow("frame", frame)
    # cv2.waitKey(0)

    thresh_value = cv2.Canny(im,100,200)
    # cv2.imshow('Binary Threshold', thresh_value)
    # cv2.waitKey(0)

    kernel = np.ones((4,1),np.uint8)
    Ydilated_value = cv2.dilate(thresh_value, kernel, iterations=1)
    # cv2.imshow('after y dilation', Ydilated_value)
    # cv2.waitKey(0)

    kernel = np.ones((1,5),np.uint8)
    Xdilated_value = cv2.dilate(thresh_value, kernel, iterations=1)
    # cv2.imshow('after x dilation', Xdilated_value)
    # cv2.waitKey(0)

    dilated_value=Ydilated_value + Xdilated_value
    # cv2.imshow('after  X&Y  dilation  ', dilated_value)
    # cv2.waitKey(0)

    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(dilated_value,kernel,iterations = 1)
    # cv2.imshow('after  Erosion  ', dilated_value)
    # cv2.waitKey(0)


    # _, contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cordinates = []
    # for cnt in contours:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     if 25>w>15 :
    #         if 25>h>15 :
    #             # cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             index=index+1
    #             if 0<y<30:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/0"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/4"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 25<y<50:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/1"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/5"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 45<y<70:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/10"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/27"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 65<y<90:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/11"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/28"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 85<y<110:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/12"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/29"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 115<y<140:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/13"
    #                 elif type==2 :save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/30"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 135<y<160:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/14"
    #                 elif type==2 :save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/31"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 155 < y < 180:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/15"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/32"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 175<y<200:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/16"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/33"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 195<y<220:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/17"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/34"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 115<y<240:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/18"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/35"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 135<y<265:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/19"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/36"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 155 < y < 285:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/20"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/37"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 175 < y < 305:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/21"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/38"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 195 < y < 325:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/22"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/39"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 315< y < 345:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/23"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/40"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 335 < y < 365:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/24"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/41"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 355 < y < 385:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/25"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/6"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 375 < y < 405:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/26"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/7"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 395 < y < 425:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/2"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/8"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 315< y < 445:
    #                 if type==1:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/3"
    #                 elif type==2:save_path="/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/9"
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 crop_img = cv2.resize(crop_img, dsize)
    #                 cordinates.append((x, y, w, h))
    #                 cv2.imwrite(save_path+"/"+str(index)+".jpg",crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #             if 335 < y < 465:
    #                 crop_img = im1[y:y + h, x:x + w]
    #                 cordinates.append((x, y, w, h))
    #                 # cv2.imwrite(save_path,crop_img)
    #                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)


# cv2.imshow('after bounding the images  ', im)
# cv2.waitKey(0)
# cv2.imshow("",im2)
# cv2.waitKey(0)



