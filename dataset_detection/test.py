import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from copy import copy, deepcopy


file = "test.jpg"
i=hs=ws=0
im1 = cv2.imread(file, 0)
im = cv2.imread(file)
im2=cv2.imread(file)
thresh_value = cv2.Canny(im,100,200)
cv2.imshow('Binary Threshold', thresh_value)
cv2.waitKey(0)

##dilation
#kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel = np.ones((4,1),np.uint8)
Ydilated_value = cv2.dilate(thresh_value, kernel, iterations=1)
cv2.imshow('after y dilation', Ydilated_value)
cv2.waitKey(0)

##dilation
#kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel = np.ones((1,5),np.uint8)
Xdilated_value = cv2.dilate(thresh_value, kernel, iterations=1)
cv2.imshow('after x dilation', Xdilated_value)
cv2.waitKey(0)


dilated_value=Ydilated_value + Xdilated_value
cv2.imshow('after  X&Y  dilation  ', dilated_value)
cv2.waitKey(0)

kernel = np.ones((2,2),np.uint8)
erosion = cv2.erode(dilated_value,kernel,iterations = 1)
cv2.imshow('after  Erosion  ', dilated_value)
cv2.waitKey(0)


_, contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cordinates = []
for cnt in contours:
    # cv2.imshow('countours',cnt)
    # cv2.waitKey(0)
    # print(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    if 30>w>15 :
        if 30>h>15 :
            i=i+1
            if 0<y<30:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 25<y<50:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 45<y<70:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 65<y<90:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 85<y<110:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 115<y<140:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 135<y<160:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 155 < y < 180:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 175<y<200:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 195<y<220:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 115<y<240:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 135<y<265:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 155 < y < 285:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 175 < y < 305:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 195 < y < 325:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 315< y < 345:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 335 < y < 365:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 355 < y < 385:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 375 < y < 405:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 395 < y < 425:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 315< y < 445:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
            if 335 < y < 465:
                crop_img = im1[y:y + h, x:x + w]
                cordinates.append((x, y, w, h))
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255), 1)
print(i)
plt.imshow(im)
print(im.shape)
# cv2.namedWindow('detecttable', cv2.WINDOW_NORMAL)
# cv2.imwrite('detecttable.jpg', im)
cv2.imshow('after bounding the images  ', im)
cv2.waitKey(0)
cv2.imshow("",im2)
cv2.waitKey(0)
