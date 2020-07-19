import cv2
import numpy as np
import matplotlib.pyplot as plt

file = "test.jpg"
i=hs=ws=0
im1 = cv2.imread(file, 0)
im = cv2.imread(file)

ret, thresh_value = cv2.threshold(im1, 205, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Binary Threshold', thresh_value)
cv2.waitKey(0)

# size = 100  # bilateral filter size (diameter)
# sigma_color = 10.5
# sigma_space = 10
# thresh_value = cv2.bilateralFilter(thresh_value,size, sigma_color, sigma_space)
# cv2.imshow('After bi..filter', thresh_value)
# cv2.waitKey(0)



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

# ## opening
# #kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# kernel = np.ones((2,1),np.uint8)
# dilated_value = cv2.morphologyEx(dilated_value, cv2.MORPH_OPEN, kernel)
# cv2.imshow('After Openning', dilated_value)
# cv2.waitKey(0)

# ## closing
# kernel = np.ones((2,2),np.uint8)
# dilated_value = cv2.morphologyEx(dilated_value, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('After Closing', dilated_value)
# cv2.waitKey(0)

cv2.imshow('after  dilation', dilated_value)
cv2.waitKey(0)




_, contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cordinates = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if 30>w>15 :
        if 30>h>15 :
            i=i+1
            cordinates.append((x, y, w, h))
             # bounding the images
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)
print(i)
plt.imshow(im)
print(im.shape)
# cv2.namedWindow('detecttable', cv2.WINDOW_NORMAL)
# cv2.imwrite('detecttable.jpg', im)
cv2.imshow('after bounding the images  ', im)
cv2.waitKey(0)