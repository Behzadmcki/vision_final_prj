import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


tx = 10
ty = 10
index=0
num_page=0
dsize=(28,28)
fnames = glob.glob("dataset/*")
print(fnames)
# Load the dictionary that was used to generate the markers.
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters_create()



for file in fnames:
    num_page=num_page+1
    print(file)
    # find out what is the type of form
    path = file
    frame = cv2.imread(file)
    if path[-6]=="1":type=1
    elif path[-6]=="2":type=2
    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    # selec for corner of the paper according to aruco
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
    # use projective transformation to get full image of the form
    M1 = np.array([[1, 0, tx],
                   [0, 1, ty]]).astype(np.float32)
    output_size = (330, 490)  # output image size

    pts = np.float32([[330, 0],
                      [0, 0],
                      [330, 490],
                      [0, 490]]).reshape(-1, 1, 2)

    points2 = np.array([(325, 5), (5, 5), (325, 485), (5, 485)], dtype=np.float32)

    H1 = cv2.getPerspectiveTransform(points1, points2)

    dst1 = cv2.perspectiveTransform(pts, np.linalg.inv(H1)).reshape(4, 2)

    H3 = cv2.getPerspectiveTransform(dst1, points2)

    J1 = cv2.warpPerspective(frame, H3, output_size)# this is full image


    # find Aruco on more time in full page of form
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(J1, dictionary, parameters=parameters)
    arr = np.concatenate(markerCorners, axis=0)

    for x in range(4):
        if markerIds[x] == 30:
            p4 = markerCorners[x].mean(1).squeeze()[0], markerCorners[x].mean(1).squeeze()[1]
        if markerIds[x] == 32:
            p3 = markerCorners[x].mean(1).squeeze()[0], markerCorners[x].mean(1).squeeze()[1]
        if markerIds[x] == 31:
            p2 = markerCorners[x].mean(1).squeeze()[0], markerCorners[x].mean(1).squeeze()[1]
        if markerIds[x] == 33:
            p1 = markerCorners[x].mean(1).squeeze()[0], markerCorners[x].mean(1).squeeze()[1]

    #show each box of for width and height
    # print(((p1[0] -p3[0])/12) )
    # print(((p1[1] -p2[1])/19) )
    #use circle to find out where are p1,p2,p3,p4 on form
    cv2.circle(J1, p3, 20, (255, 0, 0), -1)
    #cut compleat  image of form into pieces
    for j in range(21):
        y = int(p4[1] + (j-1) * 22.54)+2
        print("********"+str(np.mod(index,21)))
        if j == 0:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/0"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/4"
        elif j == 1:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/1"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/5"
        elif j == 2:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/10"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/27"
        elif j == 3:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/11"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/28"
        elif j == 4:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/12"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/29"
        elif j == 5:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/13"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/30"
        elif j == 6:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/14"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/31"
        elif j == 7:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/15"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/32"
        elif j == 8:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/16"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/33"
        elif j == 9:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/17"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/34"
        elif j == 10:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/18"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/35"
        elif j == 11:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/19"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/36"
        elif j == 12:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/20"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/37"
        elif j == 13:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/21"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/38"
        elif j == 14:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/22"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/39"
        elif j == 15:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/23"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/40"
        elif j == 16:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/24"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/41"
        elif j == 17:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/25"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/6"
        elif j == 18:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/26"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/7"
        elif j == 19:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/2"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/8"
        elif j == 20:
            if type == 1:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/3"
            elif type == 2:
                save_path = "/home/joker/computerVision/final_project/vision_final_prj/processed_dataset/9"
        if j==0 or j==1 or j==19 or j==20:
            RNG=10
            initalize=p4[0]+2*22.375
        else:
            RNG=14
            initalize=p4[0]
        for i in range(RNG):
            x = int(initalize + (i-1) * 22.375)+2
            crop_img = J1[y:y + 22, x:x + 22]
            crop_img = cv2.resize(crop_img, dsize)
            print("***********************"+str(x))
            # cv2.imwrite(save_path + "/" + str(index) + ".jpg", crop_img)
            index = index + 1
            cv2.imshow("",crop_img)
            cv2.waitKey(0)

# cv2.imshow("",J1)
cv2.waitKey()
cv2.destroyAllWindows()



