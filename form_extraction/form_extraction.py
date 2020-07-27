import cv2
import numpy as np
import matplotlib.pyplot as plt


BS_CB = []
MS_CB = []
PHD_CB = []

ID = []
FN = []
LN = []

ID_block = []
FN_block = []
LN_block = []


#perspective transform on form picture to 480*480 image
def transform(img):
    #output size
    width = 480
    length = 480

    # Load the dictionary that was used to generate the markers.
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()
    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

    arr = np.concatenate(markerCorners, axis=0)

    for x in range(4):
        if markerIds[x] == 34:
            ul = arr[x, 0, 0], arr[x, 0, 1]
        if markerIds[x] == 33:
            dl = arr[x, 3, 0], arr[x, 3, 1]
        if markerIds[x] == 35:
            ur = arr[x, 1, 0], arr[x, 1, 1]
        if markerIds[x] == 36:
            dr = arr[x, 2, 0], arr[x, 2, 1]

    pts1 = np.float32([ur, dr, ul, dl])
    pts2 = np.float32([[width, 0], [width, length], [0, 0], [0, length]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (width, length))
    plt.imshow(dst)
    plt.show()
    cv2.imwrite("Trans_out.jpg",dst)
    return dst

#extract tabel contents from transform output
def table_detection(img):


    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #extract edges with canny edge detector
    canny = cv2.Canny(img_gray,80,200)
    plt.imshow(canny)
    plt.show()

    #canny output >> binary image
    img_bin=canny

    #find countours
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Copy image for drawing bounding box
    bb=img.copy()
    global ID_block
    global FN_block
    global LN_block

    #condition on contours
    for c in contours:

        x, y, w, h = cv2.boundingRect(c)
        if (w > 250 and 40> h >20):
            if(130<y<165):
                ID_block.append(img[y+3:y + h-3, x:x + w])
                #print("ID ok")

            if (170 < y < 210):
                FN_block.append(img[y+3:y + h-3, x:x + w])
                #print("FN ok")

            if (215 < y < 250):
                LN_block.append(img[y+3:y + h-3, x:x + w])
                #print("LN OK")

            bb = cv2.rectangle(bb, (x, y), (x + w, y + h), (0, 255, 0), 1)



        if ( 275 < y < 400 and  12<h<25 and 12<w<25):
            if(20<x<100):
                PHD_CB.append(img[y:y + h, x:x + w])
                # print("PHD ok")
                cv2.imwrite("PHD.png",PHD_CB[0])
            elif(100<x<200):
                MS_CB.append(img[y:y + h, x:x + w])
                # print("Ms ok")
                cv2.imwrite("MS.png", MS_CB[0])
            elif(220<x<350):
                BS_CB.append(img[y:y + h, x:x + w])
                # print("Bs ok")
                cv2.imwrite("BS.png", BS_CB[0])

            bb = cv2.rectangle(bb, (x, y), (x + w, y + h), (255, 0, 0), 1)

    plt.imshow(bb)
    plt.show()


    #visulaise and save table contants

    fig, axs = plt.subplots(4, 8)
    fig.suptitle('form Output')


    ID1 = ID_block[0][:, 0:int(ID_block[0].shape[1] / 8)]
    cv2.imwrite("ID1.png",ID1)
    axs[0, 0].imshow(ID1)
    #print(clf(ID1))


    ID2 = ID_block[0][:, int(ID_block[0].shape[1] / 8): 2* int(ID_block[0].shape[1] / 8)]
    cv2.imwrite("ID2.png", ID2)
    axs[0, 1].imshow(ID2)

    ID3 = ID_block[0][:, 2*int(ID_block[0].shape[1] / 8): 3* int(ID_block[0].shape[1] / 8)]
    cv2.imwrite("ID3.png", ID3)
    axs[0, 2].imshow(ID3)

    ID4 = ID_block[0][:, 3*int(ID_block[0].shape[1] / 8): 4* int(ID_block[0].shape[1] / 8)]
    cv2.imwrite("ID4.png", ID4)
    axs[0, 3].imshow(ID4)

    ID5 = ID_block[0][:, 4*int(ID_block[0].shape[1] / 8): 5* int(ID_block[0].shape[1] / 8)]
    cv2.imwrite("ID5.png", ID5)
    axs[0, 4].imshow(ID5)

    ID6 = ID_block[0][:, 5*int(ID_block[0].shape[1] / 8): 6* int(ID_block[0].shape[1] / 8)]
    cv2.imwrite("ID6.png", ID6)
    axs[0, 5].imshow(ID6)

    ID7 = ID_block[0][:, 6*int(ID_block[0].shape[1] / 8): 7* int(ID_block[0].shape[1] / 8)]
    cv2.imwrite("ID7.png", ID7)
    axs[0, 6].imshow(ID7)

    ID8 = ID_block[0][:, 7*int(ID_block[0].shape[1] / 8): 8* int(ID_block[0].shape[1] / 8)]
    cv2.imwrite("ID8.png", ID8)
    axs[0, 7].imshow(ID8)

    global ID
    ID = [ID1,ID2,ID2,ID3,ID4,ID5,ID6,ID7,ID8]


    FN1 = FN_block[0][:, 0:int(FN_block[0].shape[1] / 8)]
    cv2.imwrite("FN1.png",FN1)
    axs[1, 0].imshow(FN1)

    FN2 = FN_block[0][:, int(FN_block[0].shape[1] / 8):2 * int(FN_block[0].shape[1] / 8)]
    cv2.imwrite("FN2.png", FN2)
    axs[1, 1].imshow(FN2)

    FN3 = FN_block[0][:, 2 * int(FN_block[0].shape[1] / 8): 3 * int(FN_block[0].shape[1] / 8)]
    cv2.imwrite("FN3.png", FN3)
    axs[1, 2].imshow(FN3)

    FN4 = FN_block[0][:, 3 * int(FN_block[0].shape[1] / 8): 4 * int(FN_block[0].shape[1] / 8)]
    cv2.imwrite("FN4.png", FN4)
    axs[1, 3].imshow(FN4)

    FN5 = FN_block[0][:, 4 * int(FN_block[0].shape[1] / 8): 5 * int(FN_block[0].shape[1] / 8)]
    cv2.imwrite("FN5.png", FN5)
    axs[1, 4].imshow(FN5)

    FN6 = FN_block[0][:, 5 * int(FN_block[0].shape[1] / 8): 6 * int(FN_block[0].shape[1] / 8)]
    cv2.imwrite("FN6.png", FN6)
    axs[1, 5].imshow(FN6)

    FN7 = FN_block[0][:, 6 * int(FN_block[0].shape[1] / 8): 7 * int(FN_block[0].shape[1] / 8)]
    cv2.imwrite("FN7.png", FN7)
    axs[1, 6].imshow(FN7)

    FN8 = FN_block[0][:, 7 * int(FN_block[0].shape[1] / 8): 8 * int(FN_block[0].shape[1] / 8)]
    cv2.imwrite("FN8.png", FN8)
    axs[1, 7].imshow(FN8)

    global FN
    FN = [FN1,FN2,FN3,FN4,FN5,FN6,FN7,FN8]


    LN1 = LN_block[0][:, 0:int(LN_block[0].shape[1] / 8)]
    cv2.imwrite("LN1.png",LN1)
    axs[2, 0].imshow(LN1)

    LN2 = LN_block[0][:, int(LN_block[0].shape[1] / 8):2 * int(LN_block[0].shape[1] / 8)]
    cv2.imwrite("LN2.png", LN2)
    axs[2, 1].imshow(LN2)

    LN3 = LN_block[0][:, 2 * int(LN_block[0].shape[1] / 8): 3 * int(LN_block[0].shape[1] / 8)]
    cv2.imwrite("LN3.png", LN3)
    axs[2, 2].imshow(LN3)

    LN4 = LN_block[0][:, 3 * int(LN_block[0].shape[1] / 8): 4 * int(LN_block[0].shape[1] / 8)]
    cv2.imwrite("LN4.png", LN4)
    axs[2, 3].imshow(LN4)

    LN5 = LN_block[0][:, 4 * int(LN_block[0].shape[1] / 8): 5 * int(LN_block[0].shape[1] / 8)]
    cv2.imwrite("LN5.png", LN5)
    axs[2, 4].imshow(LN5)

    LN6 = LN_block[0][:, 5 * int(LN_block[0].shape[1] / 8): 6 * int(LN_block[0].shape[1] / 8)]
    cv2.imwrite("LN6.png", LN6)
    axs[2, 5].imshow(LN6)

    LN7 = LN_block[0][:, 6 * int(LN_block[0].shape[1] / 8): 7 * int(LN_block[0].shape[1] / 8)]
    cv2.imwrite("LN7.png", LN7)
    axs[2, 6].imshow(LN7)

    LN8 = LN_block[0][:, 7 * int(LN_block[0].shape[1] / 8): 8 * int(LN_block[0].shape[1] / 8)]
    cv2.imwrite("LN8.png", LN8)
    axs[2, 7].imshow(LN8)

    global LN
    LN = [LN1,LN2,LN3,LN4,LN5,LN6,LN7,LN8]

    axs[3,2].imshow(BS_CB[0])

    axs[3, 1].imshow(MS_CB[0])

    axs[3, 0].imshow(PHD_CB[0])



    plt.show()


if __name__ == "__main__":

    addr = "examples/From (1).jpg"

    img = cv2.imread(addr)

    dst = transform(img)


    table_detection(dst)