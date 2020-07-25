import cv2
import numpy as np
import matplotlib.pyplot as plt

BS_CB = []
MS_CB = []
PHD_CB = []

ID = []
FN = []
LN = []




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

    # print(markerIds)

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
    # cv2.imwrite("Trans_out.jpg",dst)
    return dst


def table_detection(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(img_gray,100,200)
    plt.imshow(canny)
    plt.show()

    img_bin=canny
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bb=img.copy()

    for c in contours:

        x, y, w, h = cv2.boundingRect(c)
        if (w > 250 and 40> h >20):
            if(130<y<165):
                ID.append(img[y+3:y + h-3, x:x + w])
                #print("ID ok")

            if (170 < y < 210):
                FN.append(img[y+3:y + h-3, x:x + w])
                #print("FN ok")

            if (215 < y < 250):
                LN.append(img[y+3:y + h-3, x:x + w])
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
                # cv2.imwrite("MS.png", MS_CB[0])
            elif(220<x<350):
                BS_CB.append(img[y:y + h, x:x + w])
                # print("Bs ok")
                cv2.imwrite("BS.png", BS_CB[0])

            bb = cv2.rectangle(bb, (x, y), (x + w, y + h), (255, 0, 0), 1)

    plt.imshow(bb)
    plt.show()


    fig, axs = plt.subplots(4, 8)
    fig.suptitle('form Output')



    #
    ID1 = ID[0][:, 0:int(ID[0].shape[1] / 8)]
    axs[0, 0].imshow(ID1)


    ID2 = ID[0][:, int(ID[0].shape[1] / 8):2 * int(ID[0].shape[1] / 8)]
    axs[0, 1].imshow(ID2)

    ID3 = ID[0][:, 2*int(ID[0].shape[1] / 8): 3* int(ID[0].shape[1] / 8)]
    axs[0, 2].imshow(ID3)

    ID4 = ID[0][:, 3*int(ID[0].shape[1] / 8): 4* int(ID[0].shape[1] / 8)]
    axs[0, 3].imshow(ID4)

    ID5 = ID[0][:, 4*int(ID[0].shape[1] / 8): 5* int(ID[0].shape[1] / 8)]
    axs[0, 4].imshow(ID5)

    ID6 = ID[0][:, 5*int(ID[0].shape[1] / 8): 6* int(ID[0].shape[1] / 8)]
    axs[0, 5].imshow(ID6)

    ID7 = ID[0][:, 6*int(ID[0].shape[1] / 8): 7* int(ID[0].shape[1] / 8)]
    axs[0, 6].imshow(ID7)

    ID8 = ID[0][:, 7*int(ID[0].shape[1] / 8): 8* int(ID[0].shape[1] / 8)]
    axs[0, 7].imshow(ID8)



    FN1 = FN[0][:, 0:int(FN[0].shape[1] / 8)]
    axs[1, 0].imshow(FN1)

    FN2 = FN[0][:, int(FN[0].shape[1] / 8):2 * int(FN[0].shape[1] / 8)]
    axs[1, 1].imshow(FN2)

    FN3 = FN[0][:, 2 * int(FN[0].shape[1] / 8): 3 * int(FN[0].shape[1] / 8)]
    axs[1, 2].imshow(FN3)

    FN4 = FN[0][:, 3 * int(FN[0].shape[1] / 8): 4 * int(FN[0].shape[1] / 8)]
    axs[1, 3].imshow(FN4)

    FN5 = FN[0][:, 4 * int(FN[0].shape[1] / 8): 5 * int(FN[0].shape[1] / 8)]
    axs[1, 4].imshow(FN5)

    FN6 = FN[0][:, 5 * int(FN[0].shape[1] / 8): 6 * int(FN[0].shape[1] / 8)]
    axs[1, 5].imshow(FN6)

    FN7 = FN[0][:, 6 * int(FN[0].shape[1] / 8): 7 * int(FN[0].shape[1] / 8)]
    axs[1, 6].imshow(FN7)

    FN8 = FN[0][:, 7 * int(FN[0].shape[1] / 8): 8 * int(FN[0].shape[1] / 8)]
    axs[1, 7].imshow(FN8)


    LN1 = LN[0][:, 0:int(LN[0].shape[1] / 8)]
    axs[2, 0].imshow(LN1)

    LN2 = LN[0][:, int(LN[0].shape[1] / 8):2 * int(LN[0].shape[1] / 8)]
    axs[2, 1].imshow(LN2)

    LN3 = LN[0][:, 2 * int(LN[0].shape[1] / 8): 3 * int(LN[0].shape[1] / 8)]
    axs[2, 2].imshow(LN3)

    LN4 = LN[0][:, 3 * int(LN[0].shape[1] / 8): 4 * int(LN[0].shape[1] / 8)]
    axs[2, 3].imshow(LN4)

    LN5 = LN[0][:, 4 * int(LN[0].shape[1] / 8): 5 * int(LN[0].shape[1] / 8)]
    axs[2, 4].imshow(LN5)

    LN6 = LN[0][:, 5 * int(LN[0].shape[1] / 8): 6 * int(LN[0].shape[1] / 8)]
    axs[2, 5].imshow(LN6)

    LN7 = LN[0][:, 6 * int(LN[0].shape[1] / 8): 7 * int(LN[0].shape[1] / 8)]
    axs[2, 6].imshow(LN7)

    LN8 = LN[0][:, 7 * int(LN[0].shape[1] / 8): 8 * int(LN[0].shape[1] / 8)]
    axs[2, 7].imshow(LN8)

    axs[3,2].imshow(BS_CB[0])

    axs[3, 1].imshow(MS_CB[0])

    axs[3, 0].imshow(PHD_CB[0])

    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

    plt.show()

def checkbox(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # plt.imshow(img)
    # plt.show()


    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
    # plt.imshow(th2)
    # plt.show()

    #calculate number of Non zero pixel
    non_zero = cv2.countNonZero(th2)
    # print(non_zero)
    if non_zero > ((th2.shape[0]*th2.shape[1])/2):
        return True


    # ret1,th1 = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
    # plt.imshow(th1)
    # plt.show()



if __name__ == "__main__":

    addr = "examples/5.jpg"
    img = cv2.imread(addr)

    dst = transform(img)


    table_detection(dst)

    if(checkbox(BS_CB[0])):
        print("he has a BS degree")
    if(checkbox(MS_CB[0])):
        print("he has a MS degree")
    if(checkbox(PHD_CB[0])):
        print("he has a PHD degree")





    cv2.waitKey()
    cv2.destroyAllWindows()
