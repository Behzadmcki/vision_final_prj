import cv2
import numpy as np
# import matplotlib.pyplot as plt
#from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import arabic_reshaper
from bidi.algorithm import get_display
import time

model = load_model('BatchNorm_87.h5')


fa_ch = [u"۰",u"۱",u"۲",u"۳",u"۴",u"۵",u"۶",u"۷",u"۸",u"۹",u"ا",u"ب",u"پ",u"ت",u"ث",u"ج",u"چ",u"ح",u"خ",u"د",u"ذ",u"ر",u"ز",u"ژ",u"س",u"ش",u"ص",u"ض",u"ط",u"ظ",u"ع",u"غ",u"ف",u"ق",u"ک",u"گ",u"ل",u"م",u"ن",u"و",u"ه",u"ی",u" "]

BS_CB = []
MS_CB = []
PHD_CB = []

ID = []
FN = []
LN = []

ID_block = []
FN_block = []
LN_block = []

def fa_print(fa_string):
    reshaped_text = arabic_reshaper.reshape(fa_string)
    bidi_text = get_display(reshaped_text)
    print(bidi_text)


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
    # plt.imshow(dst)
    # plt.show()
    # cv2.imwrite("Trans_out.jpg",dst)
    return dst


def table_detection(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(img_gray,100,200)
    # plt.imshow(canny)
    # plt.show()

    img_bin=canny
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # bb=img.copy()
    global ID_block
    global FN_block
    global LN_block

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

            # bb = cv2.rectangle(bb, (x, y), (x + w, y + h), (0, 255, 0), 1)



        if ( 275 < y < 400 and  12<h<25 and 12<w<25):
            if(20<x<100):
                PHD_CB.append(img[y:y + h, x:x + w])
                # print("PHD ok")
                # cv2.imwrite("PHD.png",PHD_CB[0])
            elif(100<x<200):
                MS_CB.append(img[y:y + h, x:x + w])
                # print("Ms ok")
                # cv2.imwrite("MS.png", MS_CB[0])
            elif(220<x<350):
                BS_CB.append(img[y:y + h, x:x + w])
                # print("Bs ok")
                # cv2.imwrite("BS.png", BS_CB[0])

            # bb = cv2.rectangle(bb, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # plt.imshow(bb)
    # plt.show()


    # fig, axs = plt.subplots(4, 8)
    # fig.suptitle('form Output')


    ID1 = ID_block[0][:, 0:int(ID_block[0].shape[1] / 8)]
    ID2 = ID_block[0][:, int(ID_block[0].shape[1] / 8): 2* int(ID_block[0].shape[1] / 8)]
    ID3 = ID_block[0][:, 2*int(ID_block[0].shape[1] / 8): 3* int(ID_block[0].shape[1] / 8)]
    ID4 = ID_block[0][:, 3*int(ID_block[0].shape[1] / 8): 4* int(ID_block[0].shape[1] / 8)]
    ID5 = ID_block[0][:, 4*int(ID_block[0].shape[1] / 8): 5* int(ID_block[0].shape[1] / 8)]
    ID6 = ID_block[0][:, 5*int(ID_block[0].shape[1] / 8): 6* int(ID_block[0].shape[1] / 8)]
    ID7 = ID_block[0][:, 6*int(ID_block[0].shape[1] / 8): 7* int(ID_block[0].shape[1] / 8)]
    ID8 = ID_block[0][:, 7*int(ID_block[0].shape[1] / 8): 8* int(ID_block[0].shape[1] / 8)]

    # cv2.imwrite("ID1.png",ID1)
    # cv2.imwrite("ID2.png", ID2)
    # cv2.imwrite("ID3.png", ID3)
    # cv2.imwrite("ID4.png", ID4)
    # cv2.imwrite("ID5.png", ID5)
    # cv2.imwrite("ID6.png", ID6)
    # cv2.imwrite("ID7.png", ID7)
    #cv2.imwrite("ID8.png", ID8)


    # axs[0, 0].imshow(ID1)
    # axs[0, 1].imshow(ID2)
    # axs[0, 2].imshow(ID3)
    # axs[0, 3].imshow(ID4)
    # axs[0, 4].imshow(ID5)
    # axs[0, 7].imshow(ID8)
    # axs[0, 5].imshow(ID6)
    # axs[0, 6].imshow(ID7)


    global ID
    ID = [ID1,ID2,ID2,ID3,ID4,ID5,ID6,ID7,ID8]


    FN1 = FN_block[0][:, 0:int(FN_block[0].shape[1] / 8)]
    FN2 = FN_block[0][:, int(FN_block[0].shape[1] / 8):2 * int(FN_block[0].shape[1] / 8)]
    FN3 = FN_block[0][:, 2 * int(FN_block[0].shape[1] / 8): 3 * int(FN_block[0].shape[1] / 8)]
    FN4 = FN_block[0][:, 3 * int(FN_block[0].shape[1] / 8): 4 * int(FN_block[0].shape[1] / 8)]
    FN5 = FN_block[0][:, 4 * int(FN_block[0].shape[1] / 8): 5 * int(FN_block[0].shape[1] / 8)]
    FN6 = FN_block[0][:, 5 * int(FN_block[0].shape[1] / 8): 6 * int(FN_block[0].shape[1] / 8)]
    FN7 = FN_block[0][:, 6 * int(FN_block[0].shape[1] / 8): 7 * int(FN_block[0].shape[1] / 8)]
    FN8 = FN_block[0][:, 7 * int(FN_block[0].shape[1] / 8): 8 * int(FN_block[0].shape[1] / 8)]

    # axs[1, 0].imshow(FN1)
    # axs[1, 1].imshow(FN2)
    # axs[1, 2].imshow(FN3)
    # axs[1, 3].imshow(FN4)
    # axs[1, 4].imshow(FN5)
    # axs[1, 5].imshow(FN6)
    # axs[1, 6].imshow(FN7)
    # axs[1, 7].imshow(FN8)

    global FN
    FN = [FN1,FN2,FN3,FN4,FN5,FN6,FN7,FN8]


    LN1 = LN_block[0][:, 0:int(LN_block[0].shape[1] / 8)]
    LN2 = LN_block[0][:, int(LN_block[0].shape[1] / 8):2 * int(LN_block[0].shape[1] / 8)]
    LN3 = LN_block[0][:, 2 * int(LN_block[0].shape[1] / 8): 3 * int(LN_block[0].shape[1] / 8)]
    LN4 = LN_block[0][:, 3 * int(LN_block[0].shape[1] / 8): 4 * int(LN_block[0].shape[1] / 8)]
    LN5 = LN_block[0][:, 4 * int(LN_block[0].shape[1] / 8): 5 * int(LN_block[0].shape[1] / 8)]
    LN6 = LN_block[0][:, 5 * int(LN_block[0].shape[1] / 8): 6 * int(LN_block[0].shape[1] / 8)]
    LN7 = LN_block[0][:, 6 * int(LN_block[0].shape[1] / 8): 7 * int(LN_block[0].shape[1] / 8)]
    LN8 = LN_block[0][:, 7 * int(LN_block[0].shape[1] / 8): 8 * int(LN_block[0].shape[1] / 8)]


    # axs[2, 0].imshow(LN1)
    # axs[2, 1].imshow(LN2)
    # axs[2, 2].imshow(LN3)
    # axs[2, 3].imshow(LN4)
    # axs[2, 4].imshow(LN5)
    # axs[2, 5].imshow(LN6)
    # axs[2, 6].imshow(LN7)
    # axs[2, 7].imshow(LN8)

    global LN
    LN = [LN1,LN2,LN3,LN4,LN5,LN6,LN7,LN8]

    # axs[3,2].imshow(BS_CB[0])
    # axs[3, 1].imshow(MS_CB[0])
    # axs[3, 0].imshow(PHD_CB[0])
    #
    # ax = plt.gca()
    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])
    #
    # plt.show()

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

def n_clf(img):
    #img = load_img(imgname, target_size=(28, 28), grayscale=True)
    #img = cv2.imread(imgname,0)
    img =cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(28,28))
    #plt.imshow(img)
    #plt.show()
    #cv2.imshow("clf",img)
    image = img_to_array(img) / 255.
    # orig_img = image.copy()
    image = np.expand_dims(image, 0)
    predictions = model.predict(image)[0]
    label = np.argmax(predictions)
    #proba = np.max(predictions)
    # print(label)
    if label > 9:
        label =-1;
    return label

def c_clf(img):
    #img = load_img(imgname, target_size=(28, 28), grayscale=True)
    #img = cv2.imread(imgname,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(28,28))
    #plt.imshow(img)
    #plt.show()
    #cv2.imshow("clf",img)
    image = img_to_array(img) / 255.
    # orig_img = image.copy()
    image = np.expand_dims(image, 0)
    predictions = model.predict(image)[0]
    label = np.argmax(predictions)
    #proba = np.max(predictions)
    # print(label)
    if label < 9:
        label =-1
    return label

if __name__ == "__main__":

    addr = "examples/4.jpg"

    run_s = time.time()

    img = cv2.imread(addr)

    dst = transform(img)


    table_detection(dst)

    print("\n\n\n")

    I_D = ""
    for i in ID:
        I_D += fa_ch[n_clf(i)]
    fa_print(I_D[2:])

    F_N = ""
    for l in FN:
        F_N += fa_ch[c_clf(l)]

    F_N = F_N[1:]
    F_N = F_N[::-1]
    fa_print(F_N)

    L_N = ""
    for j in LN:
        L_N += fa_ch[c_clf(j)]
    L_N =L_N[1:]
    L_N = L_N[::-1]
    fa_print(L_N)



    BS_string = u"کارشناسی"
    MS_string = u"کارشناسی ارشد"
    PHD_string = u"دکتری"

    if(checkbox(BS_CB[0])):
        fa_print(BS_string)
    if(checkbox(MS_CB[0])):
        fa_print(MS_string)
    if(checkbox(PHD_CB[0])):
        fa_print(PHD_string)


    run_f=time.time()
    run_t=run_f-run_s
    print("run time :", run_t)




