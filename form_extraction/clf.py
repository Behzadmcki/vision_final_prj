import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array

model = keras.models.load_model('model.h5')

def clf(imgname):
    #img = load_img(imgname, target_size=(28, 28), grayscale=True)
    img = cv2.imread(imgname,0)
    img = cv2.resize(img,(28,28))
    plt.imshow(img)
    plt.show()
    #cv2.imshow("clf",img)
    image = img_to_array(img) / 255.
    # orig_img = image.copy()
    image = np.expand_dims(image, 0)
    predictions = model.predict(image)[0]
    label = np.argmax(predictions)
    #proba = np.max(predictions)
    return label


print(clf("ID8.png"))
