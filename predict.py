import os
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
import numpy as np
model = load_model('handwritten.model')
model.summary()

image_number = 1
while os.path.isfile(f"test/digit{image_number}.png"):
    try:
        img= cv2.imread(f"test/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        img = np.array(img, dtype="float32")
        img/=255.0
        print(img)
        prediction = model.predict(img)
        print (prediction)
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("error")
    finally:
        image_number+=1
