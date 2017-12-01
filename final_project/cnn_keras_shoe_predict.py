#much of this code influenced by https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing import image as image_utils
import argparse

import cv2
import keras.backend as K
import numpy as np

test_model = load_model('shoe_model.h5')
K.set_image_dim_ordering('th')
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

#load the original image and change it's size and predict whether it is sneaker or heel
original = cv2.imread(args["image"])
img = image_utils.load_img(args["image"], target_size=(150, 150))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = test_model.predict_classes(x)
prob = test_model.predict_proba(x)
if preds[0][0] == 0:
        print('High Heels')
else:
        print('Sneakers')

