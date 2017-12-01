#------------------------------------------------------------------------------------------------
# Hailey LeClair CS4900
# Honors Project
# Much of this code influenced by https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
#------------------------------------------------------------------------------------------------
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing import image as image_utils
import argparse
import cv2
import keras.backend as K
import numpy as np
from PIL import Image
import math

#---------------------------------------------------------------------------------------------
# Load the model so that we can make a prediction for the image with a trained network
#---------------------------------------------------------------------------------------------
dress_model = load_model('dress_model.h5')
K.set_image_dim_ordering('th')
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

#-------------------------------------------------------------------------------------
# Load the original image and change it's size and predict whether it is long dress or puffy dress
# Then using euclidean distance, try to match the test image to one of the 5 "store" images in 
# images_shoes/validation/
#---------------------------------------------------------------------------------------------
original = cv2.imread(args["image"])
img = image_utils.load_img(args["image"], target_size=(150, 150))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = dress_model.predict_classes(x)
prob = dress_model.predict_proba(x)
def euclidean_distance(x, y, length):
        distance = 0
        for i in range(length):
                for j in range(len(x)):
                        for k in range(len(x[j])):
                                distance += pow((x[i][j][k] - y[i][j][k]), 2)
        return np.sqrt(distance)

if preds[0][0] == 0:
        print('\tClass: Long Dress')
        matches = []
        distances = []
        closest_match = []
        for i in range(1, 6):
                new_img = image_utils.load_img('images_dress/validation/long_dresses/test_dress_' + `i`+ '.jpg', target_size=(150, 150))
                matches.append(new_img)
		print(img_to_array(new_img))
                distances.append(euclidean_distance(img_to_array(new_img), img_to_array(img), len(img_to_array(new_img))))
        distances = np.array(distances)
        closest_match = distances.argmin()
        matched_image = matches[closest_match]
	matched_image.show()
	img.show()
else:
	print('\tClass: Puffy Dress')
        matches = []
        distances = []
        closest_match = []
        for i in range(6, 11):
                new_img = image_utils.load_img('images_dress/validation/puffy_dresses/test_dress_' + `i`+ '.jpg', target_size=(150, 150))
                matches.append(new_img)
                distances.append(euclidean_distance(img_to_array(new_img), img_to_array(img), len(img_to_array(new_img))))
        distances = np.array(distances)
        closest_match = distances.argmin()
        matched_image = matches[closest_match]
	matched_image.show()
	img.show()
