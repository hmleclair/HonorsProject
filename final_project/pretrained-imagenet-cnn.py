#---------------------------------------
#Hailey LeClair CS4900 Project
#Pretrained Network
#Feeds into other neural networks
#---------------------------------------
#a lot of this code and some comments is taken from http://www.pyimagesearch.com/2016/08/10/imagenet-classification-with-python-and-keras/
from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
#from vgg16 import VGG16
from tensorflow.contrib.keras.python.keras.applications.vgg16 import VGG16
import numpy as np
import argparse
import cv2
import opencv
import os

#----------------------------------------------------
#construct an argument parser and parse the arguments
#----------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

#---------------------------------------------------------
#load the original image via OpenCV so we can classify it
#---------------------------------------------------------
original = cv2.imread(args["image"])

#---------------------------------------------------------------------------------
#Load the input image using the keras helper utility while ensuring that the
#image is resized to 224x224 pixels, the required input dimensions for the network 
#image is then converted to array so it is easy to work with
#----------------------------------------------------------------------------------
print("[loading  image..].")
image = image_utils.load_img(args["image"], target_size=(224, 224))
image = image_utils.img_to_array(image)

#---------------------------------------------------------------------------------------------
#we have to expand the dimensions of the array so we can pass it through the cnn with keras/tf
#we will also preprocess the imae by subtracting the mean RGB pixel intensity from the 
#ImageNet dataset, we also want the dimensions to be (1, 3, 244, 244) to pass through the cnn
#---------------------------------------------------------------------------------------------
original_image_data = image
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

#---------------------------------------------------------------------------------------------
#load the VGG16 pre trained network so it can be used to classify the image as a shoe or a dress
#---------------------------------------------------------------------------------------------
print("[loading convolutional neural network...]")
model = VGG16(weights="imagenet")

#---------------------------------------------------------------------------------------------
#classify the image and output the results!
#---------------------------------------------------------------------------------------------
print("[classifying image...]")
preds = model.predict(image)
(inID, label, probability) = decode_predictions(preds)[0][0]

#---------------------------------------------------------------------------------------------
#display the predictions on the screen
#---------------------------------------------------------------------------------------------
print('[Results...]')
print("\n\tImageNet ID: {},\n\tLabel: {},\n\tprobability: {}".format(inID, label, probability))

#-------------------------------------------------------------------------------------------------------------------------------------------------------
#if: the image contains a dress (according to imagenet) elif: the image contains a shoe, 
#else: neither according to imagenet(which isn't always correct)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
if label == 'overskirt' or  label == 'pedestal' or label == 'gown' or label == 'jumper' or label == 'Cocktail' or label == 'Sari' or label == 'Chemise' or label == 'Sundress' or label == 'Kirtle' or label == 'hoopskirt':
	import cnn_keras_dress_predict
elif label == 'clog' or label == 'Loafer' or label == 'sandal' or label == 'running_shoe' or label == 'pump' or label == 'oxford' or label == 'walking':
	import cnn_keras_shoe_predict	
else:
	print("Sorry, your image could not be classified as either a dress or a shoe... Maybe try another image?")
