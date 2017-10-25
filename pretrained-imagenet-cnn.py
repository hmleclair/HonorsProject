#import some packages
#this is from http://www.pyimagesearch.com/2016/08/10/imagenet-classification-with-python-and-keras/
from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from vgg16 import VGG16
import numpy as np
import argparse
import cv2
import opencv

#construct an argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

#load the original image via OpenCV so we can draw on it and display it to the screen
original = cv2.imread(args["image"])

#Load the input image using the keras helper utility while ensuring that the
#image is resized to 224x224 pixels, the required input dimensions for the network 
#-- then convert the PIL image to a numpy array
print("[INFO] loading and preprocessing image...")
image = image_utils.load_img(args["image"], target_size=(224, 224))
image = image_utils.img_to_array(image)

#we have to expand the dimensions of the array so we can pass it through the cnn
#we will also preprocess the imae by subtracting the mean RGB pixel intensity from the 
#ImageNet dataset, we also want the dimensions to be (1, 3, 244, 244) to pass through the cnn
original_image_data = image
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

#load the VGG16 pre trained network
print("[INFO] loading network...")
model = VGG16(weights="imagenet")

#classify the image
print("[INFO] classifying image...")
preds = model.predict(image)
(inID, label, probability) = decode_predictions(preds)[0][0]

#display the predictions on the screen
print("ImageNet ID: {},\nLabel: {},\nprobability: {}".format(inID, label, probability))

#feed the label into another neural network
#Some of the following is from https://www.tensorflow.org/versions/master/tutorials/layers

