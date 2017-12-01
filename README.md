# Hailey LeClair's Honors Project 
## For CS4900
## Supervised By Dr. Andrew Godbout
  
# Description
This Project uses Python, Keras, and Tensorflow to build and classify images of dresses and shoes, and also
makes an attempt at image matching. 
Using a neural network with a pretrained set of weights (VGG16 -  good explanation here http://www.cs.toronto.edu/~frossard/post/vgg16/)) to classify an image in a category of either a dress or shoe, and once it knows that, classify that dress as either a 'long dress' or a 'puffy dress' or that shoe as a 'sneaker' or a 'high heel'. Once it knows this is will try to match the image with one of the 5 test for that category (just to try image matching), and displays the image and it's match.

# How it Works (Visual)

Look at image 'Flow_Chart.png'

# How to Run the Program
To classify and attempt to match an image, navigate to the 'final_project' directory and run the following commands with the path to the image, an working example will be shown below


$ python pretrained-imagenet-cnn.py -i path_to_image

ex:
$ python pretrained-imagenet-cnn.py -i images_shoes/validation/heels/shoe_test_2.jpg 

