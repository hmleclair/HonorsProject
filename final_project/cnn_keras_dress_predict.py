#much of this code influenced by https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
import keras 
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
import cv2
import keras.backend as K
import numpy as np

test_model = load_model('dress_model.h5')
K.set_image_dim_ordering('th')
img = load_img('images_dress/train/puffy_dresses/dress_25.jpg', False, target_size=(150, 150))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = test_model.predict_classes(x)
prob = test_model.predict_proba(x)
if preds[0][0] == 0:
        print('Long Slim Dress')
else:
	print('Puffy Dress')




