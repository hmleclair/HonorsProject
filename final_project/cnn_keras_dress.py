#------------------------------------------------------------------------------------------------
# Hailey LeClair CS4900
# Honors Project
#------------------------------------------------------------------------------------------------
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

#------------------------------------------------------------------------------------------------
# Much of this training code inspired/taken from:
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#------------------------------------------------------------------------------------------------
datagen = ImageDataGenerator(
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')

#------------------------------------------------------------------------------------------------
# Did not use this in training, but could be very useful if space is available to store the iamges 
# this generates batch of randomly transformed images, they would just have to be saved to the proper 
# directory, an example of one image generating transformed images runs below if uncommented
#------------------------------------------------------------------------------------------------
#img = load_img('images_dress/train/long_dresses/dress_1.jpg')
#x = img_to_array(img)
#x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

#i = 0
#for batch in datagen.flow(x, batch_size=1, save_to_dir='dresses_test', save_prefix='dress', save_format="jpeg"):
#	i += 1
#	if i > 20:
#		break

#------------------------------------------------------------------------------------------------
# Neural Network layers below, simple and straightforward network
#------------------------------------------------------------------------------------------------
K.set_image_dim_ordering('th')

model = Sequential()

#------------------------------------------------------------------------------------------------
# Image sizes and stuff and filter size make sure to change images to 150 x 150 or adjust below accordingly)
# First Convolutional layer using 32 3x3 filters, using relu activation, and max pooling with 2x2 kernels
#------------------------------------------------------------------------------------------------
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150), data_format="channels_first"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#------------------------------------------------------------------------------------------------
# Second Convolutional layer using 32 3x3 filters, using relu activation, and max pooling with 2x2 kernels
#------------------------------------------------------------------------------------------------
model.add(Conv2D(32, (3, 3), data_format="channels_last"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#------------------------------------------------------------------------------------------------
# Third Convolutional layer using 64 3x3 filters, using relu activation, and max pooling with 2x2 kernels
#------------------------------------------------------------------------------------------------
model.add(Conv2D(64, (3, 3), data_format="channels_last"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
#------------------------------------------------------------------------------------------------ 
# Two flattened fully connected layers at the end of the network using the sigmoid function as activation
#------------------------------------------------------------------------------------------------
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#------------------------------------------------------------------------------------------------ 
# Compile model for training and to use accuracy to adjust for error
#------------------------------------------------------------------------------------------------
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


#------------------------------------------------------------------------------------------------ 
# Compile model for training and to use accuracy to adjust for error
#------------------------------------------------------------------------------------------------
batch_size = 6 

#------------------------------------------------------------------------------------------------
# Generates data in specific classes from training and testing image directories for dress images 
#------------------------------------------------------------------------------------------------

train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
	'images_dress/train/',
	target_size=(150, 150),
	batch_size=batch_size,
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'images_dress/validation/',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

#------------------------------------------------------------------------------------------------
# Trains and saves model 
#------------------------------------------------------------------------------------------------
model.fit_generator(train_generator, steps_per_epoch=30, epochs=10, validation_data=validation_generator, validation_steps=5)
model.save('dress_model.h5') 

