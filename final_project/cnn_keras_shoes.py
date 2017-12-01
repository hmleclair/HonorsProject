
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
#This will be fixed by adding my own images? taken from other neural network
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
datagen = ImageDataGenerator(
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')

img = load_img('images_shoes/train/sneakers/shoe_1.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

#this generates batch of randomly transformed images, pretty cool bro.
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='dress', save_format="jpeg"):
	i += 1
	if i > 20:
		break

K.set_image_dim_ordering('th')

model = Sequential()
#image sizes and stuff and filter size make sure to change images to 150 x 150 or adjust below accordingly)
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150), data_format="channels_first"))
#model.add(ZeroPadding2D((1, 1), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), data_format="channels_last"))
#model.add(ZeroPadding2D((1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), data_format="channels_last"))
#model.add(ZeroPadding2D((1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#convolutions are applied and feature maps created
 
#two flattened fully connected layers on the end

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#Scrap the first part maybe?

batch_size = 6 
train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
	'images_shoes/train/',
	target_size=(150, 150),
	batch_size=batch_size,
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'images_shoes/validation/',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(train_generator, steps_per_epoch=30, epochs=10, validation_data=validation_generator, validation_steps=5)
model.save('shoe_model.h5') 


