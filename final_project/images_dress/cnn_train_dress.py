
rt imFunctions as imf
import tensorflow as tf
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
import urllib
from PIL import Image
import os
images_data_train = []
images_data_test = []
image_array_train = []
image_array_test = []
num_channels = 3
img_width = 200
img_height = 300
image_count = 0
image_names_test = []
image_names_train = []
#PUT IMAGE NAMES IN HERE AND CHANGE THE FILES GOING IN THIS ONE IS FOR DRESSES
for i  in range(0, 30):
        image_count += 1
	#THIS WILL BE USEFUL IF I CAN ACTUALLY GET IMAGES OFF IMAGENET TO WORK!!!!!
        #urllib.urlretrieve(url, os.path.join(os.getcwd(), './dress_images/long_dress_' + str(image_count) + '.jpg'))
        image = Image.open('./dress_images/dress_' +str(image_count) + '.jpg')
	if image_count < 16:
		image_names_train.append('long_dress')
	else
		image_name_train.append('puffy_dress')
        img_resized = image.resize((img_height, img_width), Image.ANTIALIAS)
        #image = Image.open( urllib.urlretrieve(url))
        image_array_train_temp  = np.asarray(img_resized)
        if image_array_train_temp.shape == (img_width, img_height):
                temp_image = np.zeros((img_width, img_height, num_channels))
                for i in range(num_channels):
                        for x in range(img_width):
                                for y in range(img_height):
                                        temp_image[x,y,i] = image_array_train_temp[x,y]
                image_array_train_temp = temp_image
        print(len(image_array_train_temp))
        print(image_array_train_temp.shape)
        images_data_train.append(np.array(image_array_train_temp))

        #image_final = Image.fromarray(image_array)
        #plt.imshow(image_final)
        #plt.show()

x_train = images_data_train
for i in range(0, 10):
        image_count += 1
        #urllib.urlretrieve(url, os.path.join(os.getcwd(), 'test' +str(image_count) + '.jpg'))
        image = Image.open('./dress_images/puffy_dress_' + str(image_count) +'.jpg')
	if image_count < 6:
                image_names_train.append('long_dress')
        else
                image_name_train.append('puffy_dress')
        img_resized = image.resize((img_height, img_width), Image.ANTIALIAS)
        #image = Image.open(urllib.urlretrieve(url))
        image_array_test_temp = np.asarray(img_resized)
        print(image_array_test_temp.shape)
        if image_array_test_temp.shape == (img_width, img_height):
                temp_image = np.zeros((img_width, img_height, num_channels))
                for i in range(num_channels):
                        for x in range(img_width):
                                 for y in range(img_height):
                                         temp_image[x,y,i] = image_array_test_temp[x,y]
                image_array_test_temp = temp_image
        images_data_test.append(np.array(image_array_test_temp))
        #image_final = Image.fromarray(image_array)
        #plt.imshow(image_final)
        #plt.show()

#data?
train_x = np.array(images_data_train)
#train_y = [[0,1,0,1,0],[1,0,1,0,1]]
train_y = [[0,0,0,0,0],[1,1,1,1,1]]
#test_x = np.array([[images_data_test[0], images_data_test[6], images_data_test[7], images_data_test[2], images_data_test[5]],[images_data_test[4], images_data_test[8], images_data_test[1], images_data_test[9], images_data_test[3]]])
test_x = np.array(images_data_test)
test_y = []
class_batch = [0, 1]

x_test = images_data_test
classes = ['long_dress', 'puffy_dress']
num_classes = len(classes)
validation_size = 0.2
batch_size = 2
current_batch_number = 0
def next_batch_train():
	global current_batch_number
	current_batch_x = []
	current_batch_y = []
	current_batch_x.append(np.array(train_x[current_batch_number]))
	current_batch_y.append(np.array(train_y[current_batch_number]))
	current_batch_x.append(np.array(train_x[current_batch_number+1]))
        current_batch_y.append(np.array(train_y[current_batch_number+1]))
	current_batch_number += 2
	return current_batch_x, np.array(current_batch_y), image_names_train,  class_batch
current_batch_number_test = 0
def next_batch_test():
	global current_batch_number_test
        current_batch_x_test = []
        current_batch_y_test = []
        current_batch_x_test.append(np.array(test_x[current_batch_number_test]))
        current_batch_y_test.append(np.array(test_y[current_batch_number_test]))
	current_batch_x_test.append(np.array(test_x[current_batch_number_test+1]))
        current_batch_y_test.append(np.array(test_y[current_batch_number_test+1]))
	current_batch_number_test += 2
        return current_batch_x_test, np.array(current_batch_y_test), image_names_test, class_batch

##Network graph params
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
    
fc_layer_size = 128

session = tf.Session()

#WEIGHTS AND BIAS
def create_weights(shape):
	#	global weights
	weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
	#tf.variables_initializer([weights], name='weights')
	#tf.initialize_variables(weights)
	return weights
def create_biases(size):
	#global biases
	biases = tf.Variable(tf.constant(0.05, shape=[size]))
	#tf.variables_initializer([biases], name='biases')
	#tf.initialize_variables(biases)
	return biases

#NETWORK LAYERS

#an example?
#tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
	weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
	biases = create_biases(num_filters)
	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
	layer += biases
	layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	layer = tf.nn.relu(layer)
	return layer

def create_flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer = tf.reshape(layer, [-1, num_features])

	return layer

def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
	#trainable weights and biases
	weights = create_weights(shape=[num_inputs, num_outputs])
	biases = create_biases(num_outputs)
	layer = tf.matmul(input, weights) + biases
	if use_relu:
		layer = tf.nn.relu(layer)
	return layer

x = tf.placeholder(tf.float32, shape=[None, img_width, img_height, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1 = create_convolutional_layer(input=x, num_input_channels=num_channels, conv_filter_size=filter_size_conv1, num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1, num_input_channels=num_filters_conv1, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)

#leaving this out for now but shoudl be layer_conv3 feeds into layer_flat

layer_flat = create_flatten_layer(layer_conv2)

layer_fc1 = create_fc_layer(input=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(), num_outputs=fc_layer_size, use_relu=True)
#figure out why there's the number of layers there are in this case
layer_fc2 = create_fc_layer(input=layer_fc1, num_inputs=fc_layer_size, num_outputs=num_classes, use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#val_acc = session.run(accuracy, feed_dict=feed_dict_test)
#acc = session.run(accuracy, feed_dict=feed_dict_train)

saver = tf.train.Saver()
def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0
#FIX THIS CAUSE THERE'S SOMETHING WRONG WITH IT!!!!!!!!!!!!!!
def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   #total_iterations + num_iteration):
    	for i in range(0, 5):

        	x_batch, y_true_batch, _, cls_batch = next_batch_train()
        	x_test_batch, y_test_batch, _, valid_cls_batch = next_batch_test()
	
		print(y_test_batch)
        
       	 	feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        	feed_dict_test = {x: x_test_batch,
                              y_true: y_test_batch}

		init_op = tf.global_variables_initializer()
		#tf.variables_initializer([biases, weights], name='weights_and_biases')
		#tf.initialize_local_variables()
		#	tf.variables_initializer(local_variables())
        	session.run(init_op, feed_dict=feed_dict_tr)

        	if i % int(len(train_x)/batch_size) == 0: 
        		val_loss = session.run(cost, feed_dict=feed_dict_test)
        		epoch = int(i / int(len(train_x)/batch_size))    
            
        		show_progress(epoch, feed_dict_tr, feed_dict_test, val_loss)
 
        		saver.save(session, 'pen-shoe-model') 


    total_iterations += num_iteration

train(num_iteration=1)


