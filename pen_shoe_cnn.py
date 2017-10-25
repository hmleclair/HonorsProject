#//www.oreilly.com/ideas/visualizing-convolutional-neural-networks?imm_mid=0f647c&cmp=em-prog-na-na-newsltr_20170916
#import imFunctions as imf
import tensorflow as tf
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
import urllib
from PIL import Image
import os
#get images from urls

train_image_urls = ["http://farm1.static.flickr.com/42/93179130_fe3409de74.jpg","http://farm1.static.flickr.com/185/440432708_132ae73a86.jpg","http://farm4.static.flickr.com/3369/3176181497_4ae4c90200.jpg","http://farm3.static.flickr.com/2417/2270351273_20128c1728.jpg","http://farm3.static.flickr.com/2179/2321913014_f643292028.jpg","http://farm4.static.flickr.com/3262/2819714815_6a306893a4.jpg","http://farm2.static.flickr.com/1158/1126787001_373ebc5655.jpg","http://farm1.static.flickr.com/215/517468851_78192faa9b.jpg","http://farm4.static.flickr.com/3597/3390238589_310d6251c0.jpg","http://farm4.static.flickr.com/3258/3106943798_bb5b637cd3.jpg"]
test_image_urls = ["http://farm1.static.flickr.com/42/93179130_fe3409de74.jpg","http://farm1.static.flickr.com/185/440432708_132ae73a86.jpg","http://farm4.static.flickr.com/3597/3390238589_310d6251c0.jpg","http://farm4.static.flickr.com/3258/3106943798_bb5b637cd3.jpg"]
image_count = 0 
images_data_train = []
images_data_test = []
image_array_train = []
image_array_test = []
num_channels = 3
img_width = 150
img_height = 100
filter_hor = np.array([np.array([0,0,0,0,0]),np.array([0,0,0,0,0]),np.array([1,1,1,1,1]),np.array([0,0,0,0,0]), np.array([0,0,0,0,0])])
#filter_hor_2 = np.array([np.array([0,0,0,0,0]),np.array([0,0,0,0,0]),np.array([1,1,1,1,1]),np.array([0,0,0,0,0]), np.array([0,0,0,0,0])])
#vertical filter
filter_ver = np.array([np.array([0,0,1,0,0]),np.array([0,0,1,0,0]),np.array([0,0,1,0,0]),np.array([0,0,1,0,0]), np.array([0,0,1,0,0])])
#diagonal X filter
filter_diagonal = np.array([np.array([1,0,0,0,1]),np.array([0,1,0,1,0]),np.array([0,0,1,0,0]),np.array([0,1,0,1,0]), np.array([1,0,0,0,1])])
filters = np.array([filter_hor, filter_ver, filter_diagonal])
for i in range(0, 10):
	image_count += 1
       	#urllib.urlretrieve(url, os.path.join(os.getcwd(), 'train' + str(image_count) + '.jpg'))
       	image = Image.open('./train' +str(image_count) + '.jpg')
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
	print(len(images_data_train))

for i in range(0, 10):
        image_count += 1
        #urllib.urlretrieve(url, os.path.join(os.getcwd(), 'test' +str(image_count) + '.jpg'))
        image = Image.open('./test' + str(image_count) +'.jpg')
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
	
#write buildDataset function
def build_dataset():		
	#train_x = np.array([[images_data_train[0], images_data_train[1], images_data_train[2], images_data_train[3], images_data_train[4]],[images_data_train[5], images_data_train[6], images_data_train[7], images_data_train[8], images_data_train[9]]])
	train_x = np.array(images_data_train)
	#train_y = [[1, 1, 1, 1, 1],[ 0, 0, 0, 0, 0]]
	train_y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]	
	#test_x = np.array([[images_data_test[0], images_data_test[1], images_data_test[2], images_data_test[3], images_data_test[4]],[images_data_test[5], images_data_test[6], images_data_test[7], images_data_test[8], images_data_test[9]]])
	test_x = np.array(images_data_test)
	test_y = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0] 
	classes = [[1, 1, 0, 0, 1], [1, 1, 0, 0 ,0]]
	possible_classes = [0,1]
	class_labels = ['pen', 'shoe']
	return train_x, train_y, test_x, test_y, classes, possible_classes, class_labels
 
train_x, train_y, test_x, test_y, classes, possible_classes, class_labels = build_dataset()
#gray = np.mean(image, -1)
#X = tf.placeholder(tf.float32, None)#size?
#conv = tf.nn.conv2d(X, filters, [1,1,1,1], padding="SAME")
#test = tf.Session()
#test.run(tf.global_variables_initializer())
#filteredImage = test.run(conv, feed_dict={X: gray.reshape(1,224,224,1)})

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[10, img_width, img_height, 3])
Y_ = tf.placeholder(tf.float32, [None, 2])
print(Y_)
keepRate1 = tf.placeholder(tf.float32)
keepRate2 = tf.placeholder(tf.float32)
#convolution first layer
with tf.name_scope('conv1_1'):
        filter1_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32, stddev=1e-1), name='weights1_1')
        stride = [1, 1, 1, 1]
        conv = tf.nn.conv2d(X, filter1_1, stride, padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),trainable=True, name='biases1_1')
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out)

with tf.name_scope('conv1_2'):
	filter1_2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], dtype=tf.float32, stddev=1e-1), name='weights1_2')
        conv = tf.nn.conv2d(conv1_1, filter1_2, [1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),trainable=True, name='biases1_2')
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out)

#pooling first layer
with tf.name_scope('pool1'):
	pool1_1 = tf.nn.conv2d(conv1_1, filter1_2, [1,1,1,1], padding='SAME', name='pool1_1')
	pool1_1_drop = tf.nn.dropout(pool1_1, keepRate1)
	
with tf.name_scope('conv2_1'):
	filter2_1 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32, stddev=1e-1), name='weights2_1')
	conv = tf.nn.conv2d(pool1_1_drop, filter2_1, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases2_1')
	out = tf.nn.bias_add(conv, biases)
	conv2_1 = tf.nn.relu(out)

with tf.name_scope('conv2_1'):
	filter2_1 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32, stddev=1e-1), name='weights2_1')
	conv = tf.nn.conv2d(pool1_1_drop, filter2_1, [1, 1, 1, 1], padding='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
	trainable=True, name='biases2_1')
	out = tf.nn.bias_add(conv, biases)
	conv2_1 = tf.nn.relu(out)

#convolution 2 - 2
with tf.name_scope('conv2_2'):
	filter2_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights2_2')
	conv = tf.nn.conv2d(conv2_1, filter2_2, [1, 1, 1, 1], padding ='SAME')
	biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases2_2')
	out = tf.nn.bias_add(conv, biases)
	conv2_2 = tf.nn.relu(out)

#pool

with tf.name_scope('pool2'):
	pool2_1 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2_1')

	pool2_1_drop = tf.nn.dropout(pool2_1, keepRate1)

#Fully Connected Layer
with tf.name_scope('fc1') as scope:
	shape = int(np.prod(pool2_1_drop.get_shape()[1:]))
	fc1w = tf.Variable(tf.truncated_normal([shape, 512], dtype=tf.float32, stddev=1e-1), name='weights3_1')
	fc1b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32), trainable=True, name='biases3_1')
	pool2_flat = tf.reshape(pool2_1_drop, [-1, shape])
	out = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
	fc1 = tf.nn.relu(out)
	fc1_drop = tf.nn.dropout(fc1, keepRate2)

with tf.name_scope('softmax') as scope:
	fc2w = tf.Variable(tf.truncated_normal([shape, len(classes[0])], dtype=tf.float32, stddev=1e-1), name='weights3_2')
	print(fc2w)
	fc2b = tf.Variable(tf.constant(1.0, shape=[len(classes[0])], dtype=tf.float32), trainable=True, name='biases3_2')
	print("fc1_drop shape \n", fc1_drop.shape)
	print("fc2w shape \n", fc2w.shape)
	print("fc2b shape \n", fc2b.shape)
	Ylogits = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
	Y = tf.nn.softmax(Ylogits)


#print(Ylogits)
print(classes)
#create loss & optimal
numEpochs = 2
batchSize = 5
alpha = 1e-5
with tf.name_scope('cross_entropy'):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
	loss = tf.reduce_mean(cross_entropy)
	
with tf.name_scope('accuracy'):
	correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
	accuracy = tf.reduce_mean(cross_entropy)

with tf.name_scope('train'):
	train_step = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)
	
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

writer_1 = tf.summary.FileWriter("/cnn/train")
writer_2 = tf.summary.FileWriter("/cnn/test")
writer_1.add_graph(sess.graph)
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
tf.summary.histogram("weights1_1", filter1_1)
write_op = tf.summary.merge_all()

#Train the model
steps = int(len(train_x[0])/batchSize)
print('steps' + str(steps) + 'training set' + str(len(train_x[0])))
for i in range(numEpochs):
	accHist = []
	for ii in range(steps):
	#calculate current steps
		step = i * steps + ii
		#Feed forward batch of train images into graph and log accuracy
		acc = sess.run([accuracy], feed_dict={X: np.array(train_x)[(ii*batchSize):((ii+1)*batchSize),:,:,:], Y_: train_y[(ii*batchSize):((ii+1)*batchSize)], keepRate1:1, keepRate2: 1})
		accHist.append(acc)
		print('acc' + str(acc))
		if step % 5 == 0:
		#Get Train Summary for one batch and add summary to TensorBoard#Not this part
			summary = sess.run(write_op, feed_dict={X: np.array(train_x)[(ii*batchSize):((ii+1)*batchSize),:,:,:], Y_: train_y[(ii*batchSize):((ii+1)*batchSize)], keepRate1: 1, keepRate2: 1})			
		writer_1.add_summary(summary, step)
		writer_1.flush()
		print(Y_)
		#stuff missing from website her but may not need
		#back propogate using adam optimizer to update weights and biases.
		sess.run(train_step, feed_dict={X:np.array(train_x)[(ii*batchSize):((ii+1)*batchSize),:,:,:], Y_: train_y[(ii*batchSize):((ii+1)*batchSize)], keepRate1: 0.2, keepRate2: 0.5})
		print('Epoch number {} Training Accuracy: {}'.format(i+1, np.mean(accHist)))

		#feed forward all test images into graph and log accuracy
		Y_t = tf.placeholder(tf.float32, [len(test_y)])
		X_t = tf.placeholder(tf.float32, shape=[None, img_width, img_height, 3])
		for iii in range(int(len(test_x[0])/batchSize)):
			acc = sess.run(accuracy, feed_dict={X: np.array(test_x)[(iii*batchSize):((iii+1)*batchSize),:,:,:], Y: test_y[(iii*batchSize):((iii+1)*batchSize)], keepRate1: 1, keepRate2: 1})
			accHist2.append(acc)
		print("Test set Accuracy: {}".format(np.mean(accHist2)))
	
