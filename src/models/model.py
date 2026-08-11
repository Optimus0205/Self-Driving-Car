import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

x=tf.placeholder(tf.float32, shape=[None, 66, 200, 3])      # None -> Batch size
y_=tf.placeholder(tf.float32, shape=[None, 1])

x_image=x

# First Convolution layer
W_conv1=weight_variable([5, 5, 3, 24])
b_conv1=bias_variable([24])

h_conv1=tf.nn.relu(conv2d(x_image, W_conv1, 2)+b_conv1)

# Second Convolution layer
W_conv2=weight_variable([5, 5, 24, 36])
b_conv2=bias_variable([36])

h_conv2=tf.nn.relu(conv2d(h_conv1, W_conv2, 2)+b_conv2)

# Third Convolution layer
W_conv3=weight_variable([5, 5, 36, 48])
b_conv3=bias_variable([48])

h_conv3=tf.nn.relu(conv2d(h_conv2, W_conv3, 2)+b_conv3)

# Fourth Convolution layer
W_conv4=weight_variable([3, 3, 48, 64])
b_conv4=bias_variable([64])

h_conv4=tf.nn.relu(conv2d(h_conv3, W_conv4, 1)+b_conv4)

# Fifth Convolution layer
W_conv5=weight_variable([3, 3, 64, 64])
b_conv5=bias_variable([64])

h_conv5=tf.nn.relu(conv2d(h_conv4, W_conv5, 1)+b_conv5)

# FCL 1
W_fcl1=weight_variable([1152, 1164])
b_fcl1=bias_variable([1164])

h_conv5_flat=tf.reshape(h_conv5,[-1,1152])
h_fcl1=tf.nn.relu(tf.matmul(h_conv5_flat, W_fcl1)+b_fcl1)

keep_prob=tf.placeholder(tf.float32)
h_fcl1_drop=tf.nn.dropout(h_fcl1,keep_prob)

# FCL 2
W_fcl2=weight_variable([1164, 100])
b_fcl2=bias_variable([100])

h_fcl2=tf.nn.relu(tf.matmul(h_fcl1_drop, W_fcl2)+b_fcl2)

h_fcl2_drop=tf.nn.dropout(h_fcl2,keep_prob)

# FCL 3
W_fcl3=weight_variable([100, 50])
b_fcl3=bias_variable([50])

h_fcl3=tf.nn.relu(tf.matmul(h_fcl2_drop, W_fcl3)+b_fcl3)

h_fcl3_drop=tf.nn.dropout(h_fcl3,keep_prob)

# FCL 4
W_fcl4=weight_variable([50, 10])
b_fcl4=bias_variable([10])

h_fcl4=tf.nn.relu(tf.matmul(h_fcl3_drop, W_fcl4)+b_fcl4)

h_fcl4_drop=tf.nn.dropout(h_fcl4,keep_prob)

# Output
W_fcl5=weight_variable([10, 1])
b_fcl5=bias_variable([1])

y=tf.multiply(tf.atan(tf.matmul(h_fcl4_drop, W_fcl5)+b_fcl5),2)     # Scale the atan output
                                                                       # y -> 2 to get -180,180 rotation of steering
