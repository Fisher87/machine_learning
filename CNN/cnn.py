#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# parameters
learning_rate = 0.0001
epochs = 10
batch_size = 50

# declare the training data placeholder
# 28*28 -> 784
x = tf.placeholder(tf.float32, [None, 784])
# dynamically reshape the input
x_shaped = tf.reshape(x, [-1, 28, 28, 1])

# declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# define the convolution layers
def create_new_conv_layer(input_data, num_input_channels, 
                          num_filters, filter_shape, pool_shape, name):
    '''
    conv_filt_shape: hold the shape of the weights that determine the behaviour of the
                     convolutional filter. The format that the `conv2d()` function re-
                     ceives for the filter is:
                        filter:[filter_height, filter_width, in_channels, out_channels]

    out_layer: setup the convolutional filter operation. The parameter `strides` that 
               is required in `conv2d`. In this example strides=[1,1,1,1], this means to
               move in steps of 1 in both the x and y directions(or width and height di-
               rections). This information is conveyed in the strides[1] and strides[2]
               values-both equal to 1 in this case. strides[0], strides[3] are always e-
               qual to 1, if they were not, we would be moving the filter between train-
               ing samples or between channels, which we don't want to do.

    max_pool: `max_pool()` function takes a tensor as its first input over which to per-
               from the pooling. The next two arguments `ksize` and `strides` define the
               operation of the pooling. Ignoring the first and last values of these ve-
               ctors(which will always be set to 1, i.e. ksize[0]=ksize[3]=1, strides[0]
               =strides[3]=1), the middle values of `ksize`(pool_shape[0], pool_shape[1])
               define the shape of the max pooling window in the x and y directions.
    '''


    #setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, 
                       num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.random_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    # initialise bias
    bias = tf.Variable(tf.random_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, strides=[1,1,1,1], padding='SAME')

    # add the bias
    out_layer = out_layer + bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')

    return out_layer

# create some convolutional layers
# create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name)
layer1 = create_new_conv_layer(x_shaped, 1, 32, [5,5], [2,2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5,5], [2,2], name='layer2')

# fully connected layers
flattened = tf.reshape(layer2, [-1, 7*7*64])

# setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.random_normal([7*7*64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.random_normal([1000], stddev=0.01), name='bd1')

dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

# another layer with softmax activations
wd2 = tf.Variable(tf.random_normal([1000, 10], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.random_normal([10], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)

# cross-entropy cost function
cross_entropy = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, 
                                                                labels=y))

# add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy accessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initial operation
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # init variables
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels)/batch_size)

    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x:batch_x, y:batch_y})

            avg_cost += c / total_batch
        test_acc = sess.run(accuracy, 
                            feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), 
              "test accuracy: {:.3f}".format(test_acc))

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
