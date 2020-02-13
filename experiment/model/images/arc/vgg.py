# -*- coding: utf-8 -*-
# @Time    : 2018/12/7  22:02
# @Author  : Dyn

# -*- coding: utf-8 -*-
# Time: 2018/11/6 18:01
# Author: Dyn
import tensorflow as tf
import numpy as np


class VGG16:
    def __init__(self, weights_path='./vgg16.npy'):
        self.weights = np.load(weights_path, encoding='latin1').item()
        self.activation_fn = tf.nn.relu
        self.conv_padding = 'SAME'
        self.pool_padding = 'SAME'
        self.use_bias = True

    def inference(self, images):
        with tf.variable_scope('nets'):
            self.conv1_1 = self.conv(images, 'conv1_1', 64, pre_trained=True, trainable=False)
            self.conv1_2 = self.conv(self.conv1_1, 'conv1_2', 64, pre_trained=True, trainable=False)

            # Max-pooling is performed over a 2 × 2 pixel window, with stride 2.
            self.max_pool1 = tf.layers.max_pooling2d(self.conv1_2, 2, 2, padding=self.pool_padding)

            self.conv2_1 = self.conv(self.max_pool1, 'conv2_1', 128, pre_trained=True, trainable=False)
            self.conv2_2 = self.conv(self.conv2_1, 'conv2_2', 128, pre_trained=True, trainable=False)

            self.max_pool2 = tf.layers.max_pooling2d(self.conv2_2, 2, 2, padding=self.pool_padding)

            self.conv3_1 = self.conv(self.max_pool2, 'conv3_1', 256, pre_trained=True, trainable=False)
            self.conv3_2 = self.conv(self.conv3_1, 'conv3_2', 256, pre_trained=True, trainable=False)
            self.conv3_3 = self.conv(self.conv3_2, 'conv3_3', 256, pre_trained=True, trainable=False)

            self.max_pool3 = tf.layers.max_pooling2d(self.conv3_3, 2, 2, padding=self.pool_padding)

            self.conv4_1 = self.conv(self.max_pool3, 'conv4_1', 512, pre_trained=True, trainable=False)
            self.conv4_2 = self.conv(self.conv4_1, 'conv4_2', 512, pre_trained=True, trainable=False)
            self.conv4_3 = self.conv(self.conv4_2, 'conv4_3', 512, pre_trained=True, trainable=False)

            self.max_pool4 = tf.layers.max_pooling2d(self.conv4_3, 2, 2, padding=self.pool_padding)

            self.conv5_1 = self.conv(self.max_pool4, 'conv5_1', 512, pre_trained=True, trainable=True)
            self.conv5_2 = self.conv(self.conv5_1, 'conv5_2', 512, pre_trained=True, trainable=True)
            self.conv5_3 = self.conv(self.conv5_2, 'conv5_3', 512, pre_trained=True, trainable=True)

            self.max_pool5 = tf.layers.max_pooling2d(self.conv5_3, 2, 2, padding=self.pool_padding)

            self.fc7 = tf.reduce_mean(self.max_pool5, axis=[1, 2])
            self.flatten = tf.layers.flatten(self.fc7)

            # self.flatten = tf.layers.flatten(self.max_pool5)
            #
            # self.fc6 = self.fully_connected(self.flatten, 'fc6', 4096, pre_trained=True, trainable=False)
            #
            # self.fc7 = self.fully_connected(self.fc6, 'fc7', 4096, pre_trained=True, trainable=True)

            self.predictions = self.softmax(self.fc7, 'prediction', tf.flags.FLAGS.num_classes)

            return self.predictions

    def get_features(self):
        return self.fc7

    def conv2d(self, layer, name, n_filters, trainable, k_size=3, stride=1):
        return tf.layers.conv2d(layer, n_filters, kernel_size=(k_size, k_size),
                                strides=stride,
                                activation=self.activation_fn,
                                padding=self.conv_padding, name=name, trainable=trainable,
                                kernel_initializer=tf.constant_initializer(self.weights[name][0], dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(self.weights[name][1], dtype=tf.float32),
                                use_bias=self.use_bias)

    def fc(self, layer, name, size, trainable):
        return tf.layers.dense(layer, size, activation=self.activation_fn,
                               name=name, trainable=trainable,
                               kernel_initializer=tf.constant_initializer(self.weights[name][0], dtype=tf.float32),
                               bias_initializer=tf.constant_initializer(self.weights[name][1], dtype=tf.float32),
                               use_bias=self.use_bias)

    def conv(self, layer, name, depth, k_size=3, stride=1,
             pre_trained=False, trainable=True, l2_loss=False):
        filter = self.get_weights(name, [k_size, k_size, layer.shape[-1], depth],
                                  pre_trained, trainable, l2_loss)
        conv = tf.nn.conv2d(layer, filter=filter, strides=[1, stride, stride, stride], padding='SAME')
        biases = self.get_biases(name, [depth], pre_trained, trainable)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name)
        return conv

    def fully_connected(self, layer, name, units, pre_trained=False, trainable=True, l2_loss=False):
        weights = self.get_weights(name, [layer.shape[-1], units], pre_trained, trainable, l2_loss)
        biases = self.get_biases(name, [units], pre_trained, trainable)
        fc_layer = tf.nn.relu(tf.matmul(layer, weights) + biases, name=name)
        return fc_layer

    def softmax(self, layer, name, num_classes):
        weights_initializer = tf.truncated_normal_initializer(stddev=1.0 / int(layer.shape[-1]),
                                                              dtype=tf.float32)
        weights = tf.get_variable(name+'_weights', [layer.shape[-1], num_classes],
                                  initializer=weights_initializer)
        biases = self.get_biases(name, [num_classes], False, True)
        softmax_linear = tf.add(tf.matmul(layer, weights), biases, name='pre' + name)
        return tf.nn.softmax(softmax_linear, name=name)

    def get_weights(self, name, shape, pre_trained=False, trainable=True, l2_loss=False):
        if pre_trained:
            # using pretrained variables
            initializer = tf.constant_initializer(self.weights[name][0], dtype=tf.float32)
        else:
            # using random initialized variables
            initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        var = tf.get_variable(name + '_weights', shape, initializer=initializer, trainable=trainable)
        # add l2_loss to collections
        if l2_loss and trainable:
            weight_decay = tf.nn.l2_loss(var, name=name + '_weight_loss') * tf.flags.FLAGS.l2_rate
            tf.add_to_collection('losses', weight_decay)
        return var

    def get_biases(self, name, shape, pre_trained=False, trainable=True):
        if pre_trained:
            # using pretrained variables
            initializer = tf.constant_initializer(self.weights[name][1], dtype=tf.float32)
        else:
            # using random initialized variables
            initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        var = tf.get_variable(name + '_biases', shape, initializer=initializer, trainable=trainable)
        # add l2_loss to collecetions
        return var


if __name__ == '__main__':
    pass
