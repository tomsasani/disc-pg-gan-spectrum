"""
CNN-based discriminator models for pg-gan.
Author: Sara Mathieson, Zhanpeng Wang, Jiaping Wang
Date: 2/4/21
"""

# python imports
import tensorflow as tf

import global_vars

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    DepthwiseConv2D,
    MaxPooling2D,
    Dropout,
    Concatenate,
)
from tensorflow.keras import Model

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class OnePopModel(Model):
    """Single population model - based on defiNETti software.
    
    """

    def __init__(self, pop, saved_model=None): # (self, n_haps: int, n_snps: int, n_channels: int, saved_model = None)
        super(OnePopModel, self).__init__()

        #print (f"DISCRIMINATOR should be expecting {pop} haplotypes")

        input_shape = (pop, global_vars.NUM_SNPS, global_vars.NUM_CHANNELS)

        if saved_model is None:
            self.conv1 = Conv2D(32, (1, 5), activation="relu", data_format="channels_last", input_shape=input_shape)
            self.conv2 = Conv2D(64, (1, 5), activation="relu")
            self.conv1x1_32 = Conv2D(32, (1, 1), activation="relu")
            self.conv1x1_64 = Conv2D(64, (1, 1), activation="relu")
            # self.conv3 = Conv2D(64, (1, 5), activation="relu")
            # pooling applied after each convolution.
            self.pool = MaxPooling2D(pool_size = (1,2), strides = (1,2))

            self.flatten = Flatten()
            self.dropout = Dropout(rate=0.5)

            # change from 128,128 to 32,32,16 (same # params)
            self.fc1 = Dense(128, activation='relu')
            self.fc2 = Dense(128, activation='relu')
            self.dense3 = Dense(1)#2, activation='softmax') # two classes

        else:
            self.conv1 = saved_model.conv1
            self.conv2 = saved_model.conv2

            self.pool = saved_model.pool

            self.flatten = saved_model.flatten
            self.dropout = saved_model.dropout

            self.fc1 = saved_model.fc1
            self.fc2 = saved_model.fc2
            self.dense3 = saved_model.dense3

        self.pop = pop

    def last_hidden_layer(self, x):
        """ Note this should mirror call """
        assert x.shape[2] == global_vars.NUM_SNPS

        # images, dists = x[:, :, :, :-1], x[:, :, :, -1]
        # image_conv, dist_conv = self.conv1(images), self.conv1(dists)
        # image_pool, dist_pool = self.pool(image_conv), self.pool(dists_conv)

        x = self.conv1(x)
        x = self.pool(x)
        # x = self.conv1x1_32(x)
        x = self.conv2(x)
        x = self.pool(x)
        # x = self.conv1x1_64(x)
       

        # note axis is 1 b/c first axis is batch
        # can try max or sum as the permutation-invariant function
        # x = tf.math.reduce_max(x, axis=1)

        # reduce sum will take an array of (50, 200, 36, 6)
        # and sum across haplotypes to produce a tensor of
        # (50, 36, 6).
        x = tf.math.reduce_sum(x, axis=1)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=False)

        x = self.fc2(x)
        x = self.dropout(x, training=False)

        return x

    def call(self, x, training=None):
        """x is the genotype matrix + distances"""
        assert x.shape[2] == global_vars.NUM_SNPS

        x = self.conv1(x)
        x = self.pool(x)
        # x = self.conv1x1_32(x)
        x = self.conv2(x)
        x = self.pool(x)
        # x = self.conv1x1_64(x)

        # note axis is 1 b/c first axis is batch
        # can try max or sum as the permutation-invariant function
        #x = tf.math.reduce_max(x, axis=1)
        x = tf.math.reduce_sum(x, axis=1)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)

        x = self.fc2(x)
        x = self.dropout(x, training=training)

        return self.dense3(x)

    def build_graph(self, gt_shape):
        """This is for testing, based on TF tutorials"""
        gt_shape_nobatch = gt_shape[1:]
        self.build(gt_shape) # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=gt_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)