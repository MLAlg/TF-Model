#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 03:32:10 2019

@author: samaneh
"""

import tensorflow as tf
import numpy as np

#  create 100 data points x, y in NumPy, y = x * 0.1 + 0.3
x_data =  np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# try to find values for W and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# minimize the mean squared errors
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# before starting, initialize the variables.(We will 'run' this first)
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# fit the line
for i in range(201):
    sess.run(train)
    if i % 20 == 0:
        print(i, sess.run(W), sess.run(b))
# learns that best fit is W: [0.1], b: [0.3]
