#!/usr/bin/env python3

import tensorflow as tf
import os
import os.path
import numpy as np
from lyrics import lyrics as lyr
import matplotlib.pyplot as plt



class areEqual():

    def __call__(self, x, y):

        return tf.cast(tf.equal(x,y), tf.float32)



lyr.Domain(label="Image", data=tf.zeros([48, 48]))


areEqual = areEqual()


lyr.Relation(label="equals", domains=("Image", "Image"), function = areEqual)


lyr.Constraint("forall x: equals(x, x)", 1.)



loss =  lyr.current_world.loss()
train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)