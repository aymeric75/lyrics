#!/usr/bin/env python3


import tensorflow as tf


import sys
sys.path.append(r"../latplan")
import latplan


import os
import os.path
import glob
import hashlib
import numpy as np
import latplan.model
import latplan.util.stacktrace
from latplan.util.tuning import simple_genetic_search, parameters, nn_task, reproduce, load_history, loadsNetWithWeights, loadsNetWithWeightsGOOD
from latplan.util        import curry
from latplan.main.common import train_val_test_split
from latplan.main.puzzle import load_puzzle

from latplan.util.np_distances import mse

from lyrics import lyrics as lyr

import matplotlib.pyplot as plt

from train_common import parameters



def _add_misc_info(config):
    for key in ["HOSTNAME", "LSB_JOBID"]:
        try:
            config[key] = os.environ[key]
        except KeyError:
            pass





parameters['batch_size'] = 400
parameters['N'] = 300
parameters['beta_d'] = 10
parameters['beta_z'] = 10
parameters['aae_depth'] = 2 # see Tble 9.1, PAS SUR DE LA SIGNIFICATION là !!!
parameters['aae_activation'] = 'relu' # see 9.2
parameters['aae_width'] = 1000 # not sure, see 9.1 fc(1000) and fc(6000)
parameters['max_temperature'] = 5.0
parameters['conv_depth'] = 3 # dans 9.1, encode ET decode ont chacun 3 Conv
parameters['conv_pooling'] = 1 # = train_common.py
parameters['conv_kernel'] = 5 # = train_common.py
parameters['conv_per_pooling']  = 1
parameters['conv_channel']  = 32
parameters['conv_channel_increment'] = 1
parameters['eff_regularizer'] = None
parameters['A'] = 6000 # max # of actions
parameters["optimizer"] = "radam"







parameters["aeclass"] = 'CubeSpaceAE_AMA4Conv'



transitions, states = load_puzzle('mnist', 3, 3, 40000, objects=False)
train, val, test = train_val_test_split(transitions)
transitions = train

transitions = transitions.astype('float32')


x =  transitions[:6] # (6, 2, 48, 48, 1) # 6 transitions d'image 48 x 48





# 1) Instanciation of the LatplanSAE from the lyr.functions module

res_SAE = lyr.functions.LatplanSAE("res_SAE",2304,99,[99]) # "99" because we don't use them




# 2) class that when called, return a tensor representing the truth value of x==y

# class areEqual(lyr.functions.AbstractFunction):

#     def __call__(self, x, y):

#         batch = tf.shape(x)[0]

#         x = tf.reshape(x, [batch, -1])

#         #y = SAE(x)

#         y = tf.reshape(y, [batch, -1])

#         return 1 - tf.reduce_mean(tf.abs(x - y), axis=1)


class areEqual():

    def __call__(self, x, y):

        return tf.cast(tf.equal(x,y), tf.float32)




# 3) Instanciation of the Domain class into an object "Images" 
#     the tensor of this object has same shape as an image (48, 48)

#Images = lyr.Domain(label="Image", data=tf.zeros(np.squeeze(transitions[0,0])))

#Images = lyr.Domain(label="Image", data=tf.zeros([transitions[0,0]]))
 
Images = lyr.Domain(label="Image", data=tf.zeros([48, 48]))


# 4) Instanciation of the Function class
#     with self.label = "SAE" and self.function = "res_SAE"

#F1 = lyr.Function("SAE", domains=("Image"), function=res_SAE)


# 5) Instantiation of the areEqual class
areEqual = areEqual()


# 6) Instanciation of the Relation class
lyr.Relation(label="areEqual", domains=("Image", "Image"), function = areEqual) # Constraint


# 7) Instanciation of the Constraint class
lyr.Constraint("forall x: areEqual(x, x)", 1.)




# The goal now is to do what ????

#       ===> 




loss =  lyr.current_world.loss()
train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)


exit()

#################################################################################################################



activate_rules = tf.placeholder(dtype=tf.bool, shape=[])

loss =  lyr.current_world.pointwise_loss


lr = tf.placeholder(dtype=tf.float32, shape=[])

loss = tf.cond(activate_rules, lambda: loss, lambda:loss)


print(tf.trainable_variables())


train_op = tf.train.AdamOptimizer(lr).minimize(loss)

exit()


# ensuite tu le lance
sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
sess.run(tf.global_variables_initializer())

while True:
    _, acc, ll = sess.run((train_op, loss))