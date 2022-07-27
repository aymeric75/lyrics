#!/usr/bin/env python3

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






print("OK0")





parameters["aeclass"] = 'CubeSpaceAE_AMA4Conv'



transitions, states = load_puzzle('mnist', 3, 3, 40000, objects=False)
train, val, test = train_val_test_split(transitions)
transitions = train
x =  transitions[:6] # (6, 2, 48, 48, 1) # 6 transitions d'image 48 x 48



# transitions[0,0] = 1ere image du dataset

#plt.imshow(transitions[0,0])
#plt.savefig('Image.png')

print("OK1")


# LOAD the model: FAIT dans functions.py !

# path = 'samples/puzzle_mnist_3_3_40000_CubeSpaceAE_AMA4Conv'
# task = curry(loadsNetWithWeightsGOOD, latplan.model.get(parameters["aeclass"]), path, train, train, val, val)
# _add_misc_info(parameters)
# parameters['hash'] = "8dd53f4ca49f65444250447a16903f86"
# os.chdir('../latplan')
# latplan_model, error = task(parameters)
# os.chdir('./')	





class isEqual():

    def __call__(self, x, y):
        batch = tf.shape(x)[0]

        x = tf.reshape(x, [batch, -1])
        y = tf.reshape(y, [batch, -1])
        return 1 - tf.reduce_mean(tf.abs(x - y), axis=1)



# 


Images = lyr.Domain(label="Images", data=np.squeeze(transitions[0,0]))

print("OK2")

SAE = lyr.functions.LatplanSAE("SAE",2304,99,[99]) # "99" because we don't use them

print("OK3")


lyr.Function("SAE", domains=("Images"), function=SAE)

print("OK4")

equal = isEqual()

print("OK5")

lyr.Relation(label="is", domains=("Images", "Images"), function = equal)
print("OK6")

lyr.Constraint("forall x: is(x, SAE(x))")
print("OK7")



loss =  lyr.current_world.loss()

########################################
# FROM AUTOENCODER version of Lyrics
########################################

# l = LogicFactory.create("lukasiewicz-strong")
# l1_loss_bab = 1 - l.forall(equal(bab,self.domain_B)) # reconstruction




lr = tf.placeholder(dtype=tf.float32, shape=[])
train_op = tf.train.AdamOptimizer(lr).minimize(loss)



# ensuite tu le lance
sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
sess.run(tf.global_variables_initializer())

while True:
    _, acc, ll = sess.run((train_op, loss))