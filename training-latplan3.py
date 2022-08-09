#!/usr/bin/env python3


#
import sys
import os

# All latplan imports
sys.path.append(r"../latplan")
import latplan
import latplan.model
import latplan.util.stacktrace
from latplan.util.tuning import simple_genetic_search, parameters, nn_task, reproduce, load_history, loadsNetWithWeights, loadsNetWithWeightsGOOD
from latplan.util        import curry
from latplan.main.common import train_val_test_split
from latplan.main.puzzle import load_puzzle
from train_common import parameters


from lyrics import lyrics as lyr
from lyrics import fuzzy as fuzz

import tensorflow as tf
from sklearn import datasets as data
import matplotlib.pyplot as plt






class isEqual():


    def __init__(self, var = None):
        super(isEqual, self).__init__()
        self.var = tf.get_variable(name="randomVar", shape=(36000, 1))

    def __call__(self, x, y):
        #batch = tf.shape(x)[0]
        #x = tf.reshape(x, [batch, -1])
        #y = tf.reshape(y, [batch, -1])
        print("shapess")
        print(x)
        #print(tf.shape(x))
        print(y)
        #print(tf.shape(y))
        #aa = 1. - tf.reduce_mean(tf.abs(x - y), axis=1)
        aa = 1. - tf.reduce_mean(tf.abs(x - y), [1,2,3])
        print("aa")
        print(aa)
        self.var.assign(aa)
        return self.var





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
parameters['hash'] = "8dd53f4ca49f65444250447a16903f86"
transitions, states = load_puzzle('mnist', 3, 3, 40000, objects=False)
train, val, test = train_val_test_split(transitions)

# print(len(train)) = 36000
# print(len(val)) = 2000
# print(len(test)) = 2000

#x =  transitions[:6] # (6, 2, 48, 48, 1) # 6 transitions d'image 48 x 48
path = 'samples/puzzle_mnist_3_3_40000_CubeSpaceAE_AMA4Conv'



with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    ################################################
    # build model
    # <=> declaration de tous les inputs + 
    ################################################

    print("debut")

    #train = tf.placeholder(tf.float32, shape=(36000, 2, 48, 48, 1))
    #val = tf.placeholder(tf.float32, shape=(2000, 2, 48, 48, 1))

    task = curry(loadsNetWithWeightsGOOD, latplan.model.get(parameters["aeclass"]), path, train, train, val, val)        
    _add_misc_info(parameters)
    os.chdir('../latplan')

    latplan_model, error = task(parameters)
    y = latplan_model.autoencode(train) #
    os.chdir('./')


    # predictions
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # formatting the training set
    epochs = 10
    batch_size = 64
    iterations = len(train) * epochs
    dataset = tf.data.Dataset.from_tensor_slices((train))
    dataset = dataset.repeat(epochs).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    data_y = iterator.get_next()
    data_y = tf.cast(data_y, tf.float32)



    # Loss
    l = fuzz.LogicFactory.create("lukasiewicz-strong")
    equal = isEqual()

    print("yyyyy")
    print(y)
    print("data_yyyyyy")
    print(data_y)
    print("##########")

    l1_loss = 1 - l.forall(equal(y, data_y))


    t_vars = tf.trainable_variables()
    lr = tf.placeholder(tf.float32, name='learning_rate')
    # Optimizer
    #tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(l1_loss, var_list = t_vars)
    #tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(l1_loss)
    opt = tf.train.AdamOptimizer().minimize(l1_loss)
    ################################################
    #                    TRAIN                     #
    ################################################

    print("fin")

    tf.global_variables_initializer().run()

    for epoch in range(0, 2):

        _, d_loss = sess.run([opt, l1_loss])
        print("epoch ")
        print(epoch)


exit()

# Define Domain and one Individual example for the input/output Images


dom = lyr.Domain(label="Image", data=tf.random_normal([48,48]))

Im1 = lyr.Individual(label="Im1", domain="Image", value=tf.Variable(tf.random_normal([48,48])))

Im2 = lyr.Individual(label="Im2", domain="Image", value=tf.Variable(tf.random_normal([48,48])))


# le loss sera définit à partir de la Constraint (car on fait loss =  lyr.current_world.loss())
# 				et cette Constraint s'exprime avec une function (ici "is" / AreEqual)
# 					c'est dans AreEqual que tu retourne le mean_squared_error entre les deux images


# we instanciate the AreEqual function into the simil variable
simil = lyr.functions.AreEqual("simil")


resSAE = lyr.functions.LatplanSAE("resSAE",2304,99,[99]) # "99" because we don't use them



# we associate the relation "is" to corresponds to the output of the AreEqual variable (inputs being two individual of the Domain "Image")
lyr.Relation(label="is", domains=("Image", "Image"), function =simil)




# we call a constraint consisting of the test of "is" on two individuals Im1 and Im2
lyr.Constraint("forall x: is(x, resSAE(x))")



# We create the loss and optimizer based on this Constraint

# !! loss (fnctions.AreEqual !!!) must be the ouput of mse(predictions=model, labels=y) WHERE y = tf.placeholder 

# AND model is the last layer of the constructed model 
# AND where this constructed model has been built with x(=tf placeholder) as a first layer !!!!!

loss =  lyr.current_world.loss() # retrieve all losses, included the Constraint one
train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)

