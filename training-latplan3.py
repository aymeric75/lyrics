#!/usr/bin/env python3

import tensorflow as tf

from tensorflow import keras
from keras import backend as K
from keras_radam  import RAdam
import keras.optimizers
setattr(keras.optimizers,"radam", RAdam)


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


from sklearn import datasets as data
import matplotlib.pyplot as plt


import numpy as np



def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        
        if hasattr(layer, 'kernel_initializer'):
            print("re-init weights of layer : "+layer.name)
            layer.kernel.initializer.run(session=session)




def my_loss_fn(y_true, y_pred):

    # y_true et y_pred sont (36000, 2, 48, 48, 1)

    res_equal = 1. - tf.reduce_mean(tf.abs(y_pred - y_true), [1,2,3])

    loss = 1 - tf.reduce_sum(res_equal, axis=0)

    return loss

    # doit retourner un array of loss, où chaque valeur correspond à 1 valeur du batch
    # donc retourne un array de dim le batch



class isEqual():


    def __init__(self, var = None):
        super(isEqual, self).__init__()
        self.var = tf.get_variable(name="randomVar", shape=(36000, 1))

    def __call__(self, x, y):
      
        # we compute mean across pairs (1), and accross dim of images (2, 3)
        # dim 0 (= #examples) is left for reduce_sum (see l.forall)
        aa = 1. - tf.reduce_mean(tf.abs(x - y), [1,2,3])
        self.var.assign(aa)
        return self.var


def model_loss_function(model, data):

    #model.load()

    y_true = data # car on est dans le cas d'un autoencoder ici

    y_pred = model.predict(data)

    res_equal = 1. - tf.reduce_mean(tf.abs(y_pred - y_true), [1,2,3])

    loss = 1 - tf.reduce_sum(res_equal, axis=0)

    return loss


def return_iterator(data, nb_epochs, batch_size): # return get_next

    dataset = tf.data.Dataset.from_tensor_slices(data) # = 36k Tensors de taille (2, 48, 48, 1) chacun
                                                        # <DatasetV1Adapter shapes: (2, 48, 48, 1), types: tf.float64>
    dataset = dataset.repeat(nb_epochs).batch(batch_size) # repeat the train data 'epoch' times (10 x 36k) et recombine en batchs 
                                                       # de taille 'batch_size' (64)
                                                       # => 5 625 'sets'
                                                       # <DatasetV1Adapter shapes: (?, 2, 48, 48, 1), types: tf.float64>
    iterator = dataset.make_one_shot_iterator() # creates the iterator
    yy = iterator.get_next()

    return  tf.cast(yy, tf.float32)


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


# train : (36000, 2, 48, 48, 1)
# print(len(train)) = 36000
# print(len(val)) = 2000
# print(len(test)) = 2000

#x =  transitions[:6] # (6, 2, 48, 48, 1) # 6 transitions d'image 48 x 48
path = 'samples/puzzle_mnist_3_3_40000_CubeSpaceAE_AMA4Conv'




task = curry(loadsNetWithWeightsGOOD, latplan.model.get(parameters["aeclass"]), path, train, train, val, val)        
_add_misc_info(parameters)
os.chdir('../latplan')
latplan_model, error = task(parameters)

os.chdir('./')




# instance of the Model class from Keras
model = latplan_model.autoencoder


reset_weights(model)


# numpy array of (36000, 2, 48, 48, 1)
train_data = train 


def onetrainstep(model, data):

    with tf.GradientTape() as tape:
       #logits = model(data)
       logits = model.train_on_batch(data, data)

       evals = model.evaluate(data, data, steps=400, verbose=0)
       print(evals)


print(keras.__version__)

model.compile(optimizer='adam', loss=my_loss_fn)
train_data = np.array(np.split(train_data, 90))



for step in range(100):


    the_batch_data = tf.cast(train_data[step], dtype=tf.float32)
    onetrainstep(model, the_batch_data)

    if(step%20 == 0):

        prediction = model.predict(np.expand_dims(test[step], axis=0))

        plt.figure(figsize=(6,6))
        plt.imshow(np.squeeze(prediction[0][0]),interpolation='nearest',cmap='gray')
        plt.savefig('im'+str(step)+'.png')
        # display the value of the loss function 


exit()


















with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:


    ################################################
    # build model
    # <=> declaration de tous les inputs + 
    ################################################

    # y = all predictions of the autoencoder over the training set (train)
    task = curry(loadsNetWithWeightsGOOD, latplan.model.get(parameters["aeclass"]), path, train, train, val, val)        
    _add_misc_info(parameters)
    os.chdir('../latplan')
    latplan_model, error = task(parameters)
    y = latplan_model.autoencode(train)
    #y = latplan_model.autoencode(exxxx) #

    os.chdir('./')
    y = tf.convert_to_tensor(y, dtype=tf.float32)



    # type(train) <class 'numpy.ndarray'>
    # train shape: (36000, 2, 48, 48, 1)

    # type(y) <class 'tensorflow.python.framework.ops.Tensor'>
    # y shape: (36000, 2, 48, 48, 1)
    # y : Tensor("Const_18:0", shape=(36000, 2, 48, 48, 1), dtype=float32)




    # data_y = all the original images (formatted here, into tensors)
    nb_epochs = 10
    batch_size = 64
    data_y = return_iterator(train, nb_epochs, batch_size)
    y = return_iterator(y, nb_epochs, batch_size)


    #print("data_y") Tensor("Cast_4:0", shape=(?, 2, 48, 48, 1), dtype=float32)

    #print("y") Tensor("IteratorGetNext_1:0", shape=(?, 2, 48, 48, 1), dtype=float32)


    # instanciation of isEqual (returns the mean between 2 sequences of images, i.e, originals and predicted here)
    equal = isEqual()


    # Loss class, instanciated
    l = fuzz.LogicFactory.create("lukasiewicz-strong")
    
    # call of the forall function from the "l" class
    # concretly, does a reduc_sum over all the images (e.g 36 000 the training set)
    
    #l1_loss = 1 - l.forall(equal(y, data_y))

    # 1. - tf.reduce_mean(tf.abs(x - y), [1,2,3])

    res_equal = 1. - tf.reduce_mean(tf.abs(y - data_y), [1,2,3])

    l1_loss = 1 - tf.reduce_sum(res_equal, axis=0)

    #l1_loss = 1 - l.forall(equal(latplan_model.autoencoder, data_y))



    #
    #    
    #    equal :         1. - tf.reduce_mean(tf.abs(model - y_tensor), [1,2,3])  #  
    #
    #    forall :        
    #       y_data : Tensor("Cast_4:0", shape=(?, 2, 48, 48, 1), dtype=float32)
    #       y : Tensor("Const_18:0", shape=(36000, 2, 48, 48, 1), dtype=float32)
    #
    #     



    # could be useful ... (see Lyrics Autoencoder example, UNIT_CLARE_2)
    t_vars = tf.trainable_variables()
    lr = tf.placeholder(tf.float32, name='learning_rate')

    # Optimizer
    opt = tf.train.AdamOptimizer().minimize(l1_loss)



    ################################################
    #                    TRAIN                     #
    ################################################





    train_dataset = tf.data.Dataset.from_tensor_slices((train, train))

    epochs = 2

    for epoch in range(epochs):

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            with tf.GradientTape() as tape:

                logits = model(x_batch_train, training=True)

                loss_value = my_loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))


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

