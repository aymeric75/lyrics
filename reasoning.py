#!/usr/bin/env python3

import tensorflow as tf
import os
import os.path
import numpy as np
from lyrics import lyrics as lyr
import matplotlib.pyplot as plt





people_repr_size = 1




class Equal():

    def __call__(self, a, b):

        return tf.cast(tf.equal(a,b), tf.float32)

# class IndexingFunction():

#     def __init__(self, k):
#         self.k = k
#         self.var = tf.Variable(initial_value= -4 * tf.ones([k*k]))
#     def call(self, a, b):
#         a = tf.cast(a, tf.int32)
#         b = tf.cast(b, tf.int32)
#         idx = self.k * a + b
#         return tf.sigmoid(tf.gather(self.var, idx))
k = 6




#dom = lyr.Domain(label="People", data=tf.zeros([0,1]))

dom = lyr.Domain(label="People", data=tf.zeros([1, 1]))


#lyr.Individual(label="Andrea", domain="People", value=[5])

print(" ")
print("ici0")

print(dom.tensor)

print("ici1")


print(tf.shape(dom.tensor))

print("ic2")


print(dom.tensor.get_shape())

print("ici3")


print(dom.columns)



ind1 = lyr.Individual(label="Jean", domain="People", value=tf.Variable([[45]], dtype=tf.float32) )


print("ind1.tensor")

print(ind1.tensor)


tfs = tf.InteractiveSession()


tfs.run(tf.global_variables_initializer())

print(tfs.run(ind1.tensor))


equal = Equal()
lyr.Relation(label="is", domains=("People", "People"), function =equal)



lyr.Constraint("is(Jean,Jean)",0.1)



loss =  lyr.current_world.loss()
train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)



exit()


lyr.Domain(label="Image", data=tf.zeros([48, 48]))



lyr.Individual(label="Im1", domain="Image", value=[45.])

# lyr.Individual(label="Marco", domain="People", value=[0])
# lyr.Individual(label="Giuseppe", domain="People", value=[1])
# lyr.Individual(label="Michelangelo", domain="People", value=[2])
# lyr.Individual(label="Francesco", domain="People", value=[3])
# lyr.Individual(label="Franco", domain="People", value=[4])
# lyr.Individual(label="Andrea", domain="People", value=[5])

# fo = lyr.functions.BinaryIndexFunction("fo",k,k)
# gfo = lyr.functions.BinaryIndexFunction("gfo",k,k)


equal = Equal()

# lyr.Relation(label="fatherOf", domains=("People", "People"), function=fo)
# lyr.Relation(label="grandFatherOf", domains=("People", "People"), function=gfo)
lyr.Relation(label="is", domains=("Image", "Image"), function =equal)

# lyr.Constraint("fatherOf(Marco, Giuseppe)")
# lyr.Constraint("fatherOf(Giuseppe, Michelangelo)")
# lyr.Constraint("fatherOf(Giuseppe, Francesco)")
# lyr.Constraint("fatherOf(Franco, Andrea)")


lyr.Constraint("forall x: is(x,x)",0.1)



# lyr.Constraint("forall x: forall y: forall z: fatherOf(x,z) and fatherOf(z,y) -> grandFatherOf(x,y)",0.1)
# lyr.Constraint("forall x: forall y: fatherOf(x,y) -> not grandFatherOf(x,y)", 0.1)
# lyr.Constraint("forall x: not fatherOf(x,x)")
# lyr.Constraint("forall x: not grandFatherOf(x,x)")
# lyr.Constraint("forall x: forall y: grandFatherOf(x,y) -> not fatherOf(x,y)",0.1)

# lyr.Constraint("forall x: forall y: fatherOf(x,y) -> not fatherOf(y,x)", 0.1)
# lyr.Constraint("forall x: forall y: grandFatherOf(x,y) -> not grandFatherOf(y,x)", 0.1)

# lyr.Constraint("forall x: forall y: grandFatherOf(x,y) -> not fatherOf(y,x)", 0.1)
# lyr.Constraint("forall x: forall y: fatherOf(x,y) -> not grandFatherOf(y,x)", 0.1)



loss =  lyr.current_world.loss()
train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)


print("hrllo")




#learn



