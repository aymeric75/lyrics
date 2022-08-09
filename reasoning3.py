#!/usr/bin/env python3

from lyrics import lyrics as lyr
import tensorflow as tf
from sklearn import datasets as data
import matplotlib.pyplot as plt


people_repr_size = 1




class Equal():

    def __call__(self, a, b):

        return tf.cast(tf.equal(a,b), tf.float32)




k = 6

# dom = lyr.Domain(label="Image", data=tf.random_normal([48,48]))
# Im1 = lyr.Individual(label="Im1", domain="Image", value=tf.Variable(tf.random_normal([48,48])))
# Im2 = lyr.Individual(label="Im2", domain="Image", value=tf.Variable(tf.random_normal([48,48])))



#sim = lyr.functions.AreEqual("sim")


#lyr.Relation(label="is", domains=("Image", "Image"), function =sim)
#lyr.Constraint("is(Im1, Im2)")



dom = lyr.Domain(label="People", data=tf.zeros([1, 1]))



lyr.Individual(label="Marco", domain="People", value=[0])
lyr.Individual(label="Giuseppe", domain="People", value=[1])



#fo = lyr.functions.BinaryIndexFunction("fo",k,k)
#lyr.Relation(label="fatherOf", domains=("People", "People"), function=fo)

sim = lyr.functions.AreEqual("sim")
lyr.Relation(label="is", domains=("People", "People"), function =sim)



lyr.Constraint("is(Marco, Giuseppe)")


exit()


loss =  lyr.current_world.loss()
train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)






sess = tf.Session()
sess.run(tf.global_variables_initializer())



print("fr")
epochs = 2
for i in range(epochs):
    print("fr2")
    _, l = sess.run((train_op, loss ))

    print(l)



print("fr3")


exit()