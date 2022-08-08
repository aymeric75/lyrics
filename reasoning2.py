#!/usr/bin/env python3

# from tensorflow.contrib.eager.python import tfe
# eager = True
# if eager: tfe.enable_eager_execution()
#import lyrics as lyr
from lyrics import lyrics as lyr
import tensorflow as tf
from sklearn import datasets as data
import matplotlib.pyplot as plt


people_repr_size = 1




class Equal():

    def __call__(self, a, b):

        return tf.cast(tf.equal(a,b), tf.float32)



print("ESSAI")


k = 6

lyr.Domain(label="People", data=tf.zeros([0,1]))




lyr.Individual(label="Marco", domain="People", value=[0])
lyr.Individual(label="Giuseppe", domain="People", value=[1])



fo = lyr.functions.BinaryIndexFunction("fo",k,k)



sim = lyr.functions.AreEqual("sim")



lyr.Relation(label="fatherOf", domains=("People", "People"), function=fo)



lyr.Relation(label="is", domains=("People", "People"), function =sim)




lyr.Constraint("is(Marco, Giuseppe)")


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

    # if lyr.utils.heardEnter():
    #     break
    # if i%1000==0:
    #     print(l)


print("fr3")


exit()