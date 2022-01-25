import tensorflow as tf
#fixed tensors
a=tf.zeros([2,3])
b = tf.fill([2,5],5)
c = tf.ones([4,4])
e = tf.constant(12,dtype='float')
print(a)
print(b)
print(c)
print(e)
##################################

#sequence 
f= tf.linspace(10,30,5)
g = tf.range(1,30,3)
print(b)