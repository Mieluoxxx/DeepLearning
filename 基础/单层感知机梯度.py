import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.random.normal([1, 3])
w = tf.ones([3, 1])
b = tf.ones([1])
y = tf.constant([1])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    prob = tf.sigmoid(x @ w + b)
    loss = tf.reduce_mean(tf.losses.MSE(y, prob))
grads = tape.gradient(loss, [w, b])

print(grads[0])
print(grads[1])
