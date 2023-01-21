import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
w = tf.Variable(1.)
b = tf.Variable(2.)
x = tf.Variable(3.)

with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = x * w + b
    dy_dw, dy_db = t2.gradient(y, [w, b])
d2y_dw2 = t1.gradient(dy_dw, w)

print(dy_dw)
print(dy_db)
print(d2y_dw2)

assert dy_dw.numpy() == 3.
assert d2y_dw2 is None
