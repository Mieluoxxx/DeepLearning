import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@tf.function
def tf_cube(x):
    return tf.pow(x, 3)


print(tf_cube.python_function(2))
