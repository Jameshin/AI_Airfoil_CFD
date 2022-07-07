import tensorflow as tf
import numpy as np

def relative_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(tf.reduce_mean(tf.square(pred - exact))/tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.square(pred - exact))

def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    #print(x,Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    #print(G)
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x
