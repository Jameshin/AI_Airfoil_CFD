import tensorflow.compat.v1 as tf
import numpy as np

def tf_session():
    # tf session
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(tf.global_variables())
 
    # init
    ckpt = tf.train.get_checkpoint_state('./model1')
    if ckpt  and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    return sess, saver

def relative_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(tf.reduce_mean(tf.square(pred - exact))/tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return tf.reduce_mean(tf.square(pred - exact))

def mean_fabs_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.fabs(pred - exact))
    return tf.reduce_mean(tf.abs(pred - exact))


def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    #print(x,Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    #print(G)
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x

class neural_net(object):
    def __init__(self, *inputs, layers):
        self.layers = layers
        self.num_layers = len(self.layers)
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1) #np.array(inputs)
            #print(X.shape)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)
        #print(self.X_mean, self.X_std)
 
        self.weights = []
        self.biases = []
        self.gammas = []
        
        for l in range(0,self.num_layers-1):
            in_dim = self.layers[l]
            out_dim = self.layers[l+1]
            #print(in_dim, out_dim)
            W = np.random.normal(size=[in_dim, out_dim]) #random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float64, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float64, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float64, trainable=True))            
            
    def __call__(self, *inputs):
        H = (tf.concat(inputs, 1) - self.X_mean)/self.X_std #tf.concat(inputs, 1)
        #print(inputs)
        for l in range(0, self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            #print(W.shape, '###################')
            # weight normalization
            V = W/tf.norm(W, axis = 0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            #print(H.shape, V.shape)
            # add bias
            H = g*H + b
            # activation
            if l < self.num_layers-2:
                H = H*tf.math.sigmoid(H) #tf.nn.relu(H)#tf.sigmoid(H) nn.tanh(H)
        print(H.shape)
        
        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
    
        return Y
