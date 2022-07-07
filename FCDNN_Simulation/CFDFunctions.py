"""
@author: Maziar Raissi
"""

import tensorflow.compat.v1 as tf
import numpy as np

def tf_session():
    # tf session
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.Session(config=config)
    
    # init
    init = tf.global_variables_initializer()
    sess.run(init)
    
    return sess

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

class neural_net(object):
    def __init__(self, *inputs, layers):
        self.layers = layers
        self.num_layers = len(self.layers)
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.array(inputs) #np.concatenate(inputs, 1)
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
        H = tf.concat(inputs, 1) #(tf.concat(inputs, 1) - self.X_mean)/self.X_std
        #print(inputs)
        for l in range(0, self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W/tf.norm(W, axis = 0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            #print(H.shape, V.shape)
            # add bias
            H = g*H + b
            # activation
            if l < self.num_layers-2:
                H = H*tf.nn.sigmoid(H) #tf.nn.relu(H)#tf.sigmoid(H) nn.tanh(H)
        print(H.shape)
        
        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
    
        return Y

def Euler_uIncomp_POD(a, phi, mean_data, t, x, y, Pec, Gamma):
  
    T = phi.shape[1]
    N = phi.shape[0]
    #t = tf.reshape(tf.tile(t_pod, tf.constant([1,N], tf.int32)), [-1])
    #x = tf.reshape(tf.transpose(tf.tile(x_pod, tf.constant([1,T], tf.int32))), [-1])
    #y = tf.reshape(tf.transpose(tf.tile(y_pod, tf.constant([1,T], tf.int32))), [-1])
    noConcernVar = 4
    #a = a_rep[0:-1:N,:]
    U_pod_pred = tf.add(tf.transpose(tf.matmul(a, tf.transpose(tf.constant(phi, tf.float64)))), 
                        tf.tile(tf.constant(mean_data, tf.float64), tf.constant([1,T], tf.int32)))
    for i in range(T):
        temp = tf.reshape(U_pod_pred[:,i], [-1, noConcernVar])
        #print(T, temp)
        if i == 0:
            U_pred = temp
        else:
            U_pred = tf.concat([U_pred, temp], axis=0)
    d = U_pred[:,0]
    u = U_pred[:,1][:,None]
    v = U_pred[:,2][:,None]
    p = U_pred[:,3][:,None]

    Y = tf.concat([u, v, p], 1)
    print(x, U_pred)
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    #Y_xx = fwd_gradients(Y_x, x)
    #Y_yy = fwd_gradients(Y_y, y)
    #print(Y_t)

    u = Y[:,0]
    v = Y[:,1]
    p = Y[:,2]

    u_t = Y_t[:,0]
    v_t = Y_t[:,1]
    #u_t = tf.gradients(u,t)
    #v_t = tf.gradients(v,t)

    u_x = Y_x[:,0]
    v_x = Y_x[:,1]
    p_x = Y_x[:,2]
    #u_x = tf.gradients(u,x)
    #v_x = tf.gradients(v,x)
    #p_x = tf.gradients(p,x)
    print(x, u, u_x, v_x)
    u_y = Y_y[:,0]
    v_y = Y_y[:,1]
    p_y = Y_y[:,2]
    #u_y = tf.gradients(u,y)
    #v_y = tf.gradients(v,y)
    #p_y = tf.gradients(p,y)

    #d_xx = Y_xx[:,0:1]
    #u_xx = Y_xx[:,1:2]
    #v_xx = Y_xx[:,2:3]

    #d_yy = Y_yy[:,0:1]
    #u_yy = Y_yy[:,1:2]
    #v_yy = Y_yy[:,2:3]

    e1 = d_x*u + d*u_x + d_y*v + d*v_y #+ d_t
    e2 = d*u*u_x + d*v*u_y + p_x #+ d*u_t
    e3 = d*u*v_x + d*v*v_y + p_y #+ d*v_t
    e4 = d*u*H_x + d*v*H_y #+ d*H_t + d_t*H - p_t

    return e1, e2, e3

def Euler_uComp_2D(d, u, v, p, t, x, y, Pec, Gamma):

    Y = tf.concat([d, u, v], 1)
    E = p/d/(Gamma-1) + 0.5*(u*u+v*v)
    H = E + p/d
    #print(d.shape,u.shape,v.shape)
    Yp = tf.concat([p, H], 1)

    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    Yp_t = fwd_gradients(Yp, t)
    Yp_x = fwd_gradients(Yp, x)
    Yp_y = fwd_gradients(Yp, y)

    d = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    p = Yp[:,0:1]
    H = Yp[:,1:2] 

    d_t = Y_t[:,0:1]
    u_t = Y_t[:,1:2]
    v_t = Y_t[:,2:3]
    p_t = Yp_t[:,0:1]
    H_t = Yp_t[:,1:2]

    d_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    p_x = Yp_x[:,0:1]
    H_x = Yp_x[:,1:2]

    d_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    p_y = Yp_y[:,0:1]
    H_y = Yp_y[:,1:2]

    e1 = d_x*u + d*u_x + d_y*v + d*v_y + d_t
    e2 = d*u*u_x + d*v*u_y + p_x + d*u_t
    e3 = d*u*v_x + d*v*v_y + p_y + d*v_t
    e4 = d*u*H_x + d*v*H_y + d*H_t + d_t*H - p_t
    #print(e1,e2,e3,e4)

    return e1, e2, e3, e4


def Euler_uIncomp_2D(c, u, v, p, t, x, y, Pec, Rey):
        
    Y = tf.concat([c, u, v, p], 1)

    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)

    c = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    p = Y[:,3:4]
 
    c_t = Y_t[:,0:1]
    u_t = Y_t[:,1:2]
    v_t = Y_t[:,2:3]
 
    c_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    p_x = Y_x[:,3:4]
 
    c_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    p_y = Y_y[:,3:4]

    c_xx = Y_xx[:,0:1]
    u_xx = Y_xx[:,1:2]
    v_xx = Y_xx[:,2:3]
 
    c_yy = Y_yy[:,0:1]
    u_yy = Y_yy[:,1:2]
    v_yy = Y_yy[:,2:3]
 
    e1 = c_t + (u*c_x + v*c_y) - (1.0/Pec)*(c_xx + c_yy)
    e2 = u_t + (u*u_x + v*u_y) + p_x - (1.0/Rey)*(u_xx + u_yy)
    e3 = v_t + (u*v_x + v*v_y) + p_y - (1.0/Rey)*(v_xx + v_yy)
    e4 = u_x + v_y
 
    return e1, e2, e3, e4

def NS_Incomp_2D(c, u, v, p, t, x, y, Pec, Rey):
    
    Y = tf.concat([c, u, v, p], 1)
    
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    
    c = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    p = Y[:,3:4]
    
    c_t = Y_t[:,0:1]
    u_t = Y_t[:,1:2]
    v_t = Y_t[:,2:3]
    
    c_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    p_x = Y_x[:,3:4]
    
    c_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    p_y = Y_y[:,3:4]
    
    c_xx = Y_xx[:,0:1]
    u_xx = Y_xx[:,1:2]
    v_xx = Y_xx[:,2:3]
    
    c_yy = Y_yy[:,0:1]
    u_yy = Y_yy[:,1:2]
    v_yy = Y_yy[:,2:3]
    
    e1 = c_t + (u*c_x + v*c_y) - (1.0/Pec)*(c_xx + c_yy)
    e2 = u_t + (u*u_x + v*u_y) + p_x - (1.0/Rey)*(u_xx + u_yy) 
    e3 = v_t + (u*v_x + v*v_y) + p_y - (1.0/Rey)*(v_xx + v_yy)
    e4 = u_x + v_y
    
    return e1, e2, e3, e4

def Gradient_Velocity_2D(u, v, x, y):
    
    Y = tf.concat([u, v], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    
    return [u_x, v_x, u_y, v_y]

def Strain_Rate_2D(u, v, x, y):
    
    [u_x, v_x, u_y, v_y] = Gradient_Velocity_2D(u, v, x, y)
    
    eps11dot = u_x
    eps12dot = 0.5*(v_x + u_y)
    eps22dot = v_y
    
    return [eps11dot, eps12dot, eps22dot]

def Navier_Stokes_3D(c, u, v, w, p, t, x, y, z, Pec, Rey):
    
    Y = tf.concat([c, u, v, w, p], 1)
    
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    Y_zz = fwd_gradients(Y_z, z)
    
    c = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    w = Y[:,3:4]
    p = Y[:,4:5]
    
    c_t = Y_t[:,0:1]
    u_t = Y_t[:,1:2]
    v_t = Y_t[:,2:3]
    w_t = Y_t[:,3:4]
    
    c_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    w_x = Y_x[:,3:4]
    p_x = Y_x[:,4:5]
    
    c_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    w_y = Y_y[:,3:4]
    p_y = Y_y[:,4:5]
       
    c_z = Y_z[:,0:1]
    u_z = Y_z[:,1:2]
    v_z = Y_z[:,2:3]
    w_z = Y_z[:,3:4]
    p_z = Y_z[:,4:5]
    
    c_xx = Y_xx[:,0:1]
    u_xx = Y_xx[:,1:2]
    v_xx = Y_xx[:,2:3]
    w_xx = Y_xx[:,3:4]
    
    c_yy = Y_yy[:,0:1]
    u_yy = Y_yy[:,1:2]
    v_yy = Y_yy[:,2:3]
    w_yy = Y_yy[:,3:4]
       
    c_zz = Y_zz[:,0:1]
    u_zz = Y_zz[:,1:2]
    v_zz = Y_zz[:,2:3]
    w_zz = Y_zz[:,3:4]
    
    e1 = c_t + (u*c_x + v*c_y + w*c_z) - (1.0/Pec)*(c_xx + c_yy + c_zz)
    e2 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - (1.0/Rey)*(u_xx + u_yy + u_zz)
    e3 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - (1.0/Rey)*(v_xx + v_yy + v_zz)
    e4 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - (1.0/Rey)*(w_xx + w_yy + w_zz)
    e5 = u_x + v_y + w_z
    
    return e1, e2, e3, e4, e5

def Gradient_Velocity_3D(u, v, w, x, y, z):
    
    Y = tf.concat([u, v, w], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    w_x = Y_x[:,2:3]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    w_y = Y_y[:,2:3]
    
    u_z = Y_z[:,0:1]
    v_z = Y_z[:,1:2]
    w_z = Y_z[:,2:3]
    
    return [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z]

def Shear_Stress_3D(u, v, w, x, y, z, nx, ny, nz, Rey):
        
    [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z] = Gradient_Velocity_3D(u, v, w, x, y, z)

    uu = u_x + u_x
    uv = u_y + v_x
    uw = u_z + w_x
    vv = v_y + v_y
    vw = v_z + w_y
    ww = w_z + w_z
    
    sx = (uu*nx + uv*ny + uw*nz)/Rey
    sy = (uv*nx + vv*ny + vw*nz)/Rey
    sz = (uw*nx + vw*ny + ww*nz)/Rey
    
    return sx, sy, sz
