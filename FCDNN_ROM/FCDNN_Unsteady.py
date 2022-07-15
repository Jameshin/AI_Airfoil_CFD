#"""
#@original author: Maziar Raissi
# edited by James Shin
#"""

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io
import time
import sys
import os
import pandas as pd

from CFDFunctions3 import neural_net, tf_session, mean_squared_error, relative_error

tf.compat.v1.disable_eager_execution()

class DLROM(object):
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _data: input-output data
    
    def __init__(self, xydata, t_data, x_data, y_data,
                 d_data, u_data, v_data, p_data, 
                 layers, batch_size, Pec, Rey):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size
        
        # flow properties
        self.Pec = Pec
        self.Rey = Rey
     
        noConcernsVar = 4 
        N = x_data.shape[0]
        T = t_data.shape[0]
        # data        
        [self.xydata, self.t_data, self.x_data, self.y_data, self.d_data, self.u_data, self.v_data, self.p_data] = [xydata, t_data, x_data, y_data, d_data, u_data, v_data, p_data]
        
        # placeholders
        [self.d_data_tf, self.u_data_tf, self.v_data_tf, self.p_data_tf] = [tf.placeholder(tf.float64, shape=[None, 1]) for _ in range(4)]
        [self.t_data_tf, self.x_data_tf, self.y_data_tf] = [tf.placeholder(tf.float64, shape=[None, 1]) for _ in range(3)]

        # neural networks
        self.net_duvp = neural_net(self.t_data, self.x_data, self.y_data, layers = self.layers) #[3,12,12,12,12,12,12,12,4])
        
        [self.d_data_pred, 
         self.u_data_pred,
         self.v_data_pred,
         self.p_data_pred] = self.net_duvp(self.t_data_tf,
                                           self.x_data_tf,
                                           self.y_data_tf)   
        
        # loss
        self.loss = mean_squared_error(self.d_data_pred, self.d_data_tf) + \
                    mean_squared_error(self.u_data_pred, self.u_data_tf) + \
                    mean_squared_error(self.v_data_pred, self.v_data_tf) + \
                    mean_squared_error(self.p_data_pred, self.p_data_tf) 
        
        # optimizers
        self.global_step = tf. Variable(0, trainable = False, name='global_step')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
       
        self.sess, self.saver = tf_session()
    
    def train(self, total_time, learning_rate):
        
        N_data = self.xydata.shape[0]        
        start_time = time.time()
        running_time = 0
        it = 0
        while running_time < total_time:
            
            idx_data = np.random.choice(N_data, min(self.batch_size, N_data))
            if it == 50000:
                learning_rate = 1e-3
            if it == 100000:
                learning_rate = 5e-4
            if it == 150000:
                learning_rate = 1e-4
            if it == 200000:
                learning_rate = 5e-5
            if it == 250000:
                learning_rate = 1e-5
            if it == 300000:
                learning_rate = 1e-6
            if it == 350000:
                learning_rate = 1e-7
            if it == 400000:
                learning_rate = 1e-8
            (t_data_batch,
             x_data_batch,
             y_data_batch, 
             d_data_batch, u_data_batch, 
             v_data_batch, p_data_batch) = (self.t_data[idx_data,:],
                              self.x_data[idx_data,:],
                              self.y_data[idx_data,:],
                              self.d_data[idx_data,:],
                              self.u_data[idx_data,:],
                              self.v_data[idx_data,:],
                              self.p_data[idx_data,:])
 

            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.u_data_tf: u_data_batch,
                       self.v_data_tf: v_data_batch,
                       self.d_data_tf: d_data_batch,
                       self.p_data_tf: p_data_batch,
                       self.learning_rate: learning_rate}
            
            self.sess.run([self.train_op], tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value,
                 learning_rate_value] = self.sess.run([self.loss,
                                                       self.learning_rate], tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                      %(it, loss_value, elapsed, running_time, learning_rate_value))
                sys.stdout.flush()
                start_time = time.time()
            it += 1
        # save model parameters
        self.saver.save(self.sess, './model0/dnn.ckpt', global_step = self.global_step)
    
    def predict(self, t_star, x_star, y_star, l_star):

        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star, self.l_data_tf: l_star}

        d_star = self.sess.run(self.d_data_pred, tf_dict)
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)

        return d_star, u_star, v_star, p_star 
    
  
if __name__ == "__main__":
    with tf.device('/gpu:0'):
        batch_size = 40000  #200000
        layers = [3] + 10*[4*10] + [4]  #[4] + 10*[4*10] + [4]
    
        # Load Data
        sim_data_path = "~/AIRFOIL/Unsteady/Eppler387/sol01_RANS3"
        # create directory if not exist
        #os.makedirs(os.path.dirname(res_data_path), exist_ok=True)
        # list of file names
        filenames = []
        merged = []
        Ntime = 0
        numd = 20
        initial_time = 101
        inc_time = 4
        dt = 0.1
        glayer = 145
        cuttail = 0
        for i in range(0, numd):
            filenames.append(sim_data_path+"flo001.0000"+str(initial_time+i*inc_time).rjust(3,'0')+"uns")
            Ntime += 1
        #print(Ntime, filenames)
        t_star = np.arange(initial_time, initial_time+numd*inc_time+1, inc_time)*dt # 1xT(=1)
        ###
        infer_times = np.arange(initial_time+1, initial_time+numd*inc_time+2, inc_time)*dt
        noConcernVar = 4
        zone1_i = 689
        zone1_j = 145

        ###        
        saved_npz = np.load("./array4.npz")
        #read xy-coordinates
        #pd_data = pd.read_csv('./xy.csv', dtype='float64', delimiter=' ', header=None, skipinitialspace=True)
        #xydata = pd_data.values
        XC_star = saved_npz['XC']
        YC_star = saved_npz['YC']
        xydata = np.hstack((XC_star[:,0][:,None], YC_star[:,0][:,None]))

        ### extract airfoil surface nodes
        idx_bottom = np.where(xydata[:,0] == xydata[0,0])[0]        
        for i in range(idx_bottom[1]):
            if(xydata[i,1] != xydata[idx_bottom[1]-i,1]):
                break
        idx_tip = [i-1, idx_bottom[1]-i+1]
        idx_sur = []
        for j in range(zone1_j-2):
            idx_sur[j] = idx_sur.append(np.arange(idx_tip[0]+(j+1)*(zone1_i-2*cuttail), idx_tip[1]+(j+1)*(zone1_i-2*cuttail)+1))
        idx_x_sur = np.arange(idx_tip[0],idx_tip[1]+1)
        idx_x_sur2 = np.arange(zone1_i, 2*zone1_i)
        idx_x_sur3 = np.arange(2*zone1_i, 3*zone1_i)
        idx_x_bd1 = []
        idx_x_ff_data_i1 = np.arange(0, (glayer-2)*(zone1_i-2*cuttail)+1, zone1_i-2*cuttail)
        idx_x_ff_data_i2 = np.arange(zone1_i-2*cuttail-1, (glayer-1)*(zone1_i-2*cuttail), zone1_i-2*cuttail)
        idx_x_ff_data_o = np.arange((glayer-1)*(zone1_i-2*cuttail), zone1_i-2*cuttail+(glayer-1)*(zone1_i-2*cuttail))
        idx_x_ff_data_i = np.append(idx_x_ff_data_i1, idx_x_ff_data_i2)
        idx_x_ff_data = np.append(idx_x_ff_data_i, idx_x_ff_data_o)
        idx_x_ff = np.append(idx_x_bd1, idx_x_ff_data).astype('int32')

        #TC_star = saved_npz['TC']
        #XC_star = saved_npz['XC']
        #YC_star = saved_npz['YC']
        DC_star = saved_npz['DC']
        UC_star = saved_npz['UC']
        VC_star = saved_npz['VC']
        PC_star = saved_npz['PC']
        #N = XC_star.shape[0]
        #T = XC_star.shape[1]
    ######################################################################
    ######################## Training Data ###############################
    ######################################################################
        T = Ntime
        N = xydata.shape[0]
        label = np.zeros(N)
        label[idx_x_ff] = 3
        label[idx_x_sur] = 1
        #for i in range(1):
        #    label[idx_sur[i]] = i+2
        #label[idx_x_sur2] = 3
        #label[idx_x_sur3] = 4
        T_star = np.tile(t_star[:,None], (1,N))
        X_star = np.tile(xydata[:,0], (T,1))
        Y_star = np.tile(xydata[:,1], (T,1))
        d_data = DC_star.T.flatten()[:,None]
        u_data = UC_star.T.flatten()[:,None]
        v_data = VC_star.T.flatten()[:,None]
        p_data = PC_star.T.flatten()[:,None]
        t_data = T_star.flatten()[:,None]
        x_data = X_star.flatten()[:,None]
        y_data = Y_star.flatten()[:,None]
    
        # Training
        model = DLROM(xydata, t_data, x_data, y_data, l_data, 
                    d_data, u_data, v_data, p_data, 
                    layers, batch_size, Pec = 1000, Rey = 10)

        model.train(total_time = 40, learning_rate=1e-2)
    
        # Test Data
        t_rom_test = infer_times[:,None]
        T_test = np.tile(t_rom_test, (1,N))
    
        # Write the predictions
        for i in range(20):
            t_test = T_test.T[:,i:i+1]
            x_test = X_star.T[:,i:i+1]
            y_test = Y_star.T[:,i:i+1]
            
            # Prediction
            d_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)
            p3d_result = np.hstack((xydata[:,0][:,None], xydata[:,1][:,None], d_pred, u_pred, v_pred, p_pred))
            np.savetxt("./Case_flo8_RANS_NN_cuttail2_t="+str(i)+".dat", p3d_result, delimiter=" ", header="variables = X, Y, c, u, v, p \n zone i="+str(zone1_i-2*cuttail)+" j="+str(glayer)+" ", comments=' ')
            # Error
            error_d = relative_error(d_pred, DC_star[:,i][:,None])
            error_u = relative_error(u_pred, UC_star[:,i][:,None])
            error_v = relative_error(v_pred, VC_star[:,i][:,None])
            error_p = relative_error(p_pred, PC_star[:,i][:,None])
            print('Error d: %e, u: %e, v: %e, p: %e' % (error_d, error_u, error_v, error_p))
