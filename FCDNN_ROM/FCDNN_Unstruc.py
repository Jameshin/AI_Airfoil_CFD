#"""
#@original author: Maziar Raissi
# edited by James Shin
#"""

import tensorflow.compat.v1 as tf
import numpy as np
#import scipy.io
import time
import sys
import os
#import pandas as pd

from CFDFunctions1 import neural_net, tf_session, mean_squared_error, relative_error

tf.compat.v1.disable_eager_execution()

class DNN_Unstruc(object):
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _data: input-output data
    
    def __init__(self, t_data, x_data, y_data, l_data,
                 d_data, u_data, v_data, p_data, 
                 layers, batch_size):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size

        # base space
        noConcernsVar = 4 
        N = x_data.shape[0]
        T = t_data.shape[0]
        # data
        [self.t_data, self.x_data, self.y_data, self.l_data, self.d_data, self.u_data, self.v_data, self.p_data] = [t_data, x_data, y_data, l_data, d_data, u_data, v_data, p_data]
        
        # placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.d_data_tf, self.u_data_tf, self.v_data_tf, self.p_data_tf] = [tf.placeholder(tf.float64, shape=[None, 1]) for _ in range(7)]

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
        self.global_step = tf.Variable(0, trainable = False, name='global_step')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
       
        self.sess, self.saver = tf_session()
    
    def train(self, total_time, learning_rate):
        
        N_data = self.x_data.shape[0]
        
        start_time = time.time()
        running_time = 0
        it = 0
        while running_time < total_time:
            
            idx_data = np.random.choice(N_data, min(self.batch_size, N_data))
            #idx_eqns = np.random.choice(N_eqns, min(self.batch_size, N_eqns))
            if it == 5000:
                learning_rate = 1e-3
            if it == 10000:
                learning_rate = 5e-4
            if it == 15000:
                learning_rate = 1e-4
            if it == 20000:
                learning_rate = 5e-5
            if it == 25000:
                learning_rate = 1e-5
            if it == 220000:
                learning_rate = 1e-5
            if it == 300000:
                learning_rate = 1e-6
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
        self.saver.save(self.sess, './model1/dnn.ckpt', global_step = self.global_step)
    
    def predict(self, t_star, x_star, y_star):
        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star}

        d_star = self.sess.run(self.d_data_pred, tf_dict)
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)

        return d_star, u_star, v_star, p_star 
    
if __name__ == "__main__":
    with tf.device('/gpu:0'):
        batch_size = 10134*5  #200000
        layers = [3] + 10*[4*10] + [4]  #[4] + 10*[4*10] + [4]
    
        # Load DatanoCol = 3
        numd = 21
        initial_time = 100 
        inc_time = 100 
        zone1_n = 5143 
        zone1_e = 10134
        dt = 1
        wt = 1000 
        d_inf = 1.225
        U_inf = 0.005*343
        data_path = "D:\\2D_uComp\\result_Ma0.4_AOA15\\"
        sim_data_path = "C:\\Users\\KISTI\\Documents\\Sim\\AI_Apps\\AI_Airfoil_CFD-main\\"
        Tecplot_header_in = "variables =x, y, rho, u, v, p, m"
        Tecplot_header_out = "variables =x, y, rho, u, v, p, m"
        n_xy = zone1_n*2
        n_field = zone1_n*2 + zone1_e*5
        n_elem = zone1_e*3
        # create directory if not exist
        #os.makedirs(os.path.dirname(res_data_path), exist_ok=True)
        # list of file names
        filenames = []
        Ntime = 0
        for i in range(0, numd):
            filenames.append(data_path+"mid_result_"+str(initial_time+i*inc_time).rjust(5,'0')+".rlt")
            Ntime += 1
        #print(Ntime, filenames)
        t_star = np.arange(initial_time, initial_time+numd*inc_time*2, inc_time*2)*dt # 1xT(=1)
        ###
        #perform coefficient interpolation here, using numpy for it
        total_steps = 20
        infer_times = np.arange(initial_time+inc_time, initial_time+inc_time+numd*inc_time*2, inc_time*2)*dt
        noConcernVar = 5

        saved_npz = np.load("./array_Unst_21.npz")
        TC_star = saved_npz['TN']
        XC_star = saved_npz['XN']
        YC_star = saved_npz['YN']
        XE_star = saved_npz['XE']
        YE_star = saved_npz['YE']
        DC_star = saved_npz['DE']
        UC_star = saved_npz['UE']
        VC_star = saved_npz['VE']
        PC_star = saved_npz['PE']
        EL_mat = saved_npz['EL']
        #N = XC_star.shape[0]
        #T = XC_star.shape[1]
    ######################################################################
    ######################## Training Data ###############################
    ######################################################################
        T = Ntime
        N = int(XE_star.shape[0])
        label = np.zeros(N)
        #label[idx_x_ff] = 2
        #label[idx_x_sur] = 1
        #label[idx_x_sur2] = 3
        #label[idx_x_sur3] = 4
        T_star = np.tile(t_star, (N,1))
        L_star = np.tile(label, (T,1))
        d_data = DC_star.T.flatten()[:,None]
        u_data = UC_star.T.flatten()[:,None]
        v_data = VC_star.T.flatten()[:,None]
        p_data = PC_star.T.flatten()[:,None]
        t_data = T_star.T.flatten()[:,None]
        x_data = XE_star.T.flatten()[:,None]
        y_data = YE_star.T.flatten()[:,None]
        l_data = L_star.flatten()[:,None]
        print(p_data.shape, t_data.shape, x_data.shape)
    
        #sys.stdout = open('stdout.txt', 'w')
        # Training
        model = DNN_Unstruc(t_data, x_data, y_data, l_data, 
                    d_data, u_data, v_data, p_data,  
                    layers, batch_size)

        model.train(total_time = 1, learning_rate=1e-2)
    
        # Test Data
        t_rom_test = infer_times
        T_test = np.tile(t_rom_test, (N,1))
    
        # Prediction
        #a_pred = model.predict(t_test, x_data, y_data)
        # Write the predictions
#"""
        np.savetxt("EL_mat.dat", EL_mat, fmt="%4i",delimiter=" ")
        for i in range(20):
            t_test = T_test[:,i:i+1]
            x_test = XE_star[:,i:i+1]
            y_test = YE_star[:,i:i+1]
            l_test = L_star[:,i:i+1]
            d_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)
            tecplot_result = np.vstack((XC_star[:,0][:,None], YC_star[:,0][:,None], d_pred, u_pred, v_pred, p_pred))
            filename = "./Case_flo_Unstr_t="+str(i).rjust(2,'0')+".dat"
            np.savetxt(filename, tecplot_result, delimiter=" ", header="variables = X, Y, d, u, v, p \n zone t= \"0.282832E-01\", n= "+str(zone1_n)+" e= "+str(zone1_e)+"\n ,varlocation=([3,4,5,6]=cellcentered),zonetype=fetriangle \n datapacking=block", comments=' ')
            with open(filename, 'a') as outfile:
                with open("EL_mat.dat") as file:
                    outfile.write(file.read())
            # Error
            error_d = relative_error(d_pred, DC_star[:,i][:,None])
            error_u = relative_error(u_pred, UC_star[:,i][:,None])
            error_v = relative_error(v_pred, VC_star[:,i][:,None])
            error_p = relative_error(p_pred, PC_star[:,i][:,None])
            print('Error d: %e, u: %e, v: %e, p: %e' % (error_d, error_u, error_v, error_p))
        #sys.stdout.close()
#"""  
