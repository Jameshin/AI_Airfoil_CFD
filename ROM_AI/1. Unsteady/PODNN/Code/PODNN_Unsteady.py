#"""
#@original author: Maziar Raissi
#@edited by James Shin, Suhun Cho
#"""
import os
import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io
import time
import sys
import pandas as pd
import pickle
import RomObject

from CFDFunctions3 import neural_net, tf_session, mean_squared_error, relative_error

#tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()

class PODNN_ROM(object):
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _data: input-output data
    
    def __init__(self, l_pod_data, t_pod_data, t_data, x_data, y_data, 
                 d_data, u_data, v_data, p_data, phi, mean_data, a0_data, 
                 a1_data, a2_data, a3_data, a4_data, a5_data, a6_data, a7_data, 
                 a8_data, a9_data, a10_data,
                 a11_data, a12_data, a13_data, a14_data, a15_data, a16_data, a17_data,
                 a18_data, a19_data, layers, batch_size):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size

        # base space
        self.phi = phi
        self.mean_data = mean_data
        noConcernsVar = 4 
        N = x_data.shape[0]
        T = t_pod_data.shape[0]
        # data
        [self.l_pod_data, self.t_pod_data, self.d_data, self.u_data, self.v_data, self.p_data, self.a0_data, self.a1_data, self.a2_data, self.a3_data, self.a4_data, self.a5_data, self.a6_data, self.a7_data, self.a8_data, self.a9_data, self.a10_data, self.a11_data, self.a12_data, self.a13_data, self.a14_data, self.a15_data, self.a16_data, self.a17_data, self.a18_data, self.a19_data] = [l_pod_data, t_pod_data, d_data, u_data, v_data, p_data, a0_data, a1_data, a2_data, a3_data, a4_data, a5_data, a6_data, a7_data, a8_data, a9_data, a10_data, a11_data, a12_data, a13_data, a14_data, a15_data, a16_data, a17_data, a18_data, a19_data]
        [self.t_data, self.x_data, self.y_data] = [t_data, x_data, y_data]
        self.a_data = tf.concat([self.a0_data, self.a1_data,
                                    self.a2_data,
                                    self.a3_data, self.a4_data,
                                    self.a5_data, self.a6_data,
                                    self.a7_data, self.a8_data,
                                    self.a9_data, self.a10_data,
                                    self.a11_data, self.a12_data,
                                    self.a13_data, self.a14_data,
                                    self.a15_data, self.a16_data,
                                    self.a17_data, self.a18_data,
                                    self.a19_data], axis=1) 
        # placeholders
        [self.l_pod_data_tf, self.t_pod_data_tf, self.d_data_tf, self.u_data_tf, self.v_data_tf, self.p_data_tf, self.a0_data_tf, self.a1_data_tf, self.a2_data_tf, self.a3_data_tf, self.a4_data_tf, self.a5_data_tf, self.a6_data_tf, self.a7_data_tf, self.a8_data_tf, self.a9_data_tf, self.a10_data_tf, self.a11_data_tf, self.a12_data_tf, self.a13_data_tf, self.a14_data_tf, self.a15_data_tf, self.a16_data_tf, self.a17_data_tf, self.a18_data_tf, self.a19_data_tf] = [tf.placeholder(tf.float64, shape=[None, 1]) for _ in range(26)]
        [self.t_data_tf, self.x_data_tf, self.y_data_tf] = [tf.placeholder(tf.float64, shape=[None, 1]) for _ in range(3)]
        
        # neural networks
        self.net_pod= neural_net(self.l_pod_data, self.t_pod_data, layers = self.layers)
        
        [self.a0_data_pred, self.a1_data_pred, self.a2_data_pred, 
         self.a3_data_pred, self.a4_data_pred, self.a5_data_pred, 
         self.a6_data_pred, self.a7_data_pred, self.a8_data_pred, 
         self.a9_data_pred, self.a10_data_pred,
         self.a11_data_pred, self.a12_data_pred,
         self.a13_data_pred, self.a14_data_pred,
         self.a15_data_pred, self.a16_data_pred,
         self.a17_data_pred, self.a18_data_pred, 
         self.a19_data_pred] = self.net_pod(self.l_pod_data_tf, self.t_pod_data_tf)
                
        self.a_data_pred = tf.concat([self.a0_data_pred, self.a1_data_pred, 
                                     self.a2_data_pred,
                                     self.a3_data_pred, self.a4_data_pred,
                                     self.a5_data_pred, self.a6_data_pred,
                                     self.a7_data_pred, self.a8_data_pred,
                                     self.a9_data_pred, self.a10_data_pred,
                                     self.a11_data_pred, self.a12_data_pred,
                                     self.a13_data_pred, self.a14_data_pred,
                                     self.a15_data_pred, self.a16_data_pred,
                                     self.a17_data_pred, self.a18_data_pred,
                                     self.a19_data_pred], axis=1)

        self.a_data_tf = tf.concat([self.a0_data_tf, self.a1_data_tf,
                                    self.a2_data_tf,
                                    self.a3_data_tf, self.a4_data_tf,
                                    self.a5_data_tf, self.a6_data_tf,
                                    self.a7_data_tf, self.a8_data_tf,
                                    self.a9_data_tf, self.a10_data_tf,
                                    self.a11_data_tf, self.a12_data_tf,
                                    self.a13_data_tf, self.a14_data_tf,
                                    self.a15_data_tf, self.a16_data_tf,
                                    self.a17_data_tf, self.a18_data_tf,
                                    self.a19_data_tf], axis=1) 
         
        # loss
        self.loss = mean_squared_error(self.a_data_pred, self.a_data_tf) 
        
        # optimizers
        self.global_step = tf. Variable(0, trainable = False, name='global_step')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        
        self.sess, self.saver = tf_session()
    
    def train(self, total_time, learning_rate):
        start_time = time.time()
        running_time = 0
        it = 0
        while running_time < total_time:          
            #if it>8000:
            #    learning_rate = 5e-4
            if it>15000:
                learning_rate = 1e-4
            #if it>20000:
            #    learning_rate = 5e-5
            #if it>20000:
            #    learning_rate = 5e-5            
            #if it>50000:
            #    learning_rate = 1e-5
            
            (t_pod_data_batch, a0_data_batch, 
             a1_data_batch, a2_data_batch, a3_data_batch, 
             a4_data_batch, a5_data_batch, a6_data_batch,
             a7_data_batch, a8_data_batch, a9_data_batch,
             a10_data_batch, a11_data_batch, a12_data_batch,
             a13_data_batch, a14_data_batch, a15_data_batch,
             a16_data_batch, a17_data_batch, a18_data_batch,
             a19_data_batch) = (self.t_pod_data,
                                self.a0_data, self.a1_data, 
                                self.a2_data, 
                                self.a3_data, self.a4_data,
                                self.a5_data, self.a6_data,
                                self.a7_data, self.a8_data,
                                self.a9_data, self.a10_data,
                                self.a11_data, self.a12_data,
                                self.a13_data, self.a14_data,
                                self.a15_data, self.a16_data,
                                self.a17_data, self.a18_data,
                                self.a19_data)


            tf_dict = {self.t_pod_data_tf: t_pod_data_batch,
                       self.a0_data_tf: a0_data_batch, 
                       self.a1_data_tf: a1_data_batch, self.a2_data_tf: a2_data_batch,
                       self.a3_data_tf: a3_data_batch, self.a4_data_tf: a4_data_batch,
                       self.a5_data_tf: a5_data_batch, self.a6_data_tf: a6_data_batch,
                       self.a7_data_tf: a7_data_batch, self.a8_data_tf: a8_data_batch,
                       self.a9_data_tf: a9_data_batch, self.a10_data_tf: a10_data_batch,
                       self.a11_data_tf: a11_data_batch, self.a12_data_tf: a12_data_batch,
                       self.a13_data_tf: a13_data_batch, self.a14_data_tf: a14_data_batch,
                       self.a15_data_tf: a15_data_batch, self.a16_data_tf: a16_data_batch,
                       self.a17_data_tf: a17_data_batch, self.a18_data_tf: a18_data_batch,
                       self.a19_data_tf: a19_data_batch, 
                       self.l_pod_data_tf: self.l_pod_data, self.t_data_tf:self.t_data,
                       self.x_data_tf:self.x_data, self.y_data_tf:self.y_data, 
                       self.u_data_tf: self.u_data, self.v_data_tf: self.v_data,
                       self.d_data_tf: self.d_data, self.p_data_tf: self.p_data,
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
        self.saver.save(self.sess, '../Results/dnn.ckpt', global_step = self.global_step)
    
    def predict(self, l_pod_data, t_pod_dat):
        t_pod_data = np.tile(t_pod_dat, [1,1])  #t_pod_data = np.tile(t_pod_dat.T, [T,1]) orig code before debug
        tf_dict = {self.l_pod_data_tf: l_pod_data,
                       self.t_pod_data_tf: t_pod_data}
        
        a0_star = self.sess.run(self.a0_data_pred, tf_dict)
        a1_star = self.sess.run(self.a1_data_pred, tf_dict)
        a2_star = self.sess.run(self.a2_data_pred, tf_dict)
        a3_star = self.sess.run(self.a3_data_pred, tf_dict)
        a4_star = self.sess.run(self.a4_data_pred, tf_dict)
        a5_star = self.sess.run(self.a5_data_pred, tf_dict)
        a6_star = self.sess.run(self.a6_data_pred, tf_dict)
        a7_star = self.sess.run(self.a7_data_pred, tf_dict)
        a8_star = self.sess.run(self.a8_data_pred, tf_dict)
        a9_star = self.sess.run(self.a9_data_pred, tf_dict)
        a10_star = self.sess.run(self.a10_data_pred, tf_dict)
        a11_star = self.sess.run(self.a11_data_pred, tf_dict)
        a12_star = self.sess.run(self.a12_data_pred, tf_dict)
        a13_star = self.sess.run(self.a13_data_pred, tf_dict)
        a14_star = self.sess.run(self.a14_data_pred, tf_dict)
        a15_star = self.sess.run(self.a15_data_pred, tf_dict)
        a16_star = self.sess.run(self.a16_data_pred, tf_dict)
        a17_star = self.sess.run(self.a17_data_pred, tf_dict)
        a18_star = self.sess.run(self.a18_data_pred, tf_dict)
        a19_star = self.sess.run(self.a19_data_pred, tf_dict)
        a_star = np.hstack((a0_star, a1_star,  a2_star, a3_star, a4_star, a5_star,
                       a6_star,  a7_star, a8_star, a9_star, a10_star,
                       a11_star,  a12_star, a13_star, a14_star, a15_star,
                       a16_star,  a17_star, a18_star,
                       a19_star))
        
        return a_star
    
    
if __name__ == "__main__":
    with tf.device('/gpu:0'):
        batch_size = 10000 
        layers = [2] + 10*[2*20] + [20] #10*[4*10]
    
        # Load Data

        sim_data_path = "../../../../Database/Simdata_airfoil_unsteady/sol01_RANS3/"
        res_data_path = "../Results/"

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
        for i in range(0, numd):
            filenames.append(sim_data_path+"flo001.0000"+str(initial_time+i*inc_time).rjust(3,'0')+"uns")
            Ntime += 1
        #print(Ntime, filenames)
        t_star = np.arange(initial_time, initial_time+numd*inc_time, inc_time)*dt # 1xT(=1)
        ###
        #perform coefficient interpolation here, using numpy for it
        total_steps = 20
        #infer_times = np.arange(initial_time+2, initial_time+numd*inc_time+3, inc_time)*dt
        infer_times = np.arange(initial_time+2, initial_time+numd*inc_time+2, inc_time)*dt
        noConcernVar = 4
        zone1_i = 689
        zone1_j = 145

        #READ IN THE POD DESIGN INFORMATION
        with open(res_data_path+'times4.pkl', 'rb') as input:
            read_times = pickle.load(input)[0]
        read_times = np.array(read_times)*dt
        #read in saved rom object
        with open(res_data_path+'rom-object4.pkl', 'rb') as input:
            read_rom = pickle.load(input)
        ###
        #read xy-coordinates
        pd_data = pd.read_csv(res_data_path+'xy.csv', dtype='float64', delimiter=' ', header=None, skipinitialspace=True)
        xydata = pd_data.values

        coeffs = np.array(read_rom.coeffsmat)
        phi = np.array(read_rom.umat)
        mean_data = np.array(read_rom.mean_data[:,None])
        saved_npz = np.load(res_data_path+"array4(dataset_20timesteps_RANS_unsteady).npz")
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
        l_pod_data = np.tile(np.arange(Ntime)[:,None], (1,1)) 
        t_pod_data = t_star[:,None]
        T_star = np.tile(t_star, (1,N))
        X_star = np.tile(xydata[:,0], T)
        Y_star = np.tile(xydata[:,1], T)
        A_star = coeffs.T
        d_data = DC_star.T.flatten()[:,None]
        u_data = UC_star.T.flatten()[:,None]
        v_data = VC_star.T.flatten()[:,None]
        p_data = PC_star.T.flatten()[:,None]
        t_data = T_star.flatten()[:,None]
        x_data = X_star.flatten()[:,None]
        y_data = Y_star.flatten()[:,None]
        a0_data = A_star[:, 0][:,None]
        a1_data = A_star[:, 1][:,None]
        a2_data = A_star[:, 2][:,None]
        a3_data = A_star[:, 3][:,None]
        a4_data = A_star[:, 4][:,None]
        a5_data = A_star[:, 5][:,None]
        a6_data = A_star[:, 6][:,None]
        a7_data = A_star[:, 7][:,None]
        a8_data = A_star[:, 8][:,None]
        a9_data = A_star[:, 9][:,None]
        a10_data = A_star[:, 10][:,None]
        a11_data = A_star[:, 11][:,None]
        a12_data = A_star[:, 12][:,None]
        a13_data = A_star[:, 13][:,None]
        a14_data = A_star[:, 14][:,None]
        a15_data = A_star[:, 15][:,None]
        a16_data = A_star[:, 16][:,None]
        a17_data = A_star[:, 17][:,None]
        a18_data = A_star[:, 18][:,None]
        a19_data = A_star[:, 19][:,None]
 
        # Training
        model = PODNN_ROM(l_pod_data, t_pod_data, t_data, x_data, y_data, 
                    d_data, u_data, v_data, p_data, phi, mean_data, 
                    a0_data, a1_data, a2_data, a3_data, a4_data, a5_data,
                    a6_data, a7_data, a8_data, a9_data, a10_data, a11_data,
                    a12_data, a13_data, a14_data, a15_data, a16_data, a17_data,
                    a18_data, a19_data, layers, batch_size)

        model.train(total_time = 0.03, learning_rate=1e-3)
    
        # Test Data
        t_test = infer_times[:,None]
        l_pod_test = l_pod_data
        # Prediction
        print(l_pod_test.shape, t_test.shape, infer_times.shape) # dimension test for debugging
        a_pred = model.predict(l_pod_test, t_test)
        #np.savez("./ROM_DNN.npz", a_pred=a_pred, phi=phi, )
        print(a_pred)   
        U_pod = np.add(np.transpose(np.matmul(np.transpose(a_pred), np.transpose(phi))),
                         np.tile(mean_data, (1,Ntime)))
        for i in range(Ntime):
            U_pred = U_pod[:,i].reshape(-1, noConcernVar)
            d_pred = U_pred[:,0][:,None]
            u_pred = U_pred[:,1][:,None]
            v_pred = U_pred[:,2][:,None]
            p_pred = U_pred[:,3][:,None]
            p3d_result = np.hstack((xydata[:,0][:,None], xydata[:,1][:,None], d_pred, u_pred, v_pred, p_pred))
            np.savetxt(res_data_path+"Case_flo_R_PODDNN_t="+str(initial_time+i*inc_time+2)+".dat", p3d_result, delimiter=" ", header="variables = X, Y, c, u, v, p \n zone i="+str(zone1_i)+" j="+str(zone1_j)+" ", comments=' ')
            # Error
            error_d = relative_error(d_pred, DC_star[:,i][:,None])
            error_u = relative_error(u_pred, UC_star[:,i][:,None])
            error_v = relative_error(v_pred, VC_star[:,i][:,None])
            error_p = relative_error(p_pred, PC_star[:,i][:,None])
            print('Error d: %e, u: %e, v: %e, p: %e' % (error_d, error_u, error_v, error_p))
