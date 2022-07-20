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
import pickle
import RomObject

from CFDFunctions3 import neural_net, Euler_uIncomp_2D, Gradient_Velocity_2D, \
                      tf_session, mean_squared_error, relative_error

tf.compat.v1.disable_eager_execution()

class PODNN(object):
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _inlet: input-output data at the inlet
    # _star: preditions
    
    def __init__(self, l_pod_data, t_pod_data, t_data, x_data, y_data, 
                 d_data, u_data, v_data, p_data, phi, mean_data,  
                 A_star, layers, batch_size):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size
        
        # flow properties
        self.Pec = Pec
        self.Rey = Rey

        # base space
        self.phi = phi
        self.mean_data = mean_data
        noConcernVar = 4 
        N = x_data.shape[0]
        T = np.sqrt(t_pod_data.shape[0]).astype(np.int32)

        print(A_star.shape, phi.shape, mean_data.shape, t_pod_data.shape, l_pod_data.shape)
        # data
        [self.l_pod_data, self.t_pod_data, self.d_data, self.u_data, self.v_data, self.p_data, self.A_star] = [l_pod_data, t_pod_data, d_data, u_data, v_data, p_data, A_star]
        [self.t_data, self.x_data, self.y_data] = [t_data, x_data, y_data]
        # placeholders
        [self.l_pod_data_tf, self.t_pod_data_tf, self.d_data_tf, self.u_data_tf, self.v_data_tf, self.p_data_tf, self.A_star_tf, self.a_star_tf] = [tf.placeholder(tf.float64, shape=[None, 1]) for _ in range(8)]
        [self.t_data_tf, self.x_data_tf, self.y_data_tf] = [tf.placeholder(tf.float64, shape=[None, 1]) for _ in range(3)]

        # physics "uninformed" neural networks
        self.net_pod= neural_net(self.l_pod_data, self.t_pod_data, layers = self.layers) # 
        #self.net_duvp = neural_net(self.t_data, self.x_data, self.y_data, layers = [3,12,12,12,3])
        
        
        self.A_star_pred = self.net_pod(self.l_pod_data_tf, self.t_pod_data_tf) # 
                
        self.a_data_pred = tf.reshape(self.A_star_pred, [T,T])        
         
        U_pod_pred = tf.add(tf.transpose(tf.matmul(self.a_data_pred, tf.transpose(tf.constant(phi, tf.float64)))), tf.tile(tf.constant(mean_data, tf.float64), tf.constant([1,T], tf.int32)))
        for i in range(T):
            temp = tf.reshape(U_pod_pred[:,i], [-1, noConcernVar])
            if i == 0:
                U_pred = temp
            else:
                U_pred = tf.concat([U_pred, temp], axis=0)
        self.d_data_pred = U_pred[:,0][:,None]
        self.u_data_pred = U_pred[:,1][:,None]
        self.v_data_pred = U_pred[:,2][:,None]
        self.p_data_pred = U_pred[:,3][:,None]
        
        
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
        
        N_data = self.t_pod_data.shape[0]
        #N_eqns = self.t_pod_eqns.shape[0]
        
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
            
            tf_dict = {self.l_pod_data_tf: self.l_pod_data, 
                       self.t_pod_data_tf: self.t_pod_data,
                       self.A_star_tf: self.A_star, 
                       self.t_pod_eqns_tf: t_pod_eqns, self.t_data_tf: self.t_data,
                       self.x_data_tf: self.x_data, self.y_data_tf: self.y_data, 
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
        self.saver.save(self.sess, './model/dnn.ckpt', global_step = self.global_step) 

    def predict(self, l_pod_data, t_star):
        
        tf_dict = {self.t_pod_data_tf: t_star, self.l_pod_data_tf: l_pod_data}
        
        A_star = self.sess.run(self.A_star_pred, tf_dict)
        a_pred = np.reshape(A_star, [T,T]).T

        return a_pred
    
    
if __name__ == "__main__":
    with tf.device('/gpu:0'):
        batch_size = 10000 
        layers = [2] + 5*[10*40] + [1]
    
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
        sim_data_path = "D:\\JupyterNBook\\HFM-master\\Source\\pinn_POD\\2D_uComp\\result_Ma0.4_AOA15\\"
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
            filenames.append(sim_data_path+"mid_result_"+str(initial_time+i*inc_time*2).rjust(5,'0')+".rlt")
            Ntime += 1
        #print(Ntime, filenames)
        t_star = np.arange(initial_time, initial_time+numd*inc_time*2, inc_time*2)*dt # 1xT(=1)
        ###
        #perform coefficient interpolation here, using numpy for it
        total_steps = 50
        #input_design = [251 + x for x in range(total_steps)]
        input_times = np.arange(initial_time+inc_time, initial_time+inc_time+numd*inc_time*2, inc_time*2)*dt
        noConcernVar = 4

        #READ IN THE POD DESIGN INFORMATION
        with open('./designs.pkl', 'rb') as input:
            read_times = pickle.load(input)[0]
        read_times = np.array(read_times)*dt
        #read in saved rom object
        with open('./rom-object.pkl', 'rb') as input:
            read_rom = pickle.load(input)
        ###
        #read xy-coordinates
        pd_data = pd.read_csv('./xy.csv', dtype='float64', delimiter=' ', header=None, skipinitialspace=True)
        xydata = pd_data.values

        coeffs = np.array(read_rom.coeffsmat)
        phi = np.array(read_rom.umat)
        mean_data = np.array(read_rom.mean_data[:,None])
        #mean_tensor = tf.constant(mean_data, name="mean_data_tensor")
        #U_pod = np.add(np.transpose(np.matmul(coeffs, np.transpose(phi))), 
        #                np.tile(mean_data, (1,Ntime)))
        #for i in range(Ntime):
        #    temp = U_pod[:,i].reshape(-1, noConcernVar)
        #    if i ==0:
        #        U = temp
        #    else:
        #        U = np.vstack((U, temp))
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
    ######################################################################
    ######################## Training Data ###############################
    ######################################################################
        T = Ntime
        N = int(XE_star.shape[0])
        l_pod_data = np.tile(np.arange(Ntime)[:,None], (T,1))
        t_pod_data = np.repeat(t_star[:,None], T, axis=0)
        #print(t_pod_data.shape,l_pod_data.shape, '########')
        T_star = np.tile(t_star, (N,1))
        #X_star = np.tile(xydata[:,0], T)
        #Y_star = np.tile(xydata[:,1], T)
        #L_star = np.tile(label, (T,1))
        A_star = coeffs.flatten()[:,None]
        d_data = DC_star.T.flatten()[:,None]
        u_data = UC_star.T.flatten()[:,None]
        v_data = VC_star.T.flatten()[:,None]
        p_data = PC_star.T.flatten()[:,None]
        t_data = T_star.T.flatten()[:,None]
        x_data = XE_star.T.flatten()[:,None]
        y_data = YE_star.T.flatten()[:,None]
        #l_data = L_star.flatten()[:,None]
        t_pod_eqns = np.repeat(input_times[:,None], T, axis=0)
        #T_eqns = input_times.shape[0]
        #N_eqns = N 
        print(t_pod_data.shape,t_pod_data.shape, A_star.shape, d_data.shape, u_data.shape, p_data.shape, t_data.shape, x_data.shape, '########')
        #t_eqns = np.tile(t_pod_eqns, (1,N_eqns)).flatten()[:,None]
        #xy_eqns = np.tile(xydata, (T_eqns,1))
        #x_eqns = xy_eqns[:,0][:,None]
        #y_eqns = xy_eqns[:,1][:,None]
    
        #sys.stdout = open('stdout_PODDNN.txt', 'w')
        # Training
        model = PODNN(l_pod_data, t_pod_data, t_pod_eqns, t_data, x_data, y_data, 
                    d_data, u_data, v_data, p_data, phi, mean_data, 
                    A_star, layers, batch_size)
    
        #model.train(total_time = 1, learning_rate=5e-3)
    
        # Test Data
        t_test = np.repeat(input_times[:,None], T, axis=0)
    
        # Prediction
        a_pred = model.predict(l_pod_data, t_test)
        #np.savez("./ROM_DNN.npz", a_pred=a_pred, phi=phi, )
        #print(a_pred)   
        U_pod = np.add(np.transpose(np.matmul(np.transpose(a_pred), np.transpose(phi))), np.tile(mean_data, (1,Ntime)))
        for i in range(Ntime):
            U_pred = U_pod[:,i].reshape(-1, noConcernVar)
            d_pred = U_pred[:,0][:,None]
            u_pred = U_pred[:,1][:,None]
            v_pred = U_pred[:,2][:,None]
            p_pred = U_pred[:,3][:,None]
            tecplot_result = np.vstack((XC_star[:,0][:,None], YC_star[:,0][:,None], d_pred, u_pred, v_pred, p_pred))
            filename = "D:\\JupyterNBook\\PINN_Unstruc\\PODNNresults\\Case_flo_Unstr_t="+str(i).rjust(2,'0')+".dat"
            np.savetxt(filename, tecplot_result, delimiter=" ", header="variables = X, Y, d, u, v, p \n zone t= \"0.282832E-01\", n= "+str(zone1_n)+" e= "+str(zone1_e)+"\n ,varlocation=([3,4,5,6]=cellcentered),zonetype=fetriangle \n datapacking=block", comments=' ')
            with open(filename, 'a') as outfile:
                with open("EL_mat.dat") as file:
                    outfile.write(file.read())
            # Rel Error
            error_d = relative_error(d_pred, DC_star[:,i][:,None])
            error_u = relative_error(u_pred, UC_star[:,i][:,None])
            error_v = relative_error(v_pred, VC_star[:,i][:,None])
            error_p = relative_error(p_pred, PC_star[:,i][:,None])
            print('Error d: %e, u: %e, v: %e, p: %e' % (error_d, error_u, error_v, error_p))
        #sys.stdout.close()    
