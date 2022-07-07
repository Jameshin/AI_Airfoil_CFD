import numpy as np
import time
import sys
import pandas as pd
import os
from scipy import interpolate
from itertools import product
#import tensorflow.compat.v1 as tf
#from CFDFunctions import Gradient_Velocity_2D, \
#                      tf_session, mean_squared_error, relative_error

#tf.compat.v1.disable_eager_execution()

def predict_drag_lift(t_sur, x_sur, y_sur, x_sur2, y_sur2, usur_star, vsur_star, psur_star, u_sur2, v_sur2):

	Re_inv = 0 #1.0/1e4
	xm = 0.5*(x_sur[1:]+x_sur[0:-1]) # (Nsur-1) x 1
	ym = 0.5*(y_sur[1:]+y_sur[0:-1]) # (Nsur-1) x 1
	xm2 = 0.5*(x_sur2[1:]+x_sur2[0:-1])
	ym2 = 0.5*(y_sur2[1:]+y_sur2[0:-1])
	print(t_sur.shape,x_sur.shape)

	Nsur = x_sur.shape[0]
	Tsur = t_sur.shape[0]

	Tsur_star = np.tile(t_sur, (1,Nsur)).T # Nsur x T
	Xsur_star = np.tile(x_sur, (1,Tsur)) # Nsur x T
	Ysur_star = np.tile(y_sur, (1,Tsur)) # Nsur x T
	tsur_star = np.reshape(Tsur_star,[-1,1]) # NsurT x 1
	xsur_star = np.reshape(Xsur_star,[-1,1]) # NsurT x 1
	ysur_star = np.reshape(Ysur_star,[-1,1]) # NsurT x 1
	
	#U_star = np.reshape(usur_star, [Nsur,Tsur]) # CNsur x T
	#V_star = np.reshape(vsur_star, [Nsur,Tsur]) # CNsur x T

	#tf_dict = {self.t_eqns_tf: tsur_star, self.x_eqns_tf: xsur_star, self.y_eqns_tf: ysur_star}
	#[p_star,
	# u_x_star,
	# u_y_star,
	# v_x_star,
	# v_y_star] = self.sess.run([self.p_eqns_pred,
	#                            self.u_x_eqns_pred,
	#                            self.u_y_eqns_pred,
	#                            self.v_x_eqns_pred,
	#                            self.v_y_eqns_pred], tf_dict)

	# gradients required for the lift and drag forces
	#[xsur_tf, ysur_tf] = [tf.placeholder(tf.float64, shape=[None, 1]) for _ in range(2)]
	'''u_x_star = []
	u_y_star = []
	v_x_star = []
	v_y_star = []
	for i in range(Tsur):
		u_x_star = np.append(u_x_star, np.gradient(U_star[:,i].flatten(), x_sur.flatten()))
		u_y_star = np.append(u_y_star, np.gradient(U_star[:,i].flatten(), y_sur.flatten()))
		v_x_star = np.append(v_x_star, np.gradient(V_star[:,i].flatten(), x_sur.flatten()))
		v_y_star = np.append(v_y_star, np.gradient(V_star[:,i].flatten(), y_sur.flatten()))
	#[u_x_star,v_x_star,u_y_star,v_y_star] = sess.run([u_x,v_x,u_y,v_y], feed_dict={xur_tf:xsur_star, ysur_tf:ysur_star, usur_tf:usur_star, vsur_tf:vsur_star})
    '''
	P_star = np.reshape(psur_star, [Nsur,Tsur]) # CNsur x T
	P_star = P_star - np.mean(P_star, axis=0)
	#U_x_star = np.reshape(u_x_star, [Nsur,Tsur]) # CNsur x T
	#U_y_star = np.reshape(u_y_star, [Nsur,Tsur]) # CNsur x T
	#V_x_star = np.reshape(v_x_star, [Nsur,Tsur]) # CNsur x T
	#V_y_star = np.reshape(v_y_star, [Nsur,Tsur]) # CNsur x T

	tck, u = interpolate.splprep([xm,ym],k=3,s=0)
	out = interpolate.splev(u,tck)
	der = interpolate.splev(u,tck, der=1)
	mag_der = np.sqrt(der[0]*der[0]+der[1]*der[1])
	[nx,ny] = [-der[1]/mag_der,der[0]/mag_der]
	ds = np.sqrt(np.square(x_sur[0:-1]-x_sur[1:])+np.square(y_sur[0:-1]-y_sur[1:]))
	nx_star = np.tile(nx, (Tsur,1)).T
	ny_star = np.tile(ny, (Tsur,1)).T
	ds_star = np.tile(ds, (Tsur,1)).T
	dn = np.sqrt(np.square(xm2-xm)+np.square(ym2-ym))
	dn_star = np.tile(dn, (Tsur,1)).T

	F_D = []
	F_L = []
	INT0 = ((-P_star[0:Nsur-1,:] + Re_inv*((u_sur2[0:Nsur-1,:]-u_sur[0:Nsur-1,:])*ny_star[0:Nsur-1,:]-(v_sur2[0:Nsur-1,:]-v_sur[0:Nsur-1,:])*nx_star[0:Nsur-1,:])/dn_star)*nx_star[0:Nsur-1,:])*ds_star
	INT1 = ((-P_star[1:Nsur,:] + Re_inv*((u_sur2[1:Nsur,:]-u_sur[1:Nsur,:])*ny_star[0:Nsur-1,:]-(v_sur2[1:Nsur,:]-v_sur[1:Nsur,:])*nx_star[0:Nsur-1,:])/dn_star)*nx_star[0:Nsur-1,:])*ds_star
	#INT0 = ((-P_star[0:Nsur-1,:] + 2*viscosity*U_x_star[0:Nsur-1,:])*nx_star[0:Nsur-1,:] + viscosity*(U_y_star[0:Nsur-1,:] + V_x_star[0:Nsur-1,:])*ny_star[0:Nsur-1,:])*ds_star
	#INT1 = ((-P_star[1:Nsur,:] + 2*viscosity*U_x_star[1:Nsur,:])*nx_star[0:Nsur-1,:] + viscosity*(U_y_star[1:Nsur,:] + V_x_star[1:Nsur,:])*ny_star[0:Nsur-1,:])*ds_star
	F_D = np.append(F_D, 0.5*np.sum(INT0 + INT1, axis = 0)) # F_D = Csur x T

	INT0 = ((-P_star[0:Nsur-1,:] + Re_inv*((u_sur2[0:Nsur-1,:]-u_sur[0:Nsur-1,:])*ny_star[0:Nsur-1,:]-(v_sur2[0:Nsur-1,:]-v_sur[0:Nsur-1,:])*nx_star[0:Nsur-1,:])/dn_star)*ny_star[0:Nsur-1,:])*ds_star
	INT1 = ((-P_star[1:Nsur,:] + Re_inv*((u_sur2[1:Nsur,:]-u_sur[1:Nsur,:])*ny_star[0:Nsur-1,:]-(v_sur2[1:Nsur,:]-v_sur[1:Nsur,:])*nx_star[0:Nsur-1,:])/dn_star)*ny_star[0:Nsur-1,:])*ds_star
	#INT0 = ((-P_star[0:Nsur-1,:] + 2*viscosity*V_y_star[0:Nsur-1,:])*ny_star[0:Nsur-1,:] + viscosity*(U_y_star[0:Nsur-1,:] + V_x_star[0:Nsur-1,:])*nx_star[0:Nsur-1,:])*ds_star
	#INT1 = ((-P_star[1:Nsur,:] + 2*viscosity*V_y_star[1:Nsur,:])*ny_star[0:Nsur-1,:] + viscosity*(U_y_star[1:Nsur,:] + V_x_star[1:Nsur,:])*nx_star[0:Nsur-1,:])*ds_star
	F_L = np.append(F_L, 0.5*np.sum(INT0 + INT1, axis = 0)) # F_L = Csur x T


	return F_L, F_D

if __name__ == "__main__":
    noCol = 5 #6, 11  fix
    numd = 20 
    initial_time = 0 #0, 101 fix 
    inc_time = 1 
    zone1_i = 689 
    zone1_j = 145
    glayer = 145
    cuttail = 0 
    dt = 0.1
    wt = 1000 
    d_inf = 1.225
    U_inf = 0.005*343
    #sim_data_path = "~/AIRFOIL/Unsteady/Eppler387/sol01_RANS3/" #fix
    sim_data_path = "./"
    Tecplot_header_in = "variables=X, Y, Z, Rho, U, V, W, P, T, Vor, Qcri"
    Tecplot_header_out = "variables=X, Y, Rho, U, V, P"
    # list of file names
    filenames = []
    merged = []
    Ntime = 0 
    for i in range(0, numd):
        filenames.append(sim_data_path+"Case_flo_R_unet_t="+str(initial_time+i*inc_time)+".dat") 
        #filenames.append(sim_data_path+"flo001.0000"+str(initial_time+i*inc_time).rjust(3,'0')+"uns")
        Ntime += 1
    print(Ntime, filenames)
    ###
    snapshot_data = np.array([]) 
    for i in range(0,numd):
        pd_data = pd.read_csv(sim_data_path+"Case_flo_R_unet_t="+str(initial_time+i*inc_time)+".dat", delimiter=' ', skipinitialspace=True, skiprows=2, header=None)
        #pd_data = pd.read_csv(sim_data_path+'flo001.0000'+str(initial_time+i*inc_time).rjust(3,'0')+'uns', dtype='float64', delimiter=' ', skipinitialspace=True, skiprows=2, header=None)
        #make it an np ndarray
        data = np.nan_to_num(pd_data.values)
        array_data = data.flatten()
        array_data = array_data.reshape(-1,noCol)
        if i==0:
            snapshot_data = array_data
            single_data = array_data[:]
            xy = single_data[:,0:2]
            N = single_data.shape[0]
        else:
            snapshot_data = np.vstack((snapshot_data, array_data))
            #snapshot_pod = array_data[:,[2,3,4,5]].flatten()[:,None] 
    array_data1 = snapshot_data
    shp = array_data1.shape
    print(shp)
    ###
    t_star = np.arange(Ntime)[:,None]*dt # T(=1) x 1
    T = t_star.shape[0]
    xc_star = array_data1[:,0] # NT x 1
    yc_star = array_data1[:,1] # NT x 1
    NT = xc_star.shape[0]
    uc_star = array_data1[:,2] #fix 3 4
    vc_star = array_data1[:,3] #fix 4 5
    pc_star = array_data1[:,4] #fix 5 7
    print(xc_star.shape) #"X","Y","rh","u","v","w","p","M","vorticity"

    UC = np.reshape(uc_star, (T,N)).T # N x T
    VC = np.reshape(vc_star, (T,N)).T # N x T
    PC = np.reshape(pc_star, (T,N)).T # N x T
    XC = np.reshape(xc_star, (T,N)).T # N x T
    YC = np.reshape(yc_star, (T,N)).T # N x T
    TC = np.tile(t_star, (1,N)).T # N x T
    ###
    #idx_x_slice = np.array([])
    #for i in range(glayer):
    #    idx_x_slice = np.append(idx_x_slice, np.arange(cuttail+i*zone1_i, (zone1_i-cuttail)+i*zone1_i)).astype('int32')
    #    print(idx_x_slice.shape[0])
    #UC_star = UC[idx_x_slice,:]
    #VC_star = VC[idx_x_slice,:]
    #PC_star = PC[idx_x_slice,:]
    #XC_star = XC[idx_x_slice,:]
    #YC_star = YC[idx_x_slice,:]
    #TC_star = TC[idx_x_slice,:]
    ###
    # Surface Data for CL/CD calculation
    #idx_tip = np.where(single_data[:,0] == 1.0)[0][0:2]
    t_bd = T
    idx_bottom = np.where(single_data[:,0] == single_data[0,0])[0]
    print(idx_bottom)
    for i in range(idx_bottom[0], idx_bottom[1]):
        if(single_data[i,1] != single_data[idx_bottom[1]-i,1]):
            break
    idx_tip = [i-1, idx_bottom[1]-i+1]
    #print(single_data[0,0], single_data[idx_bottom[1],0])
    idx_t_bd = np.concatenate([np.array([0]), np.random.choice(T-2, t_bd-2, replace=False)+1, np.array([T-1])]) #idx_t_data
    T_b = idx_t_bd.shape[0]
    #idx_C_bd = idx_C_data
    print(idx_tip[0],idx_tip[1])
    idx_x_sur = np.arange(idx_tip[0],idx_tip[1]+1)
    idx_x_sur2 = np.arange(idx_tip[0]+zone1_i,idx_tip[1]+zone1_i+1)
    idx_t_bd = np.sort(idx_t_bd.astype('int32'))
    idx_xc_bd = idx_x_sur #idx_x_bd
    t_sur = t_star[idx_t_bd]
    x_sur = single_data[:,0][idx_x_sur]
    y_sur = single_data[:,1][idx_x_sur]
    x_sur2 = single_data[:,0][idx_x_sur2]
    y_sur2 = single_data[:,1][idx_x_sur2]
    print(t_sur,x_sur)
    #t_sur = TC_star[idx_xc_bd,:][:, idx_t_bd].flatten()[:,None]
    #x_sur = XC_star[idx_xc_bd,:][:, idx_t_bd].flatten()[:,None]
    #y_sur = YC_star[idx_xc_bd,:][:, idx_t_bd].flatten()[:,None]
    u_sur = UC[idx_xc_bd,:][:, idx_t_bd]#.flatten()[:,None]
    v_sur = VC[idx_xc_bd,:][:, idx_t_bd]#.flatten()[:,None]
    p_sur= PC[idx_xc_bd,:][:, idx_t_bd].flatten()[:,None]
    u_sur2 = UC[idx_x_sur2,:][:,idx_t_bd]#.flatten()[:,None]
    v_sur2 = VC[idx_x_sur2,:][:,idx_t_bd]

    F_L, F_D = predict_drag_lift(t_sur, x_sur, y_sur, x_sur2, y_sur2, u_sur, v_sur, p_sur, u_sur2, v_sur2)
    result = np.hstack((F_L[:,None], F_D[:,None]))
    np.savetxt(sim_data_path+"PINN_CLCD_Unet_1.dat", result, delimiter=" ", header="variables = CL, CD")
    #np.savetxt("./FOM_CLCD_RANS3_2.dat", result, delimiter=" ", header="variables = CL, CD")
