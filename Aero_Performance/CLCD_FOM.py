import numpy as np
#import tensorflow as tf
import math
import time
import sys
import pandas as pd
import os
from scipy import interpolate
from scipy.optimize import curve_fit
from itertools import product
#import tensorflow.compat.v1 as tf
#from CFDFunctions import Gradient_Velocity_2D, \
#                      tf_session, mean_squared_error, relative_error

#tf.compat.v1.disable_eager_execution()

def predict_drag_lift(Ncon, Re, AOA, t_sur, x_sur, y_sur, x_sur2, y_sur2, u_sur, v_sur, psur_star, u_sur2, v_sur2):

	NRe = Re.shape[0]
	NAOA = AOA.shape[0]
	Re_inv = np.zeros(Ncon)
	AOA_CLCD = np.zeros(Ncon)
	NR = int(Ncon/NRe)
	for i in range(NRe):
		Re_inv[NR*i:NR*(i+1)] = 0 #1.0/Re[i] #(1.0/3.5e5)
	for i in range(NAOA):
		AOA_CLCD[i:Ncon:NAOA] = AOA[i]
	xm = 0.5*(x_sur[1:]+x_sur[0:-1]) # (Nsur-1) x 1
	ym = 0.5*(y_sur[1:]+y_sur[0:-1]) # (Nsur-1) x 1
	xm2 = 0.5*(x_sur2[1:]+x_sur2[0:-1])
	ym2 = 0.5*(y_sur2[1:]+y_sur2[0:-1])
	#print(AOA_CLCD)

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
	#U_x_star = np.reshape(u_x_sur, [Nsur,Tsur])
	#U_y_star = np.reshape(u_y_sur, [Nsur,Tsur])
	#V_x_star = np.reshape(v_x_sur, [Nsur,Tsur])
	#V_y_star = np.reshape(v_y_sur, [Nsur,Tsur])

	# gradients required for the lift and drag forces
	#v_y_star = v_y[:,y_sur]
	#v_y_star = np.append(v_y_star, np.gradient(V_star[:,i], y_sur)
	#v_x_star[v_x_star == -np.inf] = 0
	#[u_x_star,v_x_star,u_y_star,v_y_star] = sess.run([u_x,v_x,u_y,v_y], feed_dict={xur_tf:xsur_star, ysur_tf:ysur_star, usur_tf:usur_star, vsur_tf:vsur_star})
	P_star = np.reshape(psur_star, [Nsur,Tsur]) # CNsur x T
	P_star = P_star - np.mean(P_star, axis=0)
	#print(U_star, U_y_star)

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
	print(u_sur2)

	F_D = []
	F_L = []
	INT0 = ((-P_star[0:Nsur-1,:] + Re_inv*((u_sur2[0:Nsur-1,:]-u_sur[0:Nsur-1,:])*ny_star[0:Nsur-1,:]-(v_sur2[0:Nsur-1,:]-v_sur[0:Nsur-1,:])*nx_star[0:Nsur-1,:])/dn_star)*nx_star[0:Nsur-1,:])*ds_star
	INT1 = ((-P_star[1:Nsur,:] + Re_inv*((u_sur2[1:Nsur,:]-u_sur[1:Nsur,:])*ny_star[0:Nsur-1,:]-(v_sur2[1:Nsur,:]-v_sur[1:Nsur,:])*nx_star[0:Nsur-1,:])/dn_star)*nx_star[0:Nsur-1,:])*ds_star
	#INT0 = ((-P_star[0:Nsur-1,:] + 2*Re_inv*U_x_star[0:Nsur-1,:])*nx_star[0:Nsur-1,:] + Re_inv*(U_y_star[0:Nsur-1,:] + V_x_star[0:Nsur-1,:])*ny_star[0:Nsur-1,:])*ds_star
	#INT1 = ((-P_star[1:Nsur,:] + 2*Re_inv*U_x_star[1:Nsur,:])*nx_star[0:Nsur-1,:] + Re_inv*(U_y_star[1:Nsur,:] + V_x_star[1:Nsur,:])*ny_star[0:Nsur-1,:])*ds_star
	F_D = np.append(F_D, 0.5*np.sum(INT0 + INT1, axis = 0)) # F_D = Csur x T

	INT0 = ((-P_star[0:Nsur-1,:] + Re_inv*((u_sur2[0:Nsur-1,:]-u_sur[0:Nsur-1,:])*ny_star[0:Nsur-1,:]-(v_sur2[0:Nsur-1,:]-v_sur[0:Nsur-1,:])*nx_star[0:Nsur-1,:])/dn_star)*ny_star[0:Nsur-1,:])*ds_star
	INT1 = ((-P_star[1:Nsur,:] + Re_inv*((u_sur2[1:Nsur,:]-u_sur[1:Nsur,:])*ny_star[0:Nsur-1,:]-(v_sur2[1:Nsur,:]-v_sur[1:Nsur,:])*nx_star[0:Nsur-1,:])/dn_star)*ny_star[0:Nsur-1,:])*ds_star
	#INT0 = ((-P_star[0:Nsur-1,:] + 2*Re_inv*V_y_star[0:Nsur-1,:])*ny_star[0:Nsur-1,:] + Re_inv*(U_y_star[0:Nsur-1,:] + V_x_star[0:Nsur-1,:])*nx_star[0:Nsur-1,:])*ds_star
	#INT1 = ((-P_star[1:Nsur,:] + 2*Re_inv*V_y_star[1:Nsur,:])*ny_star[0:Nsur-1,:] + Re_inv*(U_y_star[1:Nsur,:] + V_x_star[1:Nsur,:])*nx_star[0:Nsur-1,:])*ds_star
	F_L = np.append(F_L, 0.5*np.sum(INT0 + INT1, axis = 0)) # F_L = Csur x T

	F_L = F_L*np.sin(AOA_CLCD*math.pi/180)+F_D*np.cos(AOA_CLCD*math.pi/180) #np.sqrt(F_D*F_D+F_L*F_L)*np.cos(AOA_CLCD*math.pi/180) 
	F_D = F_L*np.cos(AOA_CLCD*math.pi/180)-F_D*np.sin(AOA_CLCD*math.pi/180) #np.sqrt(F_D*F_D+F_L*F_L)*np.sin(AOA_CLCD*math.pi/180) 

	return F_L, F_D

if __name__ == "__main__":
    Re = np.array([1.0e5]) #, 2.0e5, 3.0e5]) #, 4.0e5, 5.0e5])
    Mach = np.array([0.2]) #, 0.45, 0.5, 0.55, 0.6])
    AOA = np.array([1.0]) #, 7.0, 8.0, 9.0, 10.0])
    noCol = 11 #6, 11  fix
    numd = 1 
    zone1_i = 401 
    zone1_j = 81
    glayer = 81 #zone1_j
    cuttail = 0
    dt = 0.1
    wt = 1000 
    d_inf = 1.225
    U_inf = 0.005*343
    #sim_data_path = "~/AIRFOIL/KFLOW_2017_04_AirDB/sol_core_1/Airfoil_5_5_5/" #fix
    sim_data_path = "F:\\JupyterNBook\\Airfoildata\\UIUC_ML\\CFDdata_UIUC\\"
    #sim_data_path = "./PINNresults/"
    res_data_path = "./"
    # create directory if not exist
    os.makedirs(os.path.dirname(res_data_path), exist_ok=True)
    # list of file names
    filenames = []
    Nfoil = 0 
    Ncon = 0
    for k in range(1,Re.shape[0]+1):
        for l in range(1,Mach.shape[0]+1):
            for m in range(1,AOA.shape[0]+1):
                filenames.append(sim_data_path+"result_"+str(k)+"_"+str(l)+"_"+str(m).rjust(2, '0')+"\\flo0001.dat")
                #filenames.append(sim_data_path+"Case_flo_con="+str(Ncon)+".dat")
                #filenames.append(sim_data_path+"Case_flo_unet_UIUC_n="+str(Ncon).rjust(2,'0')+".dat")
                Ncon += 1
    print(Ncon, filenames)
    ###
    snapshot_data = np.array([]) 
    POD = np.array([])
    pd_data1 = pd.DataFrame([])
    for i in range(Ncon):
        pd_data1 = pd.read_csv(filenames[i], na_filter=True, dtype='float64', delimiter=' ', skipinitialspace=True, skiprows=2, header=None)
        data = np.nan_to_num(pd_data1.values)
        single_data = data.flatten()
        single_data = single_data.reshape(-1,noCol)
        if i==0:
            snapshot_data = single_data
            xy = snapshot_data[:,0:2]
            N = snapshot_data.shape[0]
            #POD = array_data[:,[3,4,5,7]].flatten()[:,None]
        else:
            snapshot_data = np.vstack((snapshot_data, single_data))
            #snapshot_pod = array_data[:,[3,4,5,7]].flatten()[:,None]
            #POD = np.hstack((POD,snapshot_pod))
    array_data1 = snapshot_data
    shp = array_data1.shape
    print(shp)
    ###
    t_star = np.arange(Ncon)[:,None] # T(=1) x 1
    T = t_star.shape[0]
    xc_star = array_data1[:,0] # NT x 1
    yc_star = array_data1[:,1] # NT x 1
    NT = xc_star.shape[0]
    uc_star = array_data1[:,4] #fix 2 4
    vc_star = array_data1[:,5] #fix 3 5 
    pc_star = array_data1[:,7] #fix 4 7
    print(xc_star.shape) #"X","Y","rh","u","v","w","p","M","vorticity"

    UC = np.reshape(uc_star, (T,N)).T # N x T
    VC = np.reshape(vc_star, (T,N)).T # N x T
    PC = np.reshape(pc_star, (T,N)).T # N x T
    XC = np.reshape(xc_star, (T,N)).T # N x T
    YC = np.reshape(yc_star, (T,N)).T # N x T
    TC = np.tile(t_star, (1,N)).T # N x T
    tc_star = TC.flatten()
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
    idx_bottom = np.where(single_data[:,0] == single_data[1,0])[0]
    #print(idx_bottom)
    for i in range(0,idx_bottom[1]):
        if(single_data[i,1] != single_data[idx_bottom[1]-i,1]):
            break
    idx_tip = [i-1, idx_bottom[1]-i+1]
    #print(single_data[0,0], single_data[idx_bottom[1],0])
    idx_t_bd = np.array([0]) #np.concatenate([np.array([0]), np.random.choice(T-2, t_bd-2, replace=False)+1, np.array([T-1])]) #idx_t_data
    #T_b = idx_t_bd.shape[0]
    idx_x_sur = np.arange(idx_tip[0],idx_tip[1]+1)
    idx_x_sur2 = np.arange(idx_tip[0]+zone1_i-2*cuttail,idx_tip[1]+zone1_i-2*cuttail+1)
    idx_t_bd = np.sort(idx_t_bd.astype('int32'))
    idx_xc_bd = idx_x_sur #idx_x_bd
    t_sur = t_star[idx_t_bd]
    x_sur = single_data[:,0][idx_x_sur]
    y_sur = single_data[:,1][idx_x_sur]
    x_sur2 = single_data[:,0][idx_x_sur2]
    y_sur2 = single_data[:,1][idx_x_sur2]
    N_d = 3*zone1_i 
    idx_xc_diff = np.arange(0,N_d)
    #print(idx_x_sur)
    #t_sur = TC_star[idx_xc_bd,:][:, idx_t_bd].flatten()[:,None]
    #x_sur = XC_star[idx_xc_bd,:][:, idx_t_bd].flatten()[:,None]
    #y_sur = YC_star[idx_xc_bd,:][:, idx_t_bd].flatten()[:,None]
    u_sur = UC[idx_x_sur,:][:,idx_t_bd]#.flatten()[:,None]
    v_sur = VC[idx_x_sur,:][:,idx_t_bd]#.flatten()[:,None]
    p_sur = PC[idx_x_sur,:][:,idx_t_bd].flatten()[:,None]
    u_sur2 = UC[idx_x_sur2,:][:,idx_t_bd]#.flatten()[:,None]
    v_sur2 = VC[idx_x_sur2,:][:,idx_t_bd]#.flatten()[:,None]
    #np.savetxt(res_data_path+"psur.dat", PC[idx_xc_bd,:][:,3], delimiter=" ")
    np.savetxt(res_data_path+"u1u2.dat", np.hstack((u_sur.flatten()[:,None], u_sur2.flatten()[:,None])), delimiter=" ")
    
    F_L, F_D = predict_drag_lift(Ncon, Re, AOA, t_sur, x_sur, y_sur, x_sur2, y_sur2, u_sur, v_sur, p_sur, u_sur2, v_sur2)
    result = np.hstack((F_L[:,None], F_D[:,None]))
    np.savetxt(sim_data_path+"UIUC_CLCD.dat", result, delimiter=" ", header="variables = CL, CD")
    #np.savetxt(res_data_path+"FOM_CLCD.dat", result, delimiter=" ", header="variables = CL, CD")
