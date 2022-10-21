#"""
#@original author: Junghun Shin
#@modified by Suhun Cho
#"""
import numpy as np
import pandas as pd

noCol = 11   # number of flowfield variables
numd = 20    # number of snapshots
initial_time = 101 
inc_time = 4 # time interval of training data
zone1_i = 689 
zone1_j = 145
#glayer = 100 
#cuttail = 100 
dt = 0.1
wt = 1000 
d_inf = 1.225
U_inf = 0.005*343

sim_data_path = "../../../Database/Simdata_airfoil_unsteady/sol01_RANS3/"
res_data_path = "../Results/"
Tecplot_header_in = "variables=X, Y, Z, Rho, U, V, W, P, T, Vor, Qcri"
Tecplot_header_out = "variables=X, Y, Rho, U, V, P"

# list of file names
filenames = []
merged = []
Ntime = 0 
for i in range(0, numd):
	filenames.append(sim_data_path+"flo001.0000"+str(initial_time+i*inc_time).rjust(3,'0')+"uns") 
	Ntime += 1
print(Ntime, filenames)

###
snapshot_data = np.array([]) 
for i in range(0,numd):
	pd_data = pd.read_csv(sim_data_path+'flo001.0000'+str(initial_time+i*inc_time).rjust(3,'0')+'uns', dtype='float64', delimiter=' ', skipinitialspace=True, skiprows=2, header=None)
	#make it an np ndarray
	data = np.nan_to_num(pd_data.values)
	array_data = data.flatten()
	array_data = array_data.reshape(-1,noCol)
	if i==0:
		snapshot_data = array_data
		xy = snapshot_data[:,0:2]
		N = snapshot_data.shape[0]
		#POD = array_data[:,[3,4,5,7]].flatten()[:,None]s
	else:
		snapshot_data = np.vstack((snapshot_data, array_data))
		#snapshot_pod = array_data[:,[3,4,5,7]].flatten()[:,None]
		#POD = np.hstack((POD,snapshot_pod))
array_data1 = snapshot_data
shp = array_data1.shape
print(shp)
### Vectorize each variable
t_star = np.arange(Ntime)[:,None]*dt*inc_time # T(=1) x 1
T = t_star.shape[0]
#print(POD.shape)
xc_star = array_data1[:,0] # NT x 1
yc_star = array_data1[:,1] # NT x 1
NT = xc_star.shape[0]
dc_star = array_data1[:,3]
uc_star = array_data1[:,4]
vc_star = array_data1[:,5]
pc_star = array_data1[:,7]
print(xc_star.shape) #"X","Y","rh","u","v","w","p","M","vorticity"
#np.savez("../Results/array4(dataset_20timesteps_RANS_unsteady).npz", xy=xy, snapshot=POD)

DC = np.reshape(dc_star, (T,N)).T # N x T     
UC = np.reshape(uc_star, (T,N)).T # N x T
VC = np.reshape(vc_star, (T,N)).T # N x T
PC = np.reshape(pc_star, (T,N)).T # N x T
XC = np.reshape(xc_star, (T,N)).T # N x T
YC = np.reshape(yc_star, (T,N)).T # N x T
TC = np.tile(t_star, (1,N)).T # N x T
'''
#### Cut grid for memory efficiency 
idx_x_slice = np.array([])
for i in range(glayer):
    idx_x_slice = np.append(idx_x_slice, np.arange(cuttail+i*zone1_i, (zone1_i-cuttail)+i*zone1_i)).astype('int32')
    print(idx_x_slice.shape[0])
DC_star = DC[idx_x_slice,:]
UC_star = UC[idx_x_slice,:]
VC_star = VC[idx_x_slice,:]
PC_star = PC[idx_x_slice,:]
XC_star = XC[idx_x_slice,:]
YC_star = YC[idx_x_slice,:]
TC_star = TC[idx_x_slice,:]
'''
###
np.savez(res_data_path+'array4(dataset_20timesteps_RANS_unsteady).npz', TC=TC, XC=XC, YC=YC, DC=DC, UC=UC, VC=VC, PC=PC)
#np.savez('../Results/array4(dataset_20timesteps_RANS_unsteady).npz', TC=TC_star, XC=XC_star, YC=YC_star, DC=DC_star, UC=UC_star, VC=VC_star, PC=PC_star)
### check
#saved = np.load("./array.npz")
#print(saved['TC'])
#for i in range(Ntime):
#	p3d_result = np.hstack((x_test, y_test, D_pred[:,snap:snap+1], U_pred[:,snap:snap+1], V_pred[:,snap:snap+1], P_pred[:,snap:snap+1]))
#	np.savetxt("../Data/airfoil_unsteady/Eppler_Euler/Case_flo_(t="+str(i)+").dat", p3d_result, delimiter=" ", header="variables = X, Y, rho, u, v, p \n zone i="+str(zone1_i)+" j="+str(glayer)+" ", comments=' ')
