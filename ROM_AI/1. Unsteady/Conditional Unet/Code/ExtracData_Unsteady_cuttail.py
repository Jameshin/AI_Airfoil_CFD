#"""
#@original author: Junghun Shin
#@modified by Wontae Hwang
#"""
import numpy as np
import pandas as pd

noCol = 11
numd = 17
initial_time = 103 
inc_time = 4
zone1_i = 689 
zone1_j = 145
cuttail = 0	
glayer = 145 	
dt = 0.1
sim_data_path = "../../../../Database/Simdata_airfoil_unsteady/sol01_RANS3/"
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

snapshot_data = np.array([]) 
POD = np.array([])
for i in range(0,numd):
	pd_data = pd.read_csv(sim_data_path+'flo001.0000'+str(initial_time+i*inc_time).rjust(3,'0')+'uns', dtype='float64', delimiter=' ', skipinitialspace=True, skiprows=2, header=None)
	# Make it an np ndarray
	data = np.nan_to_num(pd_data.values)
	array_data = data.flatten()
	array_data = array_data.reshape(-1,noCol)
	if i==0:
		snapshot_data = array_data
		xy = snapshot_data[:,0:2]
		N = snapshot_data.shape[0]
	else:
		snapshot_data = np.vstack((snapshot_data, array_data))
		snapshot_pod = array_data[:,[3,4,5,7]].flatten()[:,None]
array_data1 = snapshot_data

# Vectorize ezch variables
t_star = np.arange(initial_time, initial_time+numd*inc_time, inc_time) 
print(t_star)
T = t_star.shape[0]                                                

xc_star = array_data1[:,0]
yc_star = array_data1[:,1] 
NT = xc_star.shape[0]
dc_star = array_data1[:,3]
uc_star = array_data1[:,4]
vc_star = array_data1[:,5]
pc_star = array_data1[:,7]

DC = np.reshape(dc_star, (T,N)).T    
UC = np.reshape(uc_star, (T,N)).T 
VC = np.reshape(vc_star, (T,N)).T 
PC = np.reshape(pc_star, (T,N)).T 
XC = np.reshape(xc_star, (T,N)).T 
YC = np.reshape(yc_star, (T,N)).T 
TC = np.tile(t_star, (1,N)).T 

# Cut grid for memory efficiency
idx_x_slice = np.array([])
for i in range(glayer):
    idx_x_slice = np.append(idx_x_slice, np.arange((cuttail)+i*zone1_i,(zone1_i-(cuttail))+i*zone1_i)).astype('int32')
print(idx_x_slice.shape[0])
DC_star = DC[idx_x_slice,:]
UC_star = UC[idx_x_slice,:]
VC_star = VC[idx_x_slice,:]
PC_star = PC[idx_x_slice,:]
XC_star = XC[idx_x_slice,:]
YC_star = YC[idx_x_slice,:]
TC_star = TC[idx_x_slice,:]

np.savez(res_data_path+"Unstaedy_airfoil_cuttail.npz", TC=TC_star, XC=XC_star, YC=YC_star, DC=DC_star, UC=UC_star, VC=VC_star, PC=PC_star)
#np.savez(res_data_path+"Test_data.npz", TC=TC_star, XC=XC_star, YC=YC_star, DC=DC_star, UC=UC_star, VC=VC_star, PC=PC_star)

# UI_star = np.reshape(UC_star.T, [numd, glayer, zone1_i-2*cuttail])[0:numd,:,:]
# UC_field = UI_star[:,:-1,:-1]
# UC_field = UC_field[15].flatten()[:,None]
# print(UC_field)
# f = open("C:/Users/DD-SDOL/Desktop/Input_cuttail_175.dat", 'w')
# for i in range(UC_field.shape[0]):         
#     f.write(str(UC_field[i,0]))  
#     f.write("\n")
# f.close()




