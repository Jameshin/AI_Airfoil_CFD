import numpy as np
import time
import sys
import pandas as pd
import os
noVar = 5
ElCol = 3
numd = 21
initial_time = 100 
inc_time = 200 
zone1_n = 5143 
zone1_e = 10134
dt = 1
wt = 1000 
d_inf = 1.225
U_inf = 0.005*343
sim_data_path = "D:\\JupyterNBook\\PINN_Unstruc\\cfddata\\result_Ma0.4_AOA15\\"
res_data_path = "../Data/airfoil_unsteady/results/"
Tecplot_header_in = "variables =x, y, rho, u, v, p, m"
Tecplot_header_out = "variables =x, y, rho, u, v, p, m"
# create directory if not exist
#os.makedirs(os.path.dirname(res_data_path), exist_ok=True)
n_xy = zone1_n*2
n_field = zone1_n*2 + zone1_e*noVar
n_elem = zone1_e*ElCol

# list of file names
filenames = []
merged = []
Ntime = 1 
for i in range(0, numd):
	filenames.append(sim_data_path+"mid_result_"+str(initial_time+i*inc_time).rjust(5,'0')+".rlt") 
	Ntime += 1
print(Ntime, filenames)
###
snapshot_data = np.array([]) 
for i in range(0,numd):
	pd_data = pd.read_csv(sim_data_path+"mid_result_"+str(initial_time+i*inc_time).rjust(5,'0')+'.rlt', nrows=n_field, dtype='float64', delimiter=' ', skipinitialspace=True, skiprows=4, header=None)
	data = pd_data.values	
	data = data[~np.isnan(data)]
	array_data = data.flatten()
	#print(array_data.shape, array_data_elem.shape)
	array_data_field = array_data[:n_field][:,None]	
	if i==0:
		pd_data_elem = pd.read_csv(sim_data_path+"mid_result_"+str(initial_time+i*inc_time).rjust(5,'0')+'.rlt', dtype='float64', delimiter=' ', skipinitialspace=True, skiprows=4+n_field, header=None)
		snapshot_data = array_data_field
		xy = snapshot_data[0:n_xy,:]
		data_elem = pd_data_elem.values
		data_elem = data_elem[~np.isnan(data_elem)]
		array_data_elem = data_elem.flatten().astype('int32')
		array_data_elem = array_data_elem.reshape(-1,ElCol)	
		#N = snapshot_data.shape[0]
		#POD = array_data[:,[3,4,5,7]].flatten()[:,None]
		print(snapshot_data)
	else:
		snapshot_data = np.hstack((snapshot_data, array_data_field))
		#snapshot_pod = array_data[:,[3,4,5,7]].flatten()[:,None]
		#POD = np.hstack((POD,snapshot_pod))
array_data1 = snapshot_data
shp = array_data1.shape
print(shp)
###
t_star = np.arange(Ntime)[:,None]*dt*inc_time # T(=1) x 1
T = t_star.shape[0]
print(array_data_elem.shape)
TN = np.tile(t_star, (zone1_n,1))
XN = array_data1[0:zone1_n,:]
YN = array_data1[zone1_n:n_xy,:]
DE = array_data1[n_xy:n_xy+zone1_e,:] # Ne x T
UE = array_data1[n_xy+zone1_e:n_xy+2*zone1_e,:]
VE = array_data1[n_xy+2*zone1_e:n_xy+3*zone1_e,:]
PE = array_data1[n_xy+3*zone1_e:n_xy+4*zone1_e,:]
XE = np.empty([zone1_e,numd])
YE = np.empty([zone1_e,numd])
for i in range(zone1_e):
    XE[i] = (XN[array_data_elem[i,0]-1,:] + XN[array_data_elem[i,1]-1,:] + XN[array_data_elem[i,2]-1,:])/3
    YE[i] = (YN[array_data_elem[i,0]-1,:] + YN[array_data_elem[i,1]-1,:] + YN[array_data_elem[i,2]-1,:])/3
print(XN) #"X","Y","rh","u","v","w","p","M","vorticity"

# Extract POD
for i in range(0,numd):
	if i==0:
		POD = np.hstack((DE[:,i:i+1],UE[:,i:i+1],VE[:,i:i+1],PE[:,i:i+1])).flatten()[:,None]
	else:
		snapshot_pod = np.hstack((DE[:,i:i+1],UE[:,i:i+1],VE[:,i:i+1],PE[:,i:i+1])).flatten()[:,None]
		POD = np.hstack((POD,snapshot_pod))
#np.savez("./PODarray_Unstruct.npz", xy=xy, snapshot=POD)

###
np.savez("./array_Unst_21.npz", TN=TN, XN=XN, YN=YN, XE=XE, YE=YE, DE=DE, UE=UE, VE=VE, PE=PE, EL=array_data_elem)
### check
#saved = np.load("./array_Unst_21.npz")
#print(saved['TC'])
#for i in range(Ntime):
#	t_test = T_test[:,i:i+1]
#   x_test = XE_star[:,i:i+1]
#   y_test = YE_star[:,i:i+1]
#   l_test = L_star[:,i:i+1]
#   d_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)
#   tecplot_result = np.vstack((XC_star[:,0][:,None], YC_star[:,0][:,None], d_pred, u_pred, v_pred, p_pred))
#   filename = "./Case_flo_Unstr_t="+str(i).rjust(2,'0')+".dat"
#   np.savetxt(filename, tecplot_result, delimiter=" ", header="variables = X, Y, d, u, v, p \n zone t= \"0.282832E-01\", n= "+str(zone1_n)+" e= "+str(zone1_e)+"\n ,varlocation=([3,4,5,6]=cellcentered),zonetype=fetriangle \n datapacking=block", comments=' ')
#   with open(filename, 'a') as outfile:
#       with open("EL_mat.dat") as file:
#           outfile.write(file.read())