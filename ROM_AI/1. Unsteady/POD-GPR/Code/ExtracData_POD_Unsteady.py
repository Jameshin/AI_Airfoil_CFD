#"""
#@original author: Junghun Shin
#@modified by Wontae Hwang
#"""
import numpy as np
import pandas as pd

noCol = 11		# 유동장 변수
numd = 20		# 스냅샷 개수
initial_time = 101 	
inc_time = 4	# 학습 데이터의 시간 간격
zone1_i = 689 
zone1_j = 145
sim_data_path = "../../../../Database/Simdata_airfoil_unsteady/sol01_RANS3/"
res_data_path = "../Results/"
Tecplot_header_in = "variables=X, Y, Z, Rho, U, V, W, P, T, Vor, Qcri"
Tecplot_header_out = "variables=X, Y, Rho, U, V, P"

## list of file names
filenames = []
merged = []
Ntime = 0 
for i in range(0, numd):
	filenames.append(sim_data_path+"flo001.0000"+str(initial_time+i*inc_time).rjust(3,'0')+"uns") 
	Ntime += 1
print(Ntime, filenames)

## Stack snapshots 
snapshot_data = np.array([]) 
POD = np.array([])
for i in range(0,numd):
	# Read data
	pd_data = pd.read_csv(sim_data_path+'flo001.0000'+str(initial_time+i*inc_time).rjust(3,'0')+'uns', dtype='float64', delimiter=' ', skipinitialspace=True, skiprows=2, header=None)
	data = np.nan_to_num(pd_data.values)
	array_data = data.flatten()
	array_data = array_data.reshape(-1,noCol)
	if i==0:
		snapshot_data = array_data
		xy = snapshot_data[:,0:2]
		N = snapshot_data.shape[0]
		POD = array_data[:,[3,4,5,7]].flatten()[:,None]
	else:
		snapshot_pod = array_data[:,[3,4,5,7]].flatten()[:,None]
		POD = np.hstack((POD,snapshot_pod))                                             

# Export snapshots
np.savez(res_data_path+"PODarray.npz", xy=xy, snapshot=POD)

