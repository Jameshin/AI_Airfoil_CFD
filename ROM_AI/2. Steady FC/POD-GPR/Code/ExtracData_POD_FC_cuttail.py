#"""
#@original author: Junghun Shin
#@modified by Wontae Hwang
#"""
import numpy as np
import pandas as pd

Mach=['1', '2', '3', '4', '5', '6', '7', '8', '9']
Mach=np.array(Mach)
AoA = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
AoA=np.array(AoA)
NMa = 4
NAoA = 5
noCol = 11
numd = 123
zone1_i = 401 
zone1_j = 81
cuttail = 20	
glayer = 64 	
sim_data_path = "../../../../Database/Simdata_airfoil_Steady_FC/"
res_data_path = "../Results/"
Tecplot_header_in = "variables=X, Y, Z, Rho, U, V, W, P, T, Vor, Qcri"
Tecplot_header_out = "variables=X, Y, Rho, U, V, P"

# list of file names
filenames = []
merged = []
Ntime = 0 

for i in (Mach):
	for j in (AoA):
		filenames.append(sim_data_path+"flo_" + str(i) +"_" + str(j) + ".dat")
	Ntime += 1

def is_number(num):
    try:
        float(num)
        return True #num을 float으로 변환할 수 있는 경우
    except ValueError: #num을 float으로 변환할 수 없는 경우
        return False

k = 0; kk = 0
Nm = 1; Na = 1
Nmt = 2; Nat = 3

for file in filenames:
	snapshot_data = []
	print(k)
	Testname = sim_data_path+"flo_" + str(Nmt) +"_" + str(Nat) + ".dat"
	with open(file) as f:
		lines = f.readlines()
		for line in lines:
			vals = []
			line = line.replace("\n","")
			raw_vals = line.split(" ")
			for c in raw_vals:
				if is_number(c):
					vals.append(c)
			
			if len(vals) == noCol:
				snapshot_data.append(vals)

		snapshot_data=np.array(snapshot_data)
		snapshot_data.reshape(-1,noCol)
		
		if k == 0:
			Traindata = np.zeros((123,snapshot_data.shape[0],snapshot_data.shape[1]))
			Testdata = np.zeros((12,snapshot_data.shape[0],snapshot_data.shape[1]))
			Trainlabel = np.zeros((123,2))
			Testlabel = np.zeros((12,2))

		if 	Testname == file:
			print(Testname)
			Testdata[kk,:,:] = snapshot_data
			kk += 1
			Nat +=5
			if kk%3 == 0:
				Nmt +=2
				Nat = 3
			Testlabel[kk-1,0] = Nm
			Testlabel[kk-1,1] = Na
			
		else:	
			Traindata[k,:,:] = snapshot_data
			k += 1
			Trainlabel[k-1,0] = Nm
			Trainlabel[k-1,1] = Na

		if Na%15 == 0:
			Nm +=1
			Na = 1
		Na += 1

for i in range(2):
	if i == 0:
		array_data = Traindata
		save_path = res_data_path+"Staedy_airfoil_cuttail_train.npz"
				
	if i == 1:
		array_data = Testdata
		save_path = res_data_path+"Staedy_airfoil_cuttail_test.npz"

	xc_star = array_data[:,:,0] 
	yc_star = array_data[:,:,1] 
	
	dc_star = array_data[:,:,3]
	uc_star = array_data[:,:,4]
	vc_star = array_data[:,:,5]
	pc_star = array_data[:,:,7]

	DC = dc_star.T 
	UC = uc_star.T 
	VC = vc_star.T 
	PC = pc_star.T   
	XC = xc_star.T 
	YC = yc_star.T  

	#Cut tail
	idx_x_slice = np.array([])
	for i in range(glayer):
		idx_x_slice = np.append(idx_x_slice, np.arange(cuttail+i*zone1_i, 
								(zone1_i-cuttail)+i*zone1_i)).astype('int32')
	DC_star = DC[idx_x_slice,:]
	UC_star = UC[idx_x_slice,:]
	VC_star = VC[idx_x_slice,:]
	PC_star = PC[idx_x_slice,:]
	XC_star = XC[idx_x_slice,:]
	YC_star = YC[idx_x_slice,:]

	np.savez(save_path, XC=XC_star, YC=YC_star, DC=DC_star, UC=UC_star, VC=VC_star, PC=PC_star)