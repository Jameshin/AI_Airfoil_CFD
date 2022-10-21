#"""
#@original author: Junghun Shin
#@modified by Wontae Hwang
#"""
import numpy as np
import pandas as pd
import random

Nshape = 500
NAoA = 26
Ntotal = Nshape*NAoA
Nshapet = 20
NAoAt = 5
AoAt = ['5', '11', '17', '22', '25']
Ntest = Nshapet*NAoAt

zone1_i = 401
zone1_j = 81
cuttail = 20	
glayer = 64 	
sim_data_path = "../../../Database/Simdata_airfoil_Steady_Shape_FC/"
res_data_path = "../Results/"
Tecplot_header_in = "variables=X, Y, Z, Rho, U, V, W, P, T, Vor, Qcri"
Tecplot_header_out = "variables=X, Y, Rho, U, V, P"

# list of file names
filenames = []
Ntime = 0 
for i in range (Nshape):
	for j in range (NAoA):
		filenames.append(sim_data_path+"Airfoil_"+str(i+1).rjust(4,'0')+"/result_1_1_" +str(j+1).rjust(2,'0')+ "/flo001.dat")
	Ntime += 1

# Test file names
shapet = []   
testfilenames = []    
testlist = np.zeros((Ntest,2))                        
for i in range(Nshapet):
	a = random.randint(1,Nshape)       
	while a in shapet :              
		a = random.randint(1,Nshape)
	shapet.append(a) 
shapet.sort()

jj = 0
for i in (shapet):
	for j in (AoAt):
		testfilenames.append(sim_data_path+"Airfoil_"+str(i).rjust(4,'0')+"/result_1_1_" +str(j).rjust(2,'0')+ "/flo001.dat")
		testlist[jj,:] = np.array([i,j])
		jj += 1

pd.DataFrame(testlist).astype('int').to_csv(res_data_path+'TestList.dat'.format(), header=False, index=True)
print(testlist)

k = 0; kk = 1; kkk = 0
idx_x_slice = np.array([])
for i in range(glayer):
	idx_x_slice = np.append(idx_x_slice, np.arange(cuttail+i*zone1_i, 
							(zone1_i-cuttail)+i*zone1_i)).astype('int32')

for file in filenames:
	snapshot_data = []
	pd_data = pd.read_csv(file, dtype='float32', delimiter=' ', skipinitialspace=True, skiprows=2, header=None)
	pd_data = pd_data.drop([2,3,6,8,9,10], axis=1)
	pd_data = pd_data.loc[idx_x_slice,:]
	snapshot_data = pd_data

	if k == 0:
		Traindata = []
		Testdata = []
		trainlab = np.zeros(Ntotal-Ntest)
		testlab = np.zeros(Ntest)
	if 	file in testfilenames:
		print(file, testlab[kkk])
		Testdata.append(snapshot_data)
		testlab[kkk] = kk
		kkk += 1
	else:	
		Traindata.append(snapshot_data)
		trainlab[k] = kk
		print(k+1, trainlab[k])
		k += 1
	if kk%26 == 0:
		kk = 0
	kk += 1
pd.DataFrame(trainlab).astype('int').to_csv(res_data_path+'TrainLabel.dat'.format(), header=False, index=False)
pd.DataFrame(testlab).astype('int').to_csv(res_data_path+'TestLabel.dat'.format(), header=False, index=False)

for i in range(2):
	if i == 0:
		array_data = np.array(Traindata)
		save_path = res_data_path+"Staedy_airfoil_cuttail_train.npz"
		
	if i == 1:
		array_data = np.array(Testdata)
		save_path = res_data_path+"Staedy_airfoil_cuttail_test.npz"
	xc_star = array_data[:,:,0] 
	yc_star = array_data[:,:,1] 
	uc_star = array_data[:,:,2]
	vc_star = array_data[:,:,3]
	pc_star = array_data[:,:,4]
 
	UC = uc_star.T  
	VC = vc_star.T   
	PC = pc_star.T   
	XC = xc_star.T   
	YC = yc_star.T 
	print(XC.shape)

	np.savez(save_path, XC=XC, YC=YC, UC=UC, VC=VC, PC=PC)

# print(UC_star.shape)
# print()
# PI_star = np.reshape(PC_star.T, [12, glayer, zone1_i-2*cuttail])[0:12,:,:]
# PC_field = PI_star[:,:,:-1]
# PC_field = PC_field[0].flatten()[:,None]
# print(PC_field)
# f = open("C:/Users/ddsdol/Desktop/Input_cuttail_18.dat", 'w')
# for i in range(PC_field.shape[0]):         
# 	f.write(str(PC_field[i,0]))  
# 	f.write("\n")
# f.close()
