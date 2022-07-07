import numpy as np
import time
import sys
import pandas as pd
import os

Re = np.array([1.0e5])
Mach = np.array([0.2])
AOA = np.array([1.0])
noCol = 11
zone1_i = 401 
zone1_j = 81
glayer = 64
cuttail = 0 
d_inf = 1.225
U_inf = 0.005*343
data_path = "C:\\SimData\\UIUC_ML\\CFDdataset\\"
res_data_path = "~/DaDri/AirfoilDB/"
Tecplot_header_in = "variables=X, Y, Z, Rho, U, V, W, P, T, Vor, Qcri"
Tecplot_header_out = "variables=X, Y, Rho, U, V, P"
# create directory if not exist
#os.makedirs(os.path.dirname(res_data_path), exist_ok=True)

# list of file names
filenames = [] 
Nfoil = 1550
Ncon = 0
for j in range(Nfoil):
    sim_data_path = data_path + 'Airfoil_' + str(j+1).rjust(4,'0') + '/'
    for k in range(1,Re.shape[0]+1):
        for l in range(1,Mach.shape[0]+1):
            for m in range(1,AOA.shape[0]+1):
                filenames.append(sim_data_path+"result_"+str(k)+"_"+str(l)+"_"+str(m).rjust(2, '0')+"/flo001.dat")
            Ncon += 1
print(Nfoil, Ncon)
###
snapshot_data = np.array([]) 
POD = np.array([])
pd_data1 = pd.DataFrame([])
for i in range(Ncon):
    pd_data1 = pd.read_csv(filenames[i], na_filter=True, dtype='float64', delimiter=' ', skipinitialspace=True, skiprows=2, header=None)
    data = np.nan_to_num(pd_data1.values)
    array_data = data.flatten()
    array_data = array_data.reshape(-1,noCol)
    if i==0:
        snapshot_data = array_data
        xy = snapshot_data[:,0:2]
        N = snapshot_data.shape[0]
        #POD = array_data[:,[3,4,5,7]].flatten()[:,None]
    else:
        snapshot_data = np.vstack((snapshot_data, array_data))
        #snapshot_pod = array_data[:,[3,4,5,7]].flatten()[:,None]
        #POD = np.hstack((POD,snapshot_pod))
array_data1 = snapshot_data
shp = array_data1.shape
print(shp)
###
con_star = 1+np.arange(Ncon)[:,None] # T(=1) x 1
#print(POD.shape)
xc_star = array_data1[:,0] # NT x 1
yc_star = array_data1[:,1] # NT x 1
NT = xc_star.shape[0]
dc_star = array_data1[:,3]
uc_star = array_data1[:,4]
vc_star = array_data1[:,5]
pc_star = array_data1[:,7]
#print(xc_star.shape) #"X","Y","rh","u","v","w","p","M","vorticity"
#np.savez("./PODarray1.npz", xy=xy, snapshot=POD)

DC = np.reshape(dc_star, (Ncon,N)).T # N x T     
UC = np.reshape(uc_star, (Ncon,N)).T # N x T
VC = np.reshape(vc_star, (Ncon,N)).T # N x T
PC = np.reshape(pc_star, (Ncon,N)).T # N x T
XC = np.reshape(xc_star, (Ncon,N)).T # N x T
YC = np.reshape(yc_star, (Ncon,N)).T # N x T
TC = np.tile(con_star, (1,N)).T # N x T

####idx_x_slice = np.array([])
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

#DC_star = DC
#UC_star = UC
#VC_star = VC
#PC_star = PC
#XC_star = XC
#YC_star = YC
#TC_star = TC
###
np.savez("./array_foil_UIUC_cuttail0064.npz", TC=TC_star, XC=XC_star, YC=YC_star, DC=DC_star, UC=UC_star, VC=VC_star, PC=PC_star)
### check
#saved = np.load("./array_foil_cuttail.npz")
#print(saved['TC'])
'''
for snap in range(0,1):
	p3d_result = np.hstack((xy[:,0][:,None], xy[:,1][:,None], DC[:,snap:snap+1], UC[:,snap:snap+1], VC[:,snap:snap+1], PC[:,snap:snap+1]))
	np.savetxt("./Case_flo_Foil="+str(snap+1)+".dat", p3d_result, delimiter=" ", header="variables = X, Y, rho, u, v, p \n zone i="+str(zone1_i)+" j="+str(zone1_j)+" ", comments=' ')
'''