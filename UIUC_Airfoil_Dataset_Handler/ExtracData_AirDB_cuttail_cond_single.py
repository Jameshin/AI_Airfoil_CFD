import numpy as np
import time
import sys
import pandas as pd
import os

Nfoil = 1000
Re = np.array([1.0e5, 2.0e5, 1.0e5])
Mach = np.array([0.2])
AOA = np.array([1.0])
noCol = 11
zone1_i = 401 
zone1_j = 81
crop_i = 20
crop_j = 64
data_path = "D:\\SimData\\UIUC_ML\\CFDdataset\\"
write_path = "D:\\Kaggle\\"
Tecplot_header_in = "variables=X, Y, Z, Rho, U, V, W, P, T, Vor, Qcri \n zone i="+str(zone1_i)+" j="+str(zone1_j)+" "
Tecplot_header_out = "variables = X, Y, rho, u, v, p \n zone i="+str(zone1_i-2*crop_i)+" j="+str(crop_j)+" "
# create directory if not exist
#os.makedirs(os.path.dirname(write_data_path), exist_ok=True)

###
start = int(time.time())
# list of file names
Ncon = Re.shape[0] * Mach.shape[0] * AOA.shape[0]
Ncase = 0
foilcases = [] 
for j in range(Nfoil):
    sim_data_path = data_path + 'Airfoil_' + str(j+1).rjust(4,'0') + '\\'
    for k in range(1,Re.shape[0]+1):
        for l in range(1,Mach.shape[0]+1):
            for m in range(1,AOA.shape[0]+1):
                foilcases.append(sim_data_path+"result_"+str(k)+"_"+str(l)+"_"+str(m).rjust(2, '0')+"\\flo001.dat")
                Ncase += 1
print(Nfoil, Ncon, Ncase)

###
snapshot_data = [] # np.array([])   []
#POD = np.array([])
for i in range(Ncase):
    pd_data1 = pd.read_csv(foilcases[i], na_filter=True, dtype='float64', delimiter=' ', skipinitialspace=True, skiprows=2, header=None)
    data = np.nan_to_num(pd_data1.values)
    array_data = data.flatten()
    array_data = array_data.reshape(-1,noCol)
    snapshot_data.append(array_data) # snapshot_data = np.vstack((snapshot_data, array_data))
    if i==0:
        N = array_data.shape[0]
        #POD = array_data[:,[3,4,5,7]].flatten()[:,None]
    '''
    if i==0:
        snapshot_data = array_data
        N = snapshot_data.shape[0]
        #POD = array_data[:,[3,4,5,7]].flatten()[:,None]
    else:
        snapshot_data = np.vstack((snapshot_data, array_data))
        #snapshot_pod = array_data[:,[3,4,5,7]].flatten()[:,None]
        #POD = np.hstack((POD,snapshot_pod))
    '''
snapshot_dataset = np.vstack((snapshot_data))
shp = snapshot_dataset.shape
print(shp)
###
con_star = 1+np.arange(Ncon)[:,None] # T(=1) x 1
xc_star = snapshot_dataset[:,0] # N x C x 1
yc_star = snapshot_dataset[:,1] # N x C x 1
dc_star = snapshot_dataset[:,3]
uc_star = snapshot_dataset[:,4]
vc_star = snapshot_dataset[:,5]
pc_star = snapshot_dataset[:,7]
#print(POD.shape)
#np.savez("./PODarray1.npz", xy=xy, snapshot=POD)

DC = np.reshape(dc_star, (Nfoil,Ncon,N)) # Nfoil x T x N     
UC = np.reshape(uc_star, (Nfoil,Ncon,N)) # 
VC = np.reshape(vc_star, (Nfoil,Ncon,N)) # 
PC = np.reshape(pc_star, (Nfoil,Ncon,N)) # 
XC = np.reshape(xc_star, (Nfoil,Ncon,N)) # 
YC = np.reshape(yc_star, (Nfoil,Ncon,N)) # 
CC = np.tile(con_star, (Nfoil,1,N)) # Nfoil x T x N
print(DC.shape)
####idx_x_slice = np.array([])
idx_x_slice = np.array([])
for i in range(crop_j):
    idx_x_slice = np.append(idx_x_slice, np.arange(crop_i+i*zone1_i, (zone1_i-crop_i)+i*zone1_i)).astype('int32')
    print(idx_x_slice.shape[0])
DC_star = np.transpose(DC[:,:,idx_x_slice], (0,1,2))
UC_star = np.transpose(UC[:,:,idx_x_slice], (0,1,2))
VC_star = np.transpose(VC[:,:,idx_x_slice], (0,1,2))
PC_star = np.transpose(PC[:,:,idx_x_slice], (0,1,2))
XC_star = np.transpose(XC[:,:,idx_x_slice], (0,1,2))
YC_star = np.transpose(YC[:,:,idx_x_slice], (0,1,2))
CC_star = np.transpose(CC[:,:,idx_x_slice], (0,1,2))

#DC_star = DC
#UC_star = UC
#VC_star = VC
#PC_star = PC
#XC_star = XC
#YC_star = YC
#TC_star = TC
###
np.savez("./array_foil_UIUC_cuttail2064_AOA16.npz", CC=CC_star, XC=XC_star, YC=YC_star, DC=DC_star, UC=UC_star, VC=VC_star, PC=PC_star)
### check
#saved = np.load("./array_foil_cuttail.npz")
print("***run time(sec) :", int(time.time()) - start)
### To check flow field
#saved = np.load("./array_foil_cuttail.npz")
for i in range(0,2):
    for snap in range(Ncon):
        p3d_result = np.hstack((XC_star[i,snap,:][:,None], YC_star[i,snap,:][:,None], DC_star[i,snap,:][:,None], UC_star[i,snap,:][:,None], VC_star[i,snap,:][:,None], PC_star[i,snap,:][:,None]))
        np.savetxt(write_path+"Case_flo_Foil="+str(i*Ncon+snap)+"_single.dat", p3d_result, delimiter=" ", header=Tecplot_header_out, comments=' ')