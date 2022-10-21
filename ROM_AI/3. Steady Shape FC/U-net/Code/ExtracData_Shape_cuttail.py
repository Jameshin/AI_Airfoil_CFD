#"""
#@original author: Junghun Shin
#@modified by Myeongjun Song
#"""
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from itertools import product

Nfoil = 1070
Re = np.array([1.0e5])
Mach = np.array([0.2])
AOA = np.array([1.0]) 
noCol = 11
zone1_i = 401 
zone1_j = 81
crop_i = 20
crop_j = 64
data_path = "../../../../Database/Simdata_airfoil_Steady_Shape/"
res_data_path = "../Results/"
Tecplot_header_in = "variables=X, Y, Z, Rho, U, V, W, P, T, Vor, Qcri \n zone i="+str(zone1_i)+" j="+str(zone1_j)+" "
Tecplot_header_out = "variables = X, Y, rho, u, v, p \n zone i="+str(zone1_i-2*crop_i)+" j="+str(crop_j)+" "

# list of file names  
Ncon = Re.shape[0] * Mach.shape[0] * AOA.shape[0]
Ncase = 0
foilcases = [] 
for j in range(Nfoil):
    sim_data_path = data_path + 'Airfoil_' + str(j+1).rjust(4,'0') + '/'
    for k in range(1,Re.shape[0]+1):
        for l in range(1,Mach.shape[0]+1):
            for m in range(1,AOA.shape[0]+1):
                foilcases.append(sim_data_path+"result_"+str(k)+"_"+str(l)+"_"+str(m).rjust(2, '0')+"/flo001.dat")
            Ncase += 1

###
def snapshot_dataset(i,j):        
        pd_data1 = pd.read_csv(foilcases[i*Ncon+j], na_filter=True, dtype='float64', delimiter=' ', skipinitialspace=True, skiprows=2, header=None)
        data = np.nan_to_num(pd_data1.values)
        array_data = data.flatten()
        array_data = array_data.reshape(-1,noCol)
        snapshot_data = array_data
        #snapshot_POD = array_data[:,[3,4,5,7]].flatten()[:,None]
        return snapshot_data

###
if __name__ == "__main__":
        #
    POD = np.array([])
    pd_data2 = pd.read_csv(foilcases[0], na_filter=True, dtype='float64', delimiter=' ', skipinitialspace=True, skiprows=2, header=None)
    data = np.nan_to_num(pd_data2.values)
    array_data = data.flatten()
    array_data = array_data.reshape(-1,noCol)
    N = array_data.shape[0]

    ### Distributed Session
    num_cores = cpu_count() #4
    pool = Pool(num_cores)
    snapshot_dataset = np.concatenate(pool.starmap(snapshot_dataset, product(range(Nfoil), range(Ncon))))
    shp = snapshot_dataset.shape
    print(shp)
    pool.close()
    pool.join()
    ###
    con_star = 1+np.arange(Ncon)[:,None]
    xc_star = snapshot_dataset[:,0] 
    yc_star = snapshot_dataset[:,1] 
    dc_star = snapshot_dataset[:,3]
    uc_star = snapshot_dataset[:,4]
    vc_star = snapshot_dataset[:,5]
    pc_star = snapshot_dataset[:,7]

    XC = np.reshape(xc_star, (Nfoil,Ncon,N)) 
    YC = np.reshape(yc_star, (Nfoil,Ncon,N))  
    CC = np.tile(con_star, (Nfoil,1,N)) 
    DC = np.reshape(dc_star, (Nfoil,Ncon,N))   
    UC = np.reshape(uc_star, (Nfoil,Ncon,N))  
    VC = np.reshape(vc_star, (Nfoil,Ncon,N))  
    PC = np.reshape(pc_star, (Nfoil,Ncon,N))

    #### Slicing to get cropped airfoils    
    idx_x_slice = np.array([])
    for i in range(crop_j):
        idx_x_slice = np.append(idx_x_slice, np.arange(crop_i+i*zone1_i, (zone1_i-crop_i)+i*zone1_i)).astype('int32')
        print(idx_x_slice.shape[0])
    XC_star = np.transpose(XC[:,:,idx_x_slice], (0,1,2))
    YC_star = np.transpose(YC[:,:,idx_x_slice], (0,1,2))
    CC_star = np.transpose(CC[:,:,idx_x_slice], (0,1,2))
    DC_star = np.transpose(DC[:,:,idx_x_slice], (0,1,2))
    UC_star = np.transpose(UC[:,:,idx_x_slice], (0,1,2))
    VC_star = np.transpose(VC[:,:,idx_x_slice], (0,1,2))
    PC_star = np.transpose(PC[:,:,idx_x_slice], (0,1,2))

    ###
    np.savez(res_data_path+"array_foil_UIUC_cuttail2064_AOA16.npz", CC=CC_star, XC=XC_star, YC=YC_star, DC=DC_star, UC=UC_star, VC=VC_star, PC=PC_star)
    ### To check flow field
    for i in range(1070):
        if i>=1050:
            for snap in range(Ncon):
                p3d_result = np.hstack((XC_star[i,snap,:][:,None], YC_star[i,snap,:][:,None], DC_star[i,snap,:][:,None], UC_star[i,snap,:][:,None], VC_star[i,snap,:][:,None], PC_star[i,snap,:][:,None]))
                np.savetxt(res_data_path+"Case_flo_Foil="+str(i*Ncon+snap)+"_parallel.dat", p3d_result, delimiter=" ", header=Tecplot_header_out, comments=' ')
