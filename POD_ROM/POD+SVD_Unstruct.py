
# coding: utf-8
import numpy as np
import csv
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import RomObject

#initiate variables
noCol = 11
noConcernVar = 4
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
#create time_configurations
times = [list(range(initial_time, initial_time+numd*inc_time+1, inc_time))]
print(times)

#pickle the times
with open('dtimes.pkl', 'wb') as output:
    pickle.dump(times, output, pickle.HIGHEST_PROTOCOL)

#read extracted array
saved_npz = np.load("./PODarray_Unstruct.npz")
snapshot_data = saved_npz['snapshot'][:,:numd]
shp = snapshot_data.shape
xy = saved_npz['xy']
print(snapshot_data.shape)
np.savetxt("./xy.csv", xy)


#mean center the data
mean_array = None
mean_data_tensor = np.mean(snapshot_data, axis=1)
print(mean_data_tensor)

#compute the SVD of the covariance matrix
mean_centered_data = np.subtract(snapshot_data, np.tile(mean_data_tensor, (numd,1)).T)
u, s, v = np.linalg.svd(mean_centered_data, compute_uv=True, full_matrices=False)
print(mean_centered_data.shape)
print(u.shape)
print(v.shape)
print(s.shape)
compute_coeffs = np.matmul(np.transpose(mean_centered_data),u)
e = np.sum(s)
s_energy = np.divide(s,e)*100
coeffs = compute_coeffs
mean_array = mean_data_tensor

print('Raw s-matrix')
print(s)
print('Cumulative Energy')
print(e)
print('Normalized Energy')
print(s_energy)
#some assertions for correctness
#UU^T = I
uut = np.matmul(np.transpose(u),u)
np.testing.assert_almost_equal(uut, np.eye(u.shape[1],u.shape[1]))
#VV^T = I
vvt = np.matmul(np.transpose(v),v)
np.testing.assert_almost_equal(uut, np.eye(v.shape[1],v.shape[1]))

print(u.shape)
print(v.shape)
print(s.shape)

print(coeffs.shape)
print(coeffs.dtype)
print(compute_coeffs)

#Save coefficients (coeffs) , left-eigenvectors (u) and singular values (s)
rom_object = RomObject.romobject(u, s_energy, coeffs, mean_array)
with open('rom-object.pkl', 'wb') as output:
    pickle.dump(rom_object, output, pickle.HIGHEST_PROTOCOL)
