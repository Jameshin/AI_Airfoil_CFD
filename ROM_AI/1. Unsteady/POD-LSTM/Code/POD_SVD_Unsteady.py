#"""
#@original author: Junghun Shin
#@modified by Myeongjun Song
#"""
import numpy as np
import csv
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

noCol = 11
numd = 80 
size = 80
res_data_path = "../Results/"

#create time_configurations
times = [list(range(203, 280))]
with open('times4.pkl', 'wb') as output:
    pickle.dump(times, output, pickle.HIGHEST_PROTOCOL)
saved_npz = np.load(res_data_path+"PODarray.npz")
snapshot_data = saved_npz['snapshot']
shp = snapshot_data.shape
print(shp)
xy = saved_npz['xy']
print(xy.shape)
mean_array = None
mean_data_tensor = np.mean(snapshot_data, axis=1)
mean_centered_data = np.subtract(snapshot_data, np.tile(mean_data_tensor, (numd,1)).T)

# POD coefficients 
u, s, v = np.linalg.svd(mean_centered_data, compute_uv=True, full_matrices=False)
compute_coeffs = np.matmul(np.transpose(mean_centered_data),u)
e = np.sum(s)
s_energy = np.divide(s,e)*100

MC = np.zeros((numd,size))
for i in range(numd):
	for j in range(size):
		MC[i][j] = compute_coeffs[i][j]                
t = np.linspace(10.0,17.9,size)    
BF = u
mean_array3 = mean_data_tensor
