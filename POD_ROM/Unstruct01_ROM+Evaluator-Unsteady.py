# coding: utf-8

#Notebook to load data from the ROM creation and evaluate 
#based on user input

from os import X_OK
import tensorflow.compat.v1 as tf
import numpy as np
import pickle
import pandas as pd

tf.disable_v2_behavior()
#perform coefficient interpolation here, using numpy for it
numd = 20
initial_time = 100 
inc_time = 100 
zone1_n = 5143 
zone1_e = 10134
dt = 1
total_steps =20
input_design = np.arange(initial_time+inc_time, initial_time+inc_time+numd*inc_time*2+1, inc_time*2)
noConcernVar = 4
zone1_n = 5143 
zone1_e = 10134

#READ IN THE DESIGN INFORMATION
with open('designs.pkl', 'rb') as input:
    read_designs = pickle.load(input)[0]

#read in saved rom object
with open('rom-object.pkl', 'rb') as input:
    read_rom = pickle.load(input)

#read xy-coordinates 
pd_data = pd.read_csv('xy.csv', dtype='float64', delimiter=' ', header=None, skipinitialspace=True)
xydata = pd_data.values
print(xydata.shape)
print(xydata)

coeffs = read_rom.coeffsmat
u = read_rom.umat
mean_data = read_rom.mean_data[:,None]
mean_tensor = tf.constant(mean_data, name="mean_data_tensor")

# In this small case, CPU without GPU and use numpy 
for i, x in zip(range(total_steps), input_design):
    print(i)
    hi_idx = [idx for idx,v in enumerate(read_designs) if v >= x][0]
    lo_idx = hi_idx - 1
    #interpolate coefficients
    interp_coeffs = coeffs[lo_idx] + (coeffs[hi_idx]-coeffs[lo_idx])*(x-read_designs[lo_idx])/(read_designs[hi_idx]-read_designs[lo_idx])
    print(hi_idx, lo_idx, '=========')
    int_coeff_tensor = tf.Variable(interp_coeffs)
    #add a dim to make it a 2-D tensor
    int_coeff_tensor = tf.expand_dims(int_coeff_tensor, 0)
    print(coeffs.shape, u.shape, mean_data.shape, int_coeff_tensor.shape)    
    #compute the POD approximation
#    init_op = tf.global_variables_initializer()    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result_op = tf.matmul(int_coeff_tensor, tf.transpose(u) )
        modal_result = sess.run(tf.transpose(result_op))
        modal_result.flatten()
        mean_tensor.eval()
        result_op = tf.add(modal_result, mean_tensor)
        result = sess.run(result_op)
    result = result.reshape(-1, noConcernVar)
    height, width = result.shape
    #Create p3d field data
    result = result.T.flatten()[:,None]
    p3d_result = np.vstack((xydata, result))
    #Save tecplot field data
    filename = "./PODresults/Case_POD_Unstr_t="+str(x).rjust(2,'0')+".dat"
    np.savetxt(filename, p3d_result, delimiter=" ", header="variables = X, Y, d, u, v, p \n zone t= \"0.282832E-01\", n= "+str(zone1_n)+", e= "+str(zone1_e)+"\n ,varlocation=([3,4,5,6]=cellcentered),zonetype=fetriangle \n datapacking=block", comments=' ')
    with open(filename, 'a') as outfile:
        with open("EL_mat.dat") as file:
            outfile.write(file.read())

