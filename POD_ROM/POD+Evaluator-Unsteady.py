
# coding: utf-8

#Notebook to load data from the ROM creation and evaluate 
#based on user input

import tensorflow.compat.v1 as tf
import numpy as np
import pickle
import pandas as pd


#perform coefficient interpolation here, using numpy for it
total_steps =19
input_design = np.arange(202, 240, 2)
noConcernVar = 4
zone1_i = 689
zone1_j = 145

#READ IN THE DESIGN INFORMATION
with open('designs4.pkl', 'rb') as input:
    read_designs = pickle.load(input)[0]

#read in saved rom object
with open('rom-object4.pkl', 'rb') as input:
    read_rom = pickle.load(input)

#read xy-coordinates 
pd_data = pd.read_csv('xy.csv', dtype='float64', delimiter=' ', header=None, skipinitialspace=True)
xydata = pd_data.values
print(xydata.shape)
print(xydata)

coeffs = read_rom.coeffsmat
u = read_rom.umat
mean_data = read_rom.mean_data
mean_tensor = tf.constant(mean_data, name="mean_data_tensor")

print(read_designs)


# In this small case, CPU without GPU and use numpy 
for i, x in zip(range(total_steps), input_design):
    print(i)
    hi_idx = [idx for idx,v in enumerate(read_designs) if v > x][0]
    lo_idx = hi_idx - 1
    #interpolate coefficients
    interp_coeffs = coeffs[lo_idx] + (coeffs[hi_idx]-coeffs[lo_idx])*(x-read_designs[lo_idx])/(read_designs[hi_idx]-read_designs[lo_idx])
    int_coeff_tensor = tf.Variable(interp_coeffs)
    #add a dim to make it a 2-D tensor
    int_coeff_tensor = tf.expand_dims(int_coeff_tensor, 0)
    print(int_coeff_tensor)       
    #compute the POD approximation
#    init_op = tf.global_variables_initializer()    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result_op = tf.matmul(int_coeff_tensor, tf.transpose(u) )
        modal_result = sess.run(tf.transpose(result_op))
        modal_result.flatten()
        mean_tensor.eval()
        result_op = tf.add( modal_result, mean_tensor)
        result = sess.run(result_op)
    result = result.reshape((int(modal_result.shape[0]/noConcernVar), noConcernVar))
    height, width = result.shape
    #Create p3d field data
    p3d_result = np.hstack((xydata, result))
    #Save tecplot field data
    np.savetxt("./Case_flo_POD_t="+str(x)+".dat", p3d_result,     delimiter=" ", header="variables = X, Y, rho, u, v, p \n zone i= "+str(zone1_i)+", j=    "+str(zone1_j), comments=' ')


