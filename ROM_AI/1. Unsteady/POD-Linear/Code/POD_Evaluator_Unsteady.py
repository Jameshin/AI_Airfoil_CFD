#"""
#@original author: Junghun Shin
#@modified by Wontae Hwang
#"""
import tensorflow.compat.v1 as tf
import numpy as np
import pickle
import pandas as pd
import os

from CFDLib import mean_squared_error, relative_error

# Tensorflow v2 off
tf.disable_v2_behavior()

# Perform coefficient interpolation here, using numpy for it
total_steps = 19
star_time = 203
end_time = 280
inc_time = 4
noConcernVar = 4
zone1_i = 689
zone1_j = 145
res_data_path = "../Results/"
infer_times = np.arange(star_time, end_time, inc_time) 
error_u_sum = 0 ; error_v_sum = 0 ; error_p_sum = 0 
print(infer_times)

os.chdir(res_data_path)

# Read train times
with open('times4.pkl', 'rb') as input:
    train_times = pickle.load(input)[0]
print(train_times)

# Read in saved rom object
with open('rom-object4.pkl', 'rb') as input:
    read_rom = pickle.load(input)

# Read xy-coordinates 
pd_data = pd.read_csv('xy.csv', dtype='float64', delimiter=' ', header=None, skipinitialspace=True)
xydata = pd_data.values

# Read exact data
saved_npz = np.load(res_data_path+"Exact_all_data.npz")
UC = saved_npz['UC']
VC = saved_npz['VC']
PC = saved_npz['PC']

coeffs = read_rom.coeffsmat
u = read_rom.umat
mean_data = read_rom.mean_data
mean_tensor = tf.constant(mean_data, name="mean_data_tensor")

# Use Tensorflow 1.x 
for i, x in zip(range(total_steps), infer_times):
    print(i)
    ii= x-201
    hi_idx = [idx for idx,v in enumerate(train_times) if v > x][0]
    lo_idx = hi_idx - 1
    # Interpolate coefficients
    interp_coeffs = coeffs[lo_idx] + (coeffs[hi_idx]-coeffs[lo_idx])*(x-train_times[lo_idx])/(train_times[hi_idx]-train_times[lo_idx])
    print(interp_coeffs) 
    int_coeff_tensor = tf.Variable(interp_coeffs)
    print(int_coeff_tensor) 
    # Add a dim to make it a 2-D tensor
    int_coeff_tensor = tf.expand_dims(int_coeff_tensor, 0)
      
    # Compute the POD approximation
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result_op = tf.matmul(int_coeff_tensor, tf.transpose(u))
        modal_result = sess.run(tf.transpose(result_op))
        modal_result.flatten()
        mean_tensor.eval()
        result_op = tf.add_n([modal_result[:,0], mean_tensor])
        result = sess.run(result_op)
    result = result.reshape((int(modal_result.shape[0]/noConcernVar), noConcernVar))
    print(result[:,1], UC[:,ii])
    error_u = relative_error(result[:,1], UC[:,ii])
    error_v = relative_error(result[:,2], VC[:,ii])
    error_p = relative_error(result[:,3], PC[:,ii])
    print('Error u: %e, v: %e, p: %e' % (error_u, error_v, error_p))

    error_u_sum = error_u_sum + error_u
    error_v_sum = error_v_sum + error_v
    error_p_sum = error_p_sum + error_p

    height, width = result.shape
    p3d_result = np.hstack((xydata, result))
    # Save tecplot field data
    np.savetxt("./Case_flo_POD_t="+str(x)+".dat", p3d_result, delimiter=" ", header="variables = X, Y, rho, u, v, p \n zone i= "+str(zone1_i)+", j=    "+str(zone1_j), comments=' ')

error_u_mean = error_u_sum/total_steps
error_v_mean = error_v_sum/total_steps
error_p_mean = error_p_sum/total_steps
print('Mean Error u: %e, v: %e, p: %e' % (error_u_mean, error_v_mean, error_p_mean))


