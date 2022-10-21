#"""
#@original author: Junghun Shin
#@modified by Wontae Hwang
#"""
import tensorflow.compat.v1 as tf
import numpy as np
import GPy
import pandas as pd
from CFDLib import mean_squared_error, relative_error
import matplotlib.pyplot as plt

# tensorflow v2 off
tf.disable_v2_behavior()

Totnum = 135
numd = 123
numT = Totnum - numd
n_var = 2
zone1_i = 401
zone1_j = 81
cuttail = 20 
glayer = 64 
sim_data_path = "../../../Database/Simdata_airfoil_Steady_FC/"
res_data_path = "../Results/"

# Read train data
saved_npz = np.load(res_data_path+"Staedy_airfoil_cuttail_train.npz")
snapshot_data4 = saved_npz['DC']
snapshot_data1 = saved_npz['UC']
snapshot_data2 = saved_npz['VC']
snapshot_data3 = saved_npz['PC']
snapshot_data = np.vstack((snapshot_data1,snapshot_data2,snapshot_data3))
xy1 = saved_npz['XC']
xy2 = saved_npz['YC']
xydata=np.column_stack((xy1[:,0],xy2[:,0]))

# Read test data
saved_npz_e = np.load(res_data_path+"Staedy_airfoil_cuttail_test.npz")
U_true = saved_npz_e['UC']
V_true = saved_npz_e['VC']
P_true = saved_npz_e['PC']

# Mean center the data
mean_array = None
mean_data_tensor = np.mean(snapshot_data, axis=1)
mean_centered_data = np.subtract(snapshot_data, np.tile(mean_data_tensor, (numd,1)).T)

# Singular Value Decomposition 
u, s, v = np.linalg.svd(mean_centered_data, compute_uv=True, full_matrices=False)
print(mean_centered_data.shape)
print(u.shape)
print(v.shape)
print(s.shape)

# POD coefficients
compute_coeffs = np.matmul(np.transpose(mean_centered_data),u)
e = np.sum(s)
s_energy = np.divide(s,e)*100
y=s_energy
coeffs = compute_coeffs
mean_data = mean_data_tensor
mean_tensor = tf.constant(mean_data, name="mean_data_tensor")

## Make label
Nm = 1 ; Na = 1
k = 0 ; kk = 0
train_x = np.zeros((numd,2))
test_x = np.zeros((numT,2))
train_ipt_M = np.zeros(numd)
test_ipt_M = np.zeros(numT)
train_ipt_A = np.zeros(numd)
test_ipt_A = np.zeros(numT)
for i in range (Totnum):
    if Nm%2 == 0 and (Na+2)%5 == 0:
        test_ipt_M[k] = Nm
        test_ipt_A[k] = Na
        k += 1
    else:
        train_ipt_M[kk] = Nm
        train_ipt_A[kk] = Na
        kk += 1
        
    if Na%15 == 0:
        Nm += 1
        Na = 0
    Na += 1

train_x[:,0] = train_ipt_M
train_x[:,1] = train_ipt_A
test_x[:,0] = test_ipt_M
test_x[:,1] = test_ipt_A

# Run GPR
ker = GPy.kern.Matern52(n_var, ARD=True)               # dimension of x data = n_DV
m = GPy.models.GPRegression(train_x,coeffs,ker)
m.optimize(messages=True, max_f_eval=1000)

x_pred = np.zeros((1,2))
error_u_sum = 0 ; error_v_sum = 0 ; error_p_sum = 0 
# Use Tensorflow 1.x 
for i in range(numT):
    x_pred[0] = test_x[i]
    #interpolate coefficients
    interp_coeffs = m.predict(x_pred)[0][0]
    int_coeff_tensor = tf.Variable(interp_coeffs)
    #add a dim to make it a 2-D tensor
    int_coeff_tensor = tf.expand_dims(int_coeff_tensor, 0)     
    #compute the POD approximation
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result_op = tf.matmul(int_coeff_tensor, tf.transpose(u))
        modal_result = sess.run(tf.transpose(result_op))
        modal_result.flatten()
        mean_tensor.eval()
        result_op = tf.add_n([modal_result[:,0], mean_tensor])
        result = sess.run(result_op)
    U_pred = result[:23104]
    V_pred = result[23104:46208]
    P_pred = result[46208:]
    error_u = relative_error(U_pred, U_true[:,i])
    error_v = relative_error(V_pred, V_true[:,i])
    error_p = relative_error(P_pred, P_true[:,i])
    print('Error u: %e, v: %e, p: %e' % (error_u, error_v, error_p))

    error_u_sum = error_u_sum + error_u
    error_v_sum = error_v_sum + error_v
    error_p_sum = error_p_sum + error_p

    #Create p3d field data
    p3d_result = np.column_stack((xydata, U_pred, V_pred, P_pred))
    print(xydata.shape, result.shape)
    #Save tecplot field data
    np.savetxt(res_data_path+"UIUC_GPR_"+str(int(x_pred[0,0]))+"_"+str(int(x_pred[0,1]))+".dat", p3d_result, delimiter=" ", header="variables = X, Y, u, v, p \n zone i="+str(zone1_i-2*cuttail)+" j="+str(glayer)+" ", comments=' ')
    
    Cp_pred = result[46208:]
    Cp_true = P_true[:,i]
    Cp_result = np.column_stack((xydata, Cp_pred, Cp_true, Cp_true-Cp_pred))
    np.savetxt(res_data_path+"Cp_POD-GPR_UIUC_n"+str(int(x_pred[0,0]))+"_"+str(int(x_pred[0,1]))+".dat", Cp_result, delimiter=" ", header="variables = X, Y, Cp_pred, Cp_true, Error \n zone i="+str(zone1_i-2*cuttail)+" j="+str(glayer)+" ", comments=' ')

error_u_mean = error_u_sum/numT
error_v_mean = error_v_sum/numT
error_p_mean = error_p_sum/numT
print('Mean Error u: %e, v: %e, p: %e' % (error_u_mean, error_v_mean, error_p_mean))