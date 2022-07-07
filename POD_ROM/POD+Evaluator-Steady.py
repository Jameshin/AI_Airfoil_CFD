import RomObject
import tensorflow.compat.v1 as tf
import numpy as np
import pickle
import pandas as pd
import itertools

tf.compat.v1.disable_eager_execution()

#perform coefficient interpolation here, using numpy for it
#input_para = [[1.0e+05, 4.0e-01, 6.0e+00], [1.0e+05, 4.0e-01, 7.0e+00], [1.0e+05, 4.0e-01, 8.0e+00], [1.0e+05, 4.0e-01, 9.0e+00], [1.0e+05, 4.0e-01, 1.0e+01], [1.0e+05, 4.5e-01, 6.0e+00], [1.0e+05, 4.5e-01, 7.0e+00], [1.0e+05, 4.5e-01, 8.0e+00], [1.0e+05, 4.5e-01, 9.0e+00], [1.0e+05, 4.5e-01, 1.0e+01], [1.0e+05, 5.0e-01, 6.0e+00], [1.0e+05, 5.0e-01, 7.0e+00], [1.0e+05, 5.0e-01, 8.0e+00], [1.0e+05, 5.0e-01, 9.0e+00], [1.0e+05, 5.0e-01, 1.0e+01], [1.0e+05, 5.5e-01, 6.0e+00], [1.0e+05, 5.5e-01, 7.0e+00], [1.0e+05, 5.5e-01, 8.0e+00], [1.0e+05, 5.5e-01, 9.0e+00], [1.0e+05, 5.5e-01, 1.0e+01], [1.0e+05, 6.0e-01, 6.0e+00], [1.0e+05, 6.0e-01, 7.0e+00], [1.0e+05, 6.0e-01, 8.0e+00], [1.0e+05, 6.0e-01, 9.0e+00], [1.0e+05, 6.0e-01, 1.0e+01], [2.0e+05, 4.0e-01, 6.0e+00], [2.0e+05, 4.0e-01, 7.0e+00], [2.0e+05, 4.0e-01, 8.0e+00], [2.0e+05, 4.0e-01, 9.0e+00], [2.0e+05, 4.0e-01, 1.0e+01], [2.0e+05, 4.5e-01, 6.0e+00], [2.0e+05, 4.5e-01, 7.0e+00], [2.0e+05, 4.5e-01, 8.0e+00], [2.0e+05, 4.5e-01, 9.0e+00], [2.0e+05, 4.5e-01, 1.0e+01], [2.0e+05, 5.0e-01, 6.0e+00], [2.0e+05, 5.0e-01, 7.0e+00], [2.0e+05, 5.0e-01, 8.0e+00], [2.0e+05, 5.0e-01, 9.0e+00], [2.0e+05, 5.0e-01, 1.0e+01], [2.0e+05, 5.5e-01, 6.0e+00], [2.0e+05, 5.5e-01, 7.0e+00], [2.0e+05, 5.5e-01, 8.0e+00], [2.0e+05, 5.5e-01, 9.0e+00], [2.0e+05, 5.5e-01, 1.0e+01], [2.0e+05, 6.0e-01, 6.0e+00], [2.0e+05, 6.0e-01, 7.0e+00], [2.0e+05, 6.0e-01, 8.0e+00], [2.0e+05, 6.0e-01, 9.0e+00], [2.0e+05, 6.0e-01, 1.0e+01]]
#input_para = [[1.0e+05, 4.5e-01, 6.0e+00], [1.0e+05, 4.5e-01, 7.0e+00], [1.0e+05, 4.5e-01, 8.0e+00], [1.0e+05, 4.5e-01, 9.0e+00], [1.0e+05, 4.5e-01, 1.0e+01], [1.0e+05, 5.5e-01, 6.0e+00], [1.0e+05, 5.5e-01, 7.0e+00], [1.0e+05, 5.5e-01, 8.0e+00], [1.0e+05, 5.5e-01, 9.0e+00], [1.0e+05, 5.5e-01, 1.0e+01], [2.0e+05, 4.5e-01, 6.0e+00], [2.0e+05, 4.5e-01, 7.0e+00], [2.0e+05, 4.5e-01, 8.0e+00], [2.0e+05, 4.5e-01, 9.0e+00], [2.0e+05, 4.5e-01, 1.0e+01], [2.0e+05, 5.5e-01, 6.0e+00], [2.0e+05, 5.5e-01, 7.0e+00], [2.0e+05, 5.5e-01, 8.0e+00], [2.0e+05, 5.5e-01, 9.0e+00], [2.0e+05, 5.5e-01, 1.0e+01]]
Re = np.array([1.0e5, 2.0e5, 3.0e5])
Mach = np.array([0.4, 0.45, 0.5, 0.55, 0.6])
AOA = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
input_para = list(itertools.product(*[Re,Mach,AOA]))

total_steps = len(input_para)
Npara = len(input_para[0])
noConcernVar = 4
zone1_i = 401
zone1_j = 81
sim_data_path = "./PODresults_LI_0/"

#READ IN THE DESIGN INFORMATION

with open('designs678910.pkl', 'rb') as input:
    read_para= pickle.load(input)
read_paras = np.array(read_para)
#read in saved rom object
with open('rom-object678910.pkl', 'rb') as input:
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

print(coeffs.shape, u.shape, mean_data.shape)

# In this small case, CPU without GPU and use numpy

for i, x in zip(range(total_steps), input_para):
    #print(i)
    hi_para = np.zeros(Npara)
    lo_para = np.zeros(Npara)
    for k in range(Npara):
        print(k)
        hi_para[k] = min([v for idx_h, v in enumerate(read_paras[:,k]) if v >= x[k]])
        lo_para[k] = max([w for idx_l, w in enumerate(read_paras[:,k]) if w <= x[k]])
    hi_idx = np.where((read_paras == np.array(hi_para)).all(axis=1))[0]
    lo_idx = np.where((read_paras == np.array(lo_para)).all(axis=1))[0]
    print(hi_para, lo_para, hi_idx, lo_idx, '=========')
    #interpolate coefficients
    if hi_idx != lo_idx:
        interp_coeffs = coeffs[lo_idx] + (coeffs[hi_idx]-coeffs[lo_idx])*np.sqrt(np.sum((x-read_paras[lo_idx])**2))/np.sqrt(np.sum((read_paras[hi_idx]-read_paras[lo_idx])**2))
    else:
        interp_coeffs = coeffs[lo_idx]
    int_coeff_tensor = tf.Variable(interp_coeffs)
    #add a dim to make it a 2-D tensor
    #int_coeff_tensor = tf.expand_dims(int_coeff_tensor, 0)
    print(interp_coeffs.shape, "++++++++++")
    #compute the POD approximation
    with tf.device('/gpu:1'):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result_op = tf.matmul(int_coeff_tensor, tf.transpose(u) )
            modal_result = sess.run(tf.transpose(result_op))
            modal_result.flatten()
            #mean_tensor.eval()
            sess.run(mean_tensor)
            result_op = tf.add(modal_result, mean_tensor)
            result = sess.run(result_op)
        result = result.reshape(-1, noConcernVar)
        height, width = result.shape
        #Create p3d field data
        p3d_result = np.hstack((xydata, result))
        #Save tecplot field data
        np.savetxt(sim_data_path+"./Case_flo_POD_Con="+str(i)+".dat", p3d_result, delimiter=" ", header="variables = X, Y, rho, u, v, p \n zone i= "+str(zone1_i)+", j="+str(zone1_j), comments=' ')
#np.savetxt("./Case_flo_POD.dat", p3d_result, delimiter=" ", header="variables = X, Y, rho, u, v, p \n zone i= "+str(zone1_i)+", j= "+str(zone1_j), comments=' ')
