#"""
#@original author: Laura Kulowski
#@modified by Myeongjun Song
#"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib import reload
import torch
import Generate_windowed_dataset
import lstm_encoder_decoder
from POD_SVD_Unsteady import mean_array3, xy, BF, snapshot_data, MC, t
matplotlib.rcParams.update({'font.size': 17})

res_data_path2 = '../Results/'
mode_num = 80
zone1_i = 689
zone1_j = 145
noConcernVar = 4

# make Windowded dataset
t_train, y_train, t_test, y_test = Generate_windowed_dataset.train_test_split(t, MC, split = 0.7)
iw = 13
ow = 2 
s = 1
Xtrain, Ytrain= Generate_windowed_dataset.windowed_dataset(y_train, input_window = iw, output_window = ow, stride = s,num_features = mode_num)
Xtest, Ytest = Generate_windowed_dataset.windowed_dataset(y_test, input_window = iw, output_window = ow, stride = s,num_features = mode_num)
print(Ytest.shape)
X_train, Y_train, X_test, Y_test = Generate_windowed_dataset.numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest)

# convert windowed data from np.array to PyTorch tensor
model = lstm_encoder_decoder.lstm_seq2seq(input_size = mode_num, hidden_size = 8)
loss = model.train_model(X_train, Y_train, n_epochs = 100, target_len = ow, batch_size = 1, training_prediction = 'mixed_teacher_forcing', teacher_forcing_ratio = 0.6, learning_rate = 0.01, dynamic_tf = False)
Y_test_pred = np.zeros((Ytest.shape),dtype=np.double)
Y_train_pred = np.zeros((Ytest.shape),dtype=np.double)
num_rows = Ytest.shape[1]
for ii in range(num_rows):
      X_train_plt = Xtrain[:, ii, :]
      Y_train_pred[:,ii,:] = model.predict(torch.from_numpy(X_train_plt).type(torch.Tensor), target_len = ow)
      X_test_plt = Xtest[:, ii, :]
      Y_test_pred[:,ii,:] = model.predict(torch.from_numpy(X_test_plt).type(torch.Tensor), target_len = ow)     
pred_range = np.arange(70,81,2)
MC_re = np.zeros((80,80))
for i in pred_range: 
    if i == 80:
        MC_re[i-1,:] = Y_test_pred[1,i-71,:]
    else:
        MC_re[i-1,:] = Y_test_pred[0,i-70,:]

# Reconstruct flowfield from the preicted Modal Coefficient    
shp2 = mean_array3.reshape(-1,noConcernVar).shape
result=np.zeros((80,shp2[1],shp2[0]),dtype=np.double)
dif = np.zeros((shp2[1],shp2[0] ),dtype=np.double)
MC_ad = MC_re.T
for i in pred_range:  
    recons=np.matmul(BF[:,:mode_num],MC_ad[:mode_num,i-1])
    recons = (np.add(recons,mean_array3)).reshape(-1,noConcernVar)
    result[i-1,:,:] = recons.T
    res = np.hstack((xy, recons))
    np.savetxt(res_data_path2+"Reconsruction_"+str(i)+".dat", res, delimiter=" ", header="variables = X, Y, rh, u, v, p \n zone i="+str(zone1_i)+", j="+str(zone1_j), comments=' ')
    input_temp = snapshot_data[:,i-1]
    diff = recons - input_temp.reshape(-1,noConcernVar)
    differ = np.hstack((xy, diff))
    np.savetxt(res_data_path2+'Difference_'+str(i)+'.dat', differ, delimiter=" ", header="variables = X, Y, rh, u, v, p \n zone i="+str(zone1_i)+", j="+str(zone1_j), comments=' ')

# Print Relative Error
for i in range(pred_range.shape[0]):
    error_temp = np.zeros([noConcernVar])
    for j in range(noConcernVar):
        n = result.shape[2]
        input = snapshot_data[:,pred_range[i]-1].reshape(-1,noConcernVar)
        res = 0.
        den = 0.
        for k in range(n):
            res += (result[pred_range[i]-1,j,k]-input[k,j])**2
            den += (input[k,j])**2
        error_temp[j] = np.sqrt(res/den)
    print('Relative error for %dth dataset rho: %f u: %f v: %f p: %f \n ' %(pred_range[i],
                    error_temp[0],error_temp[1],error_temp[2],error_temp[3]) )

        





