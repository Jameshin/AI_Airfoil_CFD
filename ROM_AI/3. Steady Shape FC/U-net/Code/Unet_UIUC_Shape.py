#"""
#@original author: Junghun Shin
#@modified by Myeongjun Song
#"""
import os
from pickle import NONE
import time
import random
import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from keras.callbacks import LearningRateScheduler
from CFDLib import mean_squared_error, relative_error

# Data path
data_path = "../../../Database/Simdata_airfoil_Steady_Shape/"
res_data_path = "../Results/"
noConcernVar = 4
zone1_i = 401
zone1_j = 81
glayer = 64 
cuttail = 20 

# Model parameters
BATCH_SIZE = 1050
EPOCHS = 6000
IMAGE_SHAPE = [zone1_i, zone1_j]
NB_CHANNELS = 1
LR = 1e-3

### Load Data
Re = np.array([1.0e5])
Mach = np.array([0.2])
AOA = np.array([1.0])
# list of file names
filenames = [] 
Nfoil = 1070
numd = Nfoil*Re.shape[0]*Mach.shape[0]*AOA.shape[0] 
Ncon = 0

#read xy-coordinates
saved_npz = np.load(res_data_path) 
XC_star = saved_npz['XC']
YC_star = saved_npz['YC']
xydata = np.hstack((XC_star[:,0][:,None], YC_star[:,0][:,None]))

#Shape vector
idx_bottom = np.where(xydata[:,0] == xydata[1,0])[0]     
i = 1
for i in range(1,idx_bottom[1]):
    if(xydata[i,1] != xydata[idx_bottom[1]-i+1,1]):
        break
idx_tip = [i-1, idx_bottom[1]-i+2]
idx_x_sur = np.arange(idx_tip[0],idx_tip[1]+1)
idx_x_bd1 = np.arange(1, idx_tip[0]+1)
idx_x_bd2 = np.arange(idx_tip[1], idx_bottom[1]+1)

#Data Extration 
DC_star = saved_npz['DC']
UC_star = saved_npz['UC']
VC_star = saved_npz['VC']
PC_star = saved_npz['PC']

XI_star = np.reshape(XC_star, [numd, glayer, zone1_i-2*cuttail])[0:BATCH_SIZE,:,:]
XI_field = XI_star[:,:,:-1] 
YI_star = np.reshape(YC_star, [numd, glayer, zone1_i-2*cuttail])[0:BATCH_SIZE,:,:]
YI_field = YI_star[:,:,:-1] 
XT_star = np.reshape(XC_star, [numd, glayer, zone1_i-2*cuttail])[BATCH_SIZE:numd,:,:]
XT_field = XT_star[:,:,:-1] 
YT_star = np.reshape(YC_star, [numd, glayer, zone1_i-2*cuttail])[BATCH_SIZE:numd,:,:] 
YT_field = YT_star[:,:,:-1] 
UI_star = np.reshape(UC_star, [numd, glayer, zone1_i-2*cuttail])[0:BATCH_SIZE,:,:]
UC_field = UI_star[:,:,:-1] 
VI_star = np.reshape(VC_star, [numd, glayer, zone1_i-2*cuttail])[0:BATCH_SIZE,:,:]
VC_field = VI_star[:,:,:-1] 
PI_star = np.reshape(PC_star, [numd, glayer, zone1_i-2*cuttail])[0:BATCH_SIZE,:,:]
PC_field = PI_star[:,:,:-1] 
UT_star = np.reshape(UC_star, [numd, glayer, zone1_i-2*cuttail])[BATCH_SIZE:numd,:,:]
UT_field = UT_star[:,:,:-1] 
VT_star = np.reshape(VC_star, [numd, glayer, zone1_i-2*cuttail])[BATCH_SIZE:numd,:,:]
VT_field = VT_star[:,:,:-1] 
PT_star = np.reshape(PC_star, [numd, glayer, zone1_i-2*cuttail])[BATCH_SIZE:numd,:,:]
PT_field = PT_star[:,:,:-1] 

Input_field = np.stack([XI_field, YI_field], axis=3)
Test_field = np.stack([XT_field, YT_field], axis=3)
Field_star = np.stack([UC_field, VC_field, PC_field], axis=3)

@tf.function
def dataloader(paths):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)
    return dataset

datasetI = dataloader(Input_field)
train_inp = []
for batch in datasetI.take(1):
    for i, img_inp in enumerate(batch):
        img_inp_np = img_inp.numpy() 
        train_inp.insert(i, img_inp_np)
train_inp = np.array(train_inp, dtype="float32")

datasetO = dataloader(Field_star)
train_env = []
for batch in datasetO.take(1):
    for i, img in enumerate(batch):
        img_np = img.numpy() 
        train_env.insert(i, img_np)
train_env = np.array(train_env, dtype="float32")

datasetT = dataloader(Test_field)
train_test = []
for batch in datasetT.take(1):
    for i, img_test in enumerate(batch):
        img_test_np = img_test.numpy() 
        train_test.insert(i, img_test_np)
train_test = np.array(train_test, dtype="float32")

# Build CNN network
input_e = tf.keras.Input(shape=(glayer, zone1_i-2*cuttail-1, 2))
pooling_size = 2 
n_ch = 12      
conv1 = layers.Conv2D(n_ch, (3,3), activation='elu', padding = 'same')(input_e)
mp1 = layers.MaxPooling2D((pooling_size,pooling_size))(conv1)

conv2 = layers.Conv2D(n_ch*2, (3,3), activation='elu', padding = 'same')(mp1)
mp2 = layers.MaxPooling2D((pooling_size,pooling_size))(conv2)

conv3 = layers.Conv2D(n_ch*4, (3,3), activation='elu', padding = 'same')(mp2)
mp3 = layers.MaxPooling2D((pooling_size,pooling_size))(conv3)
'''     
conv4 = layers.Conv2D(n_ch*8, (3,3), activation='elu', padding = 'same')(mp3)
mp4 = layers.MaxPooling2D((pooling_size,pooling_size))(conv4)

output_e = layers.Conv2D(n_ch*8, (3,3), activation='elu', padding = 'same')(mp3)

flat1 = layers.Flatten()(mp2)
dense1 = layers.Dense(LATENT_DEPTH, activation='elu')(flat1)
time = tf.keras.Input(shape=(1,))
dense2 = layers.Concatenate(axis=1)([time, dense1]) 
print(dense2.shape)
dense3 = layers.Dense(LATENT_DEPTH+1, activation='elu')(dense2)
dense4 = layers.Dense(LATENT_DEPTH, activation='elu')(dense3)
dense5 = layers.Dense(n_ch*2*(zone1_i-2*cuttail-1)/pooling_size/pooling_size*(glayer-2)/pooling_size/pooling_size, activation='elu')(dense4)
output_f = layers.Reshape((int((glayer-2)/pooling_size/pooling_size), int((zone1_i-2*cuttail-1)/pooling_size/pooling_size), n_ch*2))(dense5)
'''
convt1 = layers.Conv2DTranspose(n_ch*4, (3,3), activation='elu', padding='same')(output_e) #512아니고 256
upsamp1 = layers.UpSampling2D((pooling_size,pooling_size))(convt1)
skipcon1 = layers.Concatenate(axis=3)([conv3, upsamp1])
conv6 = layers.Conv2D(n_ch*4, (3,3), activation = 'elu', padding='same')(skipcon1)

convt2 = layers.Conv2DTranspose(n_ch*2, (3,3), activation='elu', padding='same')(conv6)
upsamp2 = layers.UpSampling2D((pooling_size,pooling_size))(convt2)
skipcon2 = layers.Concatenate(axis=3)([conv2, upsamp2])

conv7 = layers.Conv2D(n_ch*2, (3,3), activation = 'elu', padding='same')(skipcon2)
convt3 = layers.Conv2DTranspose(n_ch*1, (3,3), activation='elu', padding='same')(conv7)
upsamp3 = layers.UpSampling2D((pooling_size,pooling_size))(convt3)
skipcon3 = layers.Concatenate(axis=3)([conv1, upsamp3])

conv8 = layers.Conv2D(n_ch*1, (3,3), activation='elu', padding='same')(skipcon3)

'''
convt4 = layers.Conv2DTranspose(n_ch, (3,3), activation='elu', padding='same')(conv8)
upsamp4 = layers.UpSampling2D((pooling_size,pooling_size))(convt4)
skipcon4 = layers.Concatenate(axis=3)([conv1, upsamp4])
conv9 = layers.Conv2D(n_ch, (3,3), activation='elu', padding='same')(skipcon4)
'''
output_d = layers.Conv2DTranspose(3, (3,3), activation='elu', padding='same')(conv8)

# Loss
def custom_mse(idx_x_bd1, idx_x_bd2): # 
    def loss(y_true,y_pred):
        # Extract boundary values
        train_bd1 = y_pred[:,0:1,:,:][:,:,idx_tip[0]:1:-1,:] #idx_x_bd1[1:]
        train_bd2 = y_pred[:,0:1,:,:][:,:,idx_tip[1]:idx_bottom[1],:] #idx_x_bd2[:-1]
        #print(train_bd1.shape,train_bd2.shape)
        # calculating squared difference between target and predicted values 
        loss1 = tf.keras.backend.square(y_true - y_pred) # (batch_size, 3)
        loss2 = tf.keras.backend.square(train_bd1 - train_bd2)    
        # multiplying the values with weights along batch dimension
        #loss = loss * [0.3, 0.7]          # (batch_size, 2)    
        # summing both loss values along batch dimension 
        loss1 = tf.keras.backend.mean(tf.keras.backend.sum(loss1, axis=1)) # (batch_size,)
        loss2 = tf.keras.backend.mean(tf.keras.backend.sum(loss2, axis=1)) 
        print(loss1.shape, loss1.shape)       
        return loss1 + loss2
    return loss

# Train model
unet = Model(inputs=input_e, outputs=output_d)
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LR,
    decay_steps=300,
    decay_rate=0.9)
reduce_lr = LearningRateScheduler(lr_schedule)
optimizer = optimizers.Adam(learning_rate=lr_schedule)
unet.compile(optimizer='adam', loss=custom_mse(idx_x_bd1,idx_x_bd2), metrics=["mse"]) 
unet.fit(train_inp, train_env, validation_split=0.1, epochs=EPOCHS, verbose=2, callbacks =[reduce_lr])
unet.save(res_data_path+'my_model.h5')

# Load saved model from checkpoint directory
#unet = tf.keras.models.load_model(res_data_path+'my_model.h5', compile=False)
decoded_imgs = unet([train_test]).numpy() #,time_test

for i in range(20):
    x_star = XT_field[i].flatten()[:,None]
    y_star = YT_field[i].flatten()[:,None]    
    u_pred = decoded_imgs[i,:,:,0].flatten()[:,None]
    v_pred = decoded_imgs[i,:,:,1].flatten()[:,None]
    p_pred = decoded_imgs[i,:,:,2].flatten()[:,None]
    
    p3d_result = np.hstack((x_star, y_star, u_pred, v_pred, p_pred)) 
    
    np.savetxt(res_data_path+"Case_flo_unet_UIUC_n="+str(i).rjust(2,'0')+".dat", p3d_result, delimiter=" ", header="variables = X, Y, u, v, p \n zone i="+str(zone1_i-2*cuttail-1)+" j="+str(glayer)+" ", comments=' ')
    # Error
    error_u = relative_error(u_pred[51:310], UT_field[i,:,:].flatten()[:,None][51:310])
    error_v = relative_error(v_pred[51:310], VT_field[i,:,:].flatten()[:,None][51:310])
    error_p = relative_error(p_pred[51:310], PT_field[i,:,:].flatten()[:,None][51:310])*100
    print('Error u: %e, v: %e, p: %e' % (error_u, error_v, error_p))
    err_result = np.hstack((x_star, y_star, abs((u_pred-UT_field[i,:,:].flatten()[:,None])), abs((v_pred-VT_field[i,:,:].flatten()[:,None])), abs((p_pred-PT_field[i,:,:].flatten()[:,None]))))
  
    np.savetxt(res_data_path+"Error_unet_UIUC_n="+str(i).rjust(2,'0')+".dat", err_result, delimiter=" ", header="variables = X, Y, eu, ev, ep \n zone i="+str(zone1_i-2*cuttail-1)+" j="+str(glayer)+" ", comments=' ')
    Cx = XT_field[i].flatten()[:][:,None]
    Cy = YT_field[i].flatten()[:][:,None]
    Cp_pred = decoded_imgs[i,:,:,2].flatten()[:][:,None]
    Cp_true = PT_field[i,:,:].flatten()[:][:,None]
    Cp_result = np.hstack((Cx, Cy, Cp_pred, Cp_true, Cp_true-Cp_pred))
    np.savetxt(res_data_path+"Cp_unet_UIUC_n="+str(i).rjust(2,'0')+".dat", Cp_result, delimiter=" ", header="variables = X, Y, Cp_pred, Cp_true, Error \n zone i="+str(zone1_i-2*cuttail-1)+" j="+str(glayer)+" ", comments=' ')
    