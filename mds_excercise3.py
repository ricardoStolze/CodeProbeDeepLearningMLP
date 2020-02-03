import keras
import numpy as np
from keras.models import Model, Sequential
import tensorflow as tf
from keras.layers import Activation, Dense, Add, Input, Conv2D, MaxPooling2D, Dropout
from keras.backend import batch_flatten
import keras.backend as K
from keras.optimizers import Adam
from pdb import set_trace as st
from sklearn.metrics import f1_score

# load data
pat29_eeg1 = np.genfromtxt("data_csv/pat29/O2M1.csv")
pat29_eeg2 = np.genfromtxt("data_csv/pat29/C4M1.csv")
pat29_emg = np.genfromtxt("data_csv/pat29/EMG_100HZ.csv")
pat29_eog = np.genfromtxt("data_csv/pat29/REOGM1_150HZ.csv")
pat29_eog2 = np.genfromtxt("data_csv/pat29/LEOGM2_150HZ.csv")
pat29_leg = np.genfromtxt("data_csv/pat29/BeinLi_50HZ.csv")

pat75_eeg1 = np.genfromtxt("data_csv/pat75/O2M1.csv")
pat75_eeg2 = np.genfromtxt("data_csv/pat75/C4M1.csv")
pat75_emg = np.genfromtxt("data_csv/pat75/EMG_100HZ.csv")
pat75_eog = np.genfromtxt("data_csv/pat75/REOGM1_150HZ.csv")
pat75_eog2 = np.genfromtxt("data_csv/pat75/LEOGM2_150HZ.csv")
pat75_leg = np.genfromtxt("data_csv/pat75/BeinLi_50HZ.csv")

pat91_eeg1 = np.genfromtxt("data_csv/pat91/O2M1.csv")
pat91_eeg2 = np.genfromtxt("data_csv/pat91/C4M1.csv")
pat91_emg = np.genfromtxt("data_csv/pat91/EMG_100HZ.csv")
pat91_eog = np.genfromtxt("data_csv/pat91/REOGM1_150HZ.csv")
pat91_eog2 = np.genfromtxt("data_csv/pat91/LEOGM2_150HZ.csv")
pat91_leg = np.genfromtxt("data_csv/pat91/BeinLi_50HZ.csv")

pat80_eeg1 = np.genfromtxt("data_csv/pat80/O2M1.csv")
pat80_eeg2 = np.genfromtxt("data_csv/pat80/C4M1.csv")
pat80_emg = np.genfromtxt("data_csv/pat80/EMG_100HZ.csv")
pat80_eog = np.genfromtxt("data_csv/pat80/REOGM1_150HZ.csv")
pat80_eog2 = np.genfromtxt("data_csv/pat80/LEOGM2_150HZ.csv")
pat80_leg = np.genfromtxt("data_csv/pat80/BeiNLi_50HZ.csv")

pat89_eeg1 = np.genfromtxt("data_csv/pat89/O2M1.csv")
pat89_eeg2 = np.genfromtxt("data_csv/pat89/C4M1.csv")
pat89_emg = np.genfromtxt("data_csv/pat89/EMG_100HZ.csv")
pat89_eog = np.genfromtxt("data_csv/pat89/REOGM1_150HZ.csv")
pat89_eog2 = np.genfromtxt("data_csv/pat89/LEOGM2_150HZ.csv")
pat89_leg = np.genfromtxt("data_csv/pat89/BeinLi_50HZ.csv")

# remove timestamp  and downsample to 25HZ
pat29_eeg1 = pat29_eeg1[1::8]
pat29_eeg2 = pat29_eeg2[1::8]
pat29_emg = pat29_emg[1::4]
pat29_eog = pat29_eog[1::6]
pat29_eog2 = pat29_eog2[1::6]
pat29_leg = pat29_leg[1::2]

pat75_eeg1 = pat75_eeg1[1::8]
pat75_eeg2 = pat75_eeg2[1::8]
pat75_emg = pat75_emg[1::4]
pat75_eog = pat75_eog[1::6]
pat75_eog2 = pat75_eog2[1::6]
pat75_leg = pat75_leg[1::2]

pat91_eeg1 = pat91_eeg1[1::8]
pat91_eeg2 = pat91_eeg2[1::8]
pat91_emg = pat91_emg[1::4]
pat91_eog = pat91_eog[1::6]
pat91_eog2 = pat91_eog2[1::6]
pat91_leg = pat91_leg[1::2]

pat80_eeg1 = pat80_eeg1[1::8]
pat80_eeg2 = pat80_eeg2[1::8]
pat80_emg = pat80_emg[1::4]
pat80_eog = pat80_eog[1::6]
pat80_eog2 = pat80_eog2[1::6]
pat80_leg = pat80_leg[1::2]

pat89_eeg1 = pat89_eeg1[1::8]
pat89_eeg2 = pat89_eeg2[1::8]
pat89_emg = pat89_emg[1::4]
pat89_eog = pat89_eog[1::6]
pat89_eog2 = pat89_eog2[1::6]
pat89_leg = pat89_leg[1::2]

#safe downsampled data as npy files 
"""
np.save('data_npy_downsampled/pat89_eeg1.npy', pat89_eeg1)
np.save('data_npy_downsampled/pat89_eeg2.npy', pat89_eeg2)
np.save('data_npy_downsampled/pat89_eog.npy', pat89_eog)
np.save('data_npy_downsampled/pat89_emg.npy', pat89_emg)
np.save('data_npy_downsampled/pat89_eog2.npy', pat89_eog2)
np.save('data_npy_downsampled/pat89_leg.npy', pat89_leg)

np.save('data_npy_downsampled/pat75_eeg1.npy', pat75_eeg1)
np.save('data_npy_downsampled/pat75_eeg2.npy', pat75_eeg2)
np.save('data_npy_downsampled/pat75_eog.npy', pat75_eog)
np.save('data_npy_downsampled/pat75_emg.npy', pat75_emg)
np.save('data_npy_downsampled/pat75_eog2.npy', pat75_eog2)
np.save('data_npy_downsampled/pat75_leg.npy', pat75_leg)

np.save('data_npy_downsampled/pat91_eeg1.npy', pat91_eeg1)
np.save('data_npy_downsampled/pat91_eeg2.npy', pat91_eeg2)
np.save('data_npy_downsampled/pat91_eog.npy', pat91_eog)
np.save('data_npy_downsampled/pat91_emg.npy', pat91_emg)
np.save('data_npy_downsampled/pat91_eog2.npy', pat91_eog2)
np.save('data_npy_downsampled/pat91_leg.npy', pat91_leg)

np.save('data_npy_downsampled/pat80_eeg1.npy', pat80_eeg1)
np.save('data_npy_downsampled/pat80_eeg2.npy', pat80_eeg2)
np.save('data_npy_downsampled/pat80_eog.npy', pat80_eog)
np.save('data_npy_downsampled/pat80_emg.npy', pat80_emg)
np.save('data_npy_downsampled/pat80_eog2.npy', pat80_eog2)
np.save('data_npy_downsampled/pat80_leg.npy', pat80_leg)

np.save('data_npy_downsampled/pat29_eeg1.npy', pat29_eeg1)
np.save('data_npy_downsampled/pat29_eeg2.npy', pat29_eeg2)
np.save('data_npy_downsampled/pat29_eog.npy', pat29_eog)
np.save('data_npy_downsampled/pat29_emg.npy', pat29_emg)
np.save('data_npy_downsampled/pat29_eog2.npy', pat29_eog2)
np.save('data_npy_downsampled/pat29_leg.npy', pat29_leg)
""""

# load label
label1 = np.genfromtxt("data_csv/pat29/SleepStaging.csv", usecols = (2), dtype = str, delimiter = ',')[1:]
label2 = np.genfromtxt("data_csv/pat75/SleepStaging.csv", usecols = (2), dtype = str, delimiter = ',')[1:]
label5 = np.genfromtxt("data_csv/pat91/SleepStaging.csv", usecols = (2), dtype = str, delimiter = ',')[1:]
label3 = np.genfromtxt("data_csv/pat80/SleepStaging.csv", usecols = (2), dtype = str, delimiter = ',')[1:]
label4 = np.genfromtxt("data_csv/pat89/SleepStaging.csv", usecols = (2), dtype = str, delimiter = ',')[1:]

# combine sensors into one array per patient
pat1 = np.stack((pat29_eeg1, pat29_eeg2, pat29_eog, pat29_emg, pat29_eog2, pat29_leg))
pat2 = np.stack((pat75_eeg1, pat75_eeg2, pat75_eog, pat75_emg, pat75_eog2, pat75_leg))
pat3 = np.stack((pat80_eeg1, pat80_eeg2, pat80_eog, pat80_emg, pat80_eog2, pat80_leg))
pat4 = np.stack((pat89_eeg1, pat89_eeg2, pat89_eog, pat89_emg, pat89_eog2, pat89_leg))
pat5 = np.stack((pat91_eeg1, pat91_eeg2, pat91_eog, pat91_emg, pat91_eog2, pat91_leg))

# transfer labels from wk, rem , n1, n2, n3 to 0 1 2 3 4
label1[label1 == 'WK'] = 0
label1[label1 == 'REM'] = 1
label1[label1 == 'N1'] = 2
label1[label1 == 'N2'] = 3
label1[label1 == 'N3'] = 4

label2[label2 == 'WK'] = 0
label2[label2 == 'REM'] = 1
label2[label2 == 'N1'] = 2
label2[label2 == 'N2'] = 3
label2[label2 == 'N3'] = 4

label3[label3 == 'WK'] = 0
label3[label3 == 'REM'] = 1
label3[label3 == 'N1'] = 2
label3[label3 == 'N2'] = 3
label3[label3 == 'N3'] = 4

label4[label4 == 'WK'] = 0
label4[label4 == 'REM'] = 1
label4[label4 == 'N1'] = 2
label4[label4 == 'N2'] = 3
label4[label4 == 'N3'] = 4

label5[label5 == 'WK'] = 0
label5[label5 == 'REM'] = 1
label5[label5 == 'N1'] = 2
label5[label5 == 'N2'] = 3
label5[label5 == 'N3'] = 4


# sometimes size of data and number of labels don't fit, so we need to change them 
# pat1 1092 -> 1091 and 818275 -> 818250
# pat2 data fits
# pat3 1301 -> 1300 and 975450 -> 975000
# pat4 1174 -> 1173 and 879875 -> 879750
# pat5 1255 -> 1254 and 940850 -> 940500
# start times are maching, so end of data will be cut
label1 = label1[0:-1]
label3 = label3[0:-1]
label4 = label4[0:-1]
label5 = label5[0:-1]

pat1 = pat1[:,0:818250]
pat3 = pat3[:,0:975000]
pat4 = pat4[:,0:879750]
pat5 = pat5[:,0:940500]

# concatenating data as 30sec units
# so patches will have size 30sec * 25Hz x 6sensors = 750 x 6sensors (1st step)
# then concatenating sensor data to 750 * 6 = 4500 (2nd step)
pat1_reshape = pat1.reshape(6,750,1091).reshape(4500,1091)
pat2_reshape = pat2.reshape(6,750,1221).reshape(4500,1221)
pat3_reshape = pat3.reshape(6,750,1300).reshape(4500,1300)
pat4_reshape = pat4.reshape(6,750,1173).reshape(4500,1173)
pat5_reshape = pat5.reshape(6,750,1254).reshape(4500,1254)

data = np.concatenate((pat1_reshape,pat2_reshape,pat3_reshape,pat4_reshape,pat5_reshape),axis = 1)
label = np.concatenate((label1,label2,label3,label4,label5))


# use k-fold cross validation
# k = 3
# first randomize data
# then split into 3 parts of same size
indices = np.arange(label.size)
np.random.shuffle(indices)
data = data[:,indices]
label = label[indices]

data1 = data[:,0:int(6039/3)]
data2 = data[:, int(6039/3):int(6039/3)*2]
data3 = data[:, int(6039/3)*2:]
label1 = label[0:int(6039/3)] 
label2 = label[int(6039/3):int(6039/3)*2]
label3 = label[int(6039/3)*2:]

# create 3 test-data/training-data distributions
kfold_1_test_data = data1.swapaxes(0,1)
kfold_1_training_data = np.concatenate((data2,data3),axis = 1).swapaxes(0,1)
kfold_1_test_label = label1.astype('int64')
kfold_1_training_label = np.concatenate((label2,label3)).astype('int64')

kfold_2_test_data = data2.swapaxes(0,1)
kfold_2_training_data = np.concatenate((data1,data3), axis = 1).swapaxes(0,1)
kfold_2_test_label = label2.astype('int64')
kfold_2_training_label = np.concatenate((label1,label3)).astype('int64')

kfold_3_test_data = data3.swapaxes(0,1)
kfold_3_training_data = np.concatenate((data1,data2), axis = 1).swapaxes(0,1)
kfold_3_test_label = label3.astype('int64')
kfold_3_training_label = np.concatenate((label1,label2)).astype('int64')



# build up neural network

# kfold 1st run
model = Sequential()
model.add(Dense(units=512, activation='relu', input_dim=4500))
model.add(Dropout(0.2))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units=5, activation='softmax'))

model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])
model.fit(kfold_1_training_data,kfold_1_training_label,epochs=15, batch_size=256)
scores = model.evaluate(kfold_1_test_data, kfold_1_test_label)
print('f1: '+ str(scores[1]))
f1 = scores[1]

#kfold 2nd run
model = Sequential()
model.add(Dense(units=512, activation='relu', input_dim=4500))
model.add(Dropout(0.2))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units=5, activation='softmax'))

model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])
model.fit(kfold_2_training_data,kfold_2_training_label,epochs=15, batch_size=256)
scores = model.evaluate(kfold_2_test_data, kfold_2_test_label)
print('f1: '+ str(scores[1]))
f1 = f1 + scores[1]

# kfold 3rd run
model = Sequential()
model.add(Dense(units=512, activation='relu', input_dim=4500))
model.add(Dropout(0.2))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units=5, activation='softmax'))

model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])
model.fit(kfold_3_training_data,kfold_3_training_label,epochs=15, batch_size=256)
scores = model.evaluate(kfold_3_test_data, kfold_3_test_label)
print('f1: '+ str(scores[1]))
f1 = f1 + scores[1]
      
print('mean_f1: ' + str(f1/3))
