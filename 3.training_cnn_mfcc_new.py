## Training a CNN network for audio classification using MFCC features extracted from UrbanSound8K dataset
## Author: leopauly ! leopauly314@gmail.com


import os
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
#from sklearn.model_selection import train_test_split

'''
dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension (the depth) is at index 1,
in 'tf' mode is it at index 3. It defaults to the  image_dim_ordering value found in your
Keras config file at ~/.keras/keras.json. If you never set it, then it will be "tf".

print K.image_dim_ordering()
'''

#---------------------------------------------------------------------------------------------------------------#

np.random.seed(1337)  # for reproducibility
# Network parameters
data_w = 40
data_h = 40
n_classes = 10
n_filters_1 = 32
n_filters_2 = 64
d_filter = 3
p_drop_1 = 0.25
p_drop_2 = 0.50

# Traning parameters
p_test = 0.2
p_train = 1-p_test
batch_size = 32
nb_epoch = 50


#---------------------------------------------------------------------------------------------------------------#

def layer_info(model, i):
    print("INPUT # ", i)
    print("at", model.get_input_at(i))
    print("mask_at", model.get_input_mask_at(i))
    print("shape_at", model.get_input_shape_at(i))
    # print("shape_for", model.get_input_shape_for(i))
    print("OUTPUT # ", i)
    print("at", model.get_output_at(i))
    print("mask_at", model.get_output_mask_at(i))
    print("shape_at", model.get_output_shape_at(i))

#---------------------------------------------------------------------------------------------------------------#


## Getting data
print("Start loading data")
fd = open("/nobackup/leopauly/UrabanSound8k_Stored_mfcc/data_x_librosa_train_1.pkl", 'rb')
fd2 = open("/nobackup/leopauly/UrabanSound8k_Stored_mfcc/data_y_librosa_train_1.pkl", 'rb')
features = pickle.load(fd,encoding='bytes')
labels = pickle.load(fd2,encoding='bytes')
print("Data loaded")


#X_train,X_test,Y_train,Y_test=train_test_split(features, label, test_size=p_test)
rnd_indices = np.random.rand(len(labels)) < p_train
X_train = features[rnd_indices]
Y_train = labels[rnd_indices]
X_test = features[~rnd_indices]
Y_test = labels[~rnd_indices]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Fix for keras
Y_train = Y_train.reshape((-1, 1))
Y_test = Y_test.reshape((-1, 1))

print('Size of trainning data_x:',X_train.shape)
print('Size of trainning data_y:',Y_train.shape)
print('Size of test data_x:',X_test.shape)
print('Size of test data_y:',Y_test.shape)

#---------------------------------------------------------------------------------------------------------------#


## CNN model
model = Sequential()

# First layer
model.add(Convolution2D(n_filters_1, d_filter, d_filter, border_mode='valid', input_shape=(data_w, data_h, 3)))
model.add(Activation('relu'))

# Second layer
model.add(Convolution2D(n_filters_1, d_filter, d_filter))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Drop layer
model.add(Dropout(p_drop_1))

# Third layer
model.add(Convolution2D(n_filters_2, d_filter, d_filter, border_mode='valid'))
model.add(Activation('relu'))

# Fouth layer
model.add(Convolution2D(n_filters_2, d_filter, d_filter))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Drop layer
model.add(Dropout(p_drop_1))

## Used to flat the input (1, 10, 2, 2) -> (1, 40)
model.add(Flatten())

# Full Connected layer
model.add(Dense(256))
model.add(Activation('relu'))
# Drop layer
model.add(Dropout(p_drop_2))

# Output Full Connected layer
model.add(Dense(n_classes))
model.add(Activation('softmax'))

#---------------------------------------------------------------------------------------------------------------#

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit(X_train, Y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_split=0,shuffle=False)

#---------------------------------------------------------------------------------------------------------------#

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#---------------------------------------------------------------------------------------------------------------#

# serialize model to JSON
model_json = model.to_json()
with open("/nobackup/leopauly/S2LStage2/logdir_0/audio_classifier_mfcc_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/nobackup/leopauly/S2LStage2/logdir_0/audio_classifier_mfcc_1.h5")
print("Saved model to disk")

#---------------------------------------------------------------------------------------------------------------#





