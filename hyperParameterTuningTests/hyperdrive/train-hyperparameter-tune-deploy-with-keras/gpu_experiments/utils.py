# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import gzip
import numpy as np
import struct
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras import Input
from unet import *


# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res


# one-hot encode a 1-D array
def one_hot_encode(array, num_of_classes):
    return np.eye(num_of_classes)[array.reshape(-1)]

def build_model_gpu_mlp(n_inputs=None,n_h1=None,n_h2=None,learning_rate=None,n_outputs=10):
    #set up for gpu
    strategy=tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Build a simple MLP model
        model = Sequential()
        # first hidden layer
        model.add(Dense(n_h1, activation='relu', input_shape=(n_inputs,)))
        # second hidden layer
        model.add(Dense(n_h2, activation='relu'))
        # output layer
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=learning_rate),
                      metrics=['accuracy'])
        return model


def build_model_gpu_unet():
    #set up for gpu
    strategy=tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_tensor=Input((256,256,1))
        model=unet(input_tensor,maxpool=False)
        model.compile(optimizer=Adam(),loss='mse',metrics=['accuracy'])
        return model
    
    
    

def create_data(no_samples=100,dx=256,dy=256,dz=1):    
    #create training data.
    X=np.random.randn(no_samples,dx,dy,dz) #0 mean 1 std_dev
    #generate reference data (label/true data)
    Y=0.8*X+2   #0.8 std_dev 2 mean
    print(X.shape,Y.shape)
    #generate some validation data
    X_val=np.random.randn(no_samples,dx,dy,dz)
    Y_val=0.8*X_val+2
    print(X_val.shape,Y_val.shape)
    return (X,Y,X_val,Y_val)    