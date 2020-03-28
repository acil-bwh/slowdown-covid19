
from __future__ import division
from future.utils import iteritems
import os
import logging
import argparse
import datetime
import json
import pickle
import time
import warnings
import numpy as np
import h5py
from skimage.transform import resize
from sklearn.metrics import roc_auc_score  # roc_curve, auc,

import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input,BatchNormalization,Conv2D,MaxPooling2D,Dense,Add,Multiply
from tensorflow.keras.layers import ReLU,GlobalMaxPooling2D
from tensorflow.keras.models import Model
from keras.utils.io_utils import HDF5Matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Lambda,Multiply
from tensorflow.keras import backend as K
from tensorflow.math import divide

import pandas as pd

import math

import h5py

class SlowdownCOVID19QCEngine():
    def __init__(self,output_folder):
        self.input_shape=(1024,1024,3)
        self.output_folder=output_folder

    def Model1(self):
        input_shape=self.input_shape
        model = models.Sequential()
        model.add(layers.Conv2D(16, (3, 3), activation='relu',
                                input_shape=input_shape))
        model.add(layers.Conv2D(16, (3, 3), activation='relu',
                                input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        if input_shape[0] > 64:
            model.add(layers.Conv2D(256, (3, 3), activation='relu'))
            model.add(layers.Conv2D(256, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))

        return model

    def Model2(self):
        input_shape=self.input_shape
        conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape)
        conv_base.trainable = False
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2, activation='softmax'))
        return model

    def conv_block(self,input_layer, n_filters, length=2, pool=True, stride=1):

        layer = input_layer
        for i in range(length):
            layer = Conv2D(n_filters, (3, 3), strides=stride, padding='same')(layer)
            layer = BatchNormalization()(layer)
            layer = ReLU()(layer)

        parallel = Conv2D(n_filters, (1, 1), strides=stride ** length, padding='same')(input_layer)
        parallel = BatchNormalization()(parallel)
        parallel = ReLU()(parallel)

        output = Add()([layer, parallel])
        
        #output = BatchNormalization()(output)

        # output=Multiply()([output,K.variable(0.5,shape=K.shape(output),dtype='float64',name='const')])
        # output=Multiply()([output,K.ones(shape=K.shape(output))])
        output=Lambda(lambda x: divide(x, 2.0))(output)

        if pool:
            output = MaxPooling2D(pool_size=(3, 3), strides=2)(output)

        return output
    def Model3(self,width=1):
        # Model implementation based on:
        # https://www.nature.com/articles/s41598-019-42557-4
        input_shape=self.input_shape
        inputs = Input(input_shape)

        output = self.conv_block(inputs, n_filters=16 * width, stride=2)
        output = self.conv_block(output, n_filters=32 * width)
        output = self.conv_block(output, n_filters=48 * width)
        output = self.conv_block(output, n_filters=64 * width)

        output = self.conv_block(output, n_filters=80 * width, pool=False)

        # Global Average Pooling
        output = GlobalMaxPooling2D()(output)

        # Dense
        output = Dense(512, activation='relu')(output)
        output = Dense(2, activation='softmax')(output)

        model = Model(outputs=output, inputs=inputs)

        return model

    def GenerateData(self,input_dir,output_h5,num_training = 6000,num_validation = 1200):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=5,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.1,
                                           zoom_range=0.2,
                                           horizontal_flip=True, validation_split=0.2)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        batch_size = 1

        input_shape = self.input_shape

        train_generator = train_datagen.flow_from_directory(
            input_dir,
            target_size=input_shape[0:2],
            batch_size=batch_size,
            class_mode='binary',
            subset='training')  # set as training data

        validation_generator = train_datagen.flow_from_directory(
            input_dir,  # same directory as training data
            target_size=input_shape[0:2],
            batch_size=batch_size,
            class_mode='binary',
            subset='validation')  # set as validation data

        ff = h5py.File(output_h5, 'w')
        X_train = ff.create_dataset("X_train", (num_training,) + input_shape, dtype='float32')
        y_train = ff.create_dataset("y_train", (num_training,), dtype='float32')

        X_val = ff.create_dataset("X_val", (num_validation,) + input_shape, dtype='float32')
        y_val = ff.create_dataset("y_val", (num_validation,), dtype='float32')

        for ii in range(num_training):
            _tmp = train_generator.next()
            X_train[ii] = _tmp[0]
            y_train[ii] = _tmp[1]
        for ii in range(num_validation):
            _tmp = validation_generator.next()
            X_val[ii] = _tmp[0]
            y_val[ii] = _tmp[1]
        ff.close()

    def train(self,training_h5):
        #Read Dataset
        X_t = HDF5Matrix(training_h5, 'X_train')
        y_t = HDF5Matrix(training_h5, 'y_train')
        X_v = HDF5Matrix(training_h5, 'X_val')
        y_v = HDF5Matrix(training_h5, 'y_val')

        #Create Model
        model=self.Model3()

        #Compile Model
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.Adam(lr=1e-4),
                      metrics=['acc'])
        #Launch Training
        model_checkpoint = ModelCheckpoint(os.path.join(self.output_folder,'modelqc-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5'), verbose=1,
                                           monitor='val_loss', save_best_only=True, mode='auto')
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

        history = model.fit(X_t, y_t, shuffle='batch', batch_size=32, epochs=20, validation_data=(np.array(X_v), y_v),
                            callbacks=[model_checkpoint, earlyStopping])

    def test(self,test_dir,model_file,weights=True,threshold=0.15):

        #Set Image reader
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        batch_size=128
        input_shape=self.input_shape

        test_generator = test_datagen.flow_from_directory(
            directory=test_dir,
            target_size=input_shape[0:2],
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

        #Loop through images
	    if weights is True:
            model = load_model(model_file)
        else:
	        model=self.Model3()
	        model.load_weights(model_file)

        STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
        test_generator.reset()
        pred = model.predict_generator(test_generator)

        filenames=test_generator.filenames
        results_df=pd.DataFrame({"Filename":filenames,
                      "prob":pred[:,1],"label":pred[:,1]>threshold})

        return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QC X-ray images for COVID project')
    parser.add_argument('--operation', help="TRAIN / TEST, etc.", type=str, required=True,
                        choices=['TRAIN', 'TEST', 'DATAGENERATOR'])
    parser.add_argument('--output_folder', type=str, help="Program output logging folder (additional)")
    parser.add_argument('--params_file', type=str, help="Parameters file. Required for both TRAIN/TEST operations.")
    parser.add_argument('--input_dir', type=str, help="Folder to use to create training/validation dataset")
    parser.add_argument('--training_h5', type=str, help="H5 for training. This file is produce by the DATAGENERATOR")
    parser.add_argument('--test_dir', type=str, help="Folder with testing images")
    parser.add_argument('--model', type=str, help="Model file for deployment")
    parser.add_argument('--model_w', type=str, help="Model weights file for deployment")
    parser.add_argument('--out_csv', type=str, help="Testing results: filename, class")

    args = parser.parse_args()

    current_folder = os.path.dirname(os.path.realpath(__file__))
    default_output_folder = os.path.realpath(os.path.join(current_folder, "..", "output"))
    output_folder = args.output_folder if args.output_folder else default_output_folder

    try:
        if not os.path.isdir(output_folder):
            print ("Creating output folder " + output_folder)
            os.makedirs(output_folder)
    except:
        # Default output folder
        output_folder = default_output_folder
        if not os.path.isdir(output_folder):
            print ("Creating output folder " + output_folder)
            os.makedirs(output_folder)

    #parameters_dict = Utils.read_parameters_dict(args.params_file)
    #use_tensorboard = parameters_dict['use_tensorboard']

    e = SlowdownCOVID19QCEngine(output_folder=output_folder)

    if args.operation == 'TRAIN':
        e.train(args.training_h5)
    elif args.operation == 'TEST':
        if model is None:
	        results_df=e.test(args.test_dir,args.model_w,weights=True)
        else:
            results_df=e.test(args.test_dir,args.model,weights=False)
        #Saving results to csv file
        results_df.to_csv(args.out_csv)


