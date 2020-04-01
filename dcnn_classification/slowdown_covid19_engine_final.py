from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
import numpy as np
from skimage.transform import resize
from sklearn.metrics import roc_auc_score  # roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.utils import HDF5Matrix

from .slowdown_covid19_network import *

import pandas as pd


def rescale_image(x):
    x = x.astype('float32')
    x *= 1/65536.
    return x


def train(h5_dataset_file, output_folder):
    X_t = HDF5Matrix(h5_dataset_file, 'X_train', normalizer=rescale_image)
    y_t = HDF5Matrix(h5_dataset_file, 'y_train')
    X_v = HDF5Matrix(h5_dataset_file, 'X_val', normalizer=rescale_image)
    y_v = HDF5Matrix(h5_dataset_file, 'y_val')

    # datagen = ImageDataGenerator(rescale=1 / 65536.)
    # batch_size = 10

    # training_generator = datagen.flow(X_t, y_t, batch_size=batch_size)
    # validation_generator = datagen.flow(X_v, y_v, batch_size=batch_size)

    path = output_folder + '/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5'

    model_checkpoint = callbacks.ModelCheckpoint(path, verbose=1, monitor='val_loss', save_best_only=False, mode='auto')
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

    p = dict()
    p['target_image_size'] = [1024, 1024]
    p['use_imagenet_weights'] = False
    p['net_type'] = 'TBNet'

    network = SlowdownCOVID19Network(**p)
    network.build_model(True, optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                        loss_function='categorical_crossentropy', additional_metrics=['acc'],
                        pretrained_weights_file_path=None)
    model = network.model
    history = model.fit(X_t, y_t, shuffle=True, batch_size=1, epochs=20, validation_data=(X_v, y_v),
                        callbacks=[model_checkpoint, earlyStopping], verbose=1, max_queue_size=100)

    # history = model.fit_generator(datagen.flow(X_t, y_t, batch_size=batch_size), steps_per_epoch=len(X_t) // batch_size,
    #                               epochs=100, validation_data=datagen.flow(X_v, y_v, batch_size=batch_size),
    #                               validation_steps=len(X_v) // batch_size,
    #                               callbacks=[model_checkpoint, earlyStopping],
    #                               verbose=1, class_weight=None,
    #                               use_multiprocessing=False,
    #                               max_queue_size=100)

    final_path = output_folder + 'cxr-final_model.h5'
    model.save(final_path, overwrite=True)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(epochs, acc, 'bo', label='Training acc')
    ax1.plot(epochs, val_acc, 'b', label='Validation acc')
    ax1.title('Training and validation accuracy')
    ax1.legend()
    accuracy_filepath = output_folder + 'accuracy_history.jpg'
    ax1.savefig(accuracy_filepath)

    ax2.figure()
    ax2.plot(epochs, loss, 'bo', label='Training loss')
    ax2.plot(epochs, val_loss, 'b', label='Validation loss')
    ax2.title('Training and validation loss')
    ax2.legend()
    loss_filepath = output_folder + 'loss_history.jpg'
    ax2.savefig(loss_filepath)

def test(h5_dataset_file, model_path,output_csv_file):

    X_v = HDF5Matrix(h5_dataset_file, 'X_val', normalizer=rescale_image)
    y_v = HDF5Matrix(h5_dataset_file, 'y_val')

    model_path = os.path.join(model_path)

    p = dict()
    p['target_image_size'] = [1024, 1024]
    p['use_imagenet_weights'] = False
    p['net_type'] = 'TBNet'

    network = SlowdownCOVID19Network(**p)
    network.build_model(True, optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                        loss_function='categorical_crossentropy', additional_metrics=['acc'],
                        pretrained_weights_file_path=model_path)
    network_model = network.model

    predictions=network_model.predict(X_v)

    aurocs = []
    ys=tf.keras.utils.to_categorical(y_v, 3)
    finding_names=['normal','mild','moderate-severe']
    for ii in range(predictions.shape[1]):
        try:
            score = roc_auc_score(ys[:, ii], predictions[:, ii])
            print('AUC {}: {}'.format(finding_names[ii], score))
            aurocs.append(score)
        except ValueError:
            score = 0
            print('AUC {}: {}'.format(finding_names[ii], score))
            aurocs.append(score)
    mean_auroc = np.mean(aurocs)
    print('Mean AUROC: ', mean_auroc)

    #Saving results to CSV
    results=dict()
    results['true']=np.argmax(np.array(y_v),axis=1)
    results['normal']=predictions[:,0]
    results['mild']=predictions[:,1]
    results['moderate-severe']=predictions[:,2]

    df = pd.DataFrame.from_dict(results)
    df.to_csv(output_csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect potential signs of COVID-19 for epidemiological control using '
                                                 'chest X-ray (CXR) as a screening tool. Last model training.')
    parser.add_argument('--operation', help="TRAIN / TEST, etc.", type=str, required=True,
                        choices=['TRAIN', 'TEST'])
    parser.add_argument('--i', help="Input h5 dataset file", type=str, required=True)
    parser.add_argument('--o', help="Output folder", type=str, required=False)
    parser.add_argument('--m', help="Model file to use for validation/testing",type=str)
    parser.add_argument('--out_csv', help="Prediction results (csv format). Use in testing or validation")

    args = parser.parse_args()
    if args.operation == 'TRAIN':
        train(args.i, args.o)
    elif args.operation == 'TEST':
        test(args.i,args.m,args.out_csv)
