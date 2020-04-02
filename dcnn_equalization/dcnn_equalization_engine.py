from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.utils import HDF5Matrix

from .dcnn_equalization_network import *


def train(h5_dataset_file, output_folder):
    X_t = HDF5Matrix(h5_dataset_file, 'X_train')
    y_t = HDF5Matrix(h5_dataset_file, 'y_train')
    X_v = HDF5Matrix(h5_dataset_file, 'X_val')
    y_v = HDF5Matrix(h5_dataset_file, 'y_val')

    path = output_folder + '/equalization_model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5'

    model_checkpoint = callbacks.ModelCheckpoint(path, verbose=1, monitor='val_loss', save_best_only=False, mode='auto')
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

    network = COVID19EqualizationNetwork(target_image_size=(1024,1024))
    network.build_model(True, optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                        loss_function='mse', additional_metrics=['mae'],
                        pretrained_weights_file_path=None)
    model = network.model
    history = model.fit(X_t, y_t, shuffle=True, batch_size=10, epochs=100, validation_data=(X_v, y_v),
                        callbacks=[model_checkpoint, earlyStopping], verbose=1, max_queue_size=100)

    final_path = output_folder + 'equalization-last_model.h5'
    model.save(final_path, overwrite=True)

    acc = history.history['mae']
    val_acc = history.history['val_mae']
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect potential signs of COVID-19 for epidemiological control using '
                                                 'chest X-ray (CXR) as a screening tool. Last model training.')
    parser.add_argument('--operation', help="TRAIN / TEST, etc.", type=str, required=True,
                        choices=['TRAIN', 'TEST'])
    parser.add_argument('--i', help="Input h5 dataset file", type=str, required=True)
    parser.add_argument('--o', help="Output folder", type=str, required=False)
    # parser.add_argument('--m', help="Model file to use for validation/testing",type=str)
    # parser.add_argument('--out_csv', help="Prediction results (csv format). Use in testing or validation")

    args = parser.parse_args()
    if args.operation == 'TRAIN':
        train(args.i, args.o)
    # elif args.operation == 'TEST':
    #     test(args.i,args.m,args.out_csv)
