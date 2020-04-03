from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import nrrd
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.utils import HDF5Matrix

from cip_python.dcnn.logic import Network, Utils


class COVID19EqualizationNetwork(Network):
    """
                 Class that will handle the architecture of the network.
    """
    def __init__(self, target_image_size=(1024, 1024)):
        """
        Constructor
        :param image_size: int
        :param nb_input_channels: int
        """
        # Calculate input/output sizes based on the patch sizes
        # Assume isometric patch size
        self.target_image_size = (target_image_size[0], target_image_size[1], 1)

        xs_sizes = ((target_image_size[0], target_image_size[1], 1),)  # input image size
        ys_sizes = ((target_image_size[0], target_image_size[1], 1),)  # output image size

        # Use parent constructor
        Network.__init__(self, xs_sizes, ys_sizes)

        self._expected_input_values_range_ = (0.0, 1.0)

    def conv2d(self, input_layer, filters=32, kernel_size=3, batch_normalization=True, activation='relu'):
        layer = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=1,
                                       activation=activation, padding='same')(input_layer)
        if batch_normalization:
            layer = tf.keras.layers.BatchNormalization()(layer)
        return layer

    def _build_model_(self):
        inputs = tf.keras.layers.Input(self.target_image_size)

        # Part 1
        conv1 = self.conv2d(inputs, filters=12, kernel_size=3, batch_normalization=True)
        conv2 = self.conv2d(conv1, filters=12, kernel_size=3, batch_normalization=True)
        conv3 = self.conv2d(conv2, filters=12, kernel_size=3, batch_normalization=True)

        # Part 2 (Down-sampling)
        pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)
        conv4 = self.conv2d(pool1, filters=24, kernel_size=3, batch_normalization=True)
        conv5 = self.conv2d(conv4, filters=24, kernel_size=3, batch_normalization=True)
        conv6 = self.conv2d(conv5, filters=12, kernel_size=3, batch_normalization=True)

        # Part 3 (Up-sampling)
        up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
        add1 = tf.keras.layers.add([up1, conv3])

        # Part 4
        conv7 = self.conv2d(add1, filters=12, kernel_size=3, batch_normalization=True)
        conv8 = self.conv2d(conv7, filters=12, kernel_size=3, batch_normalization=True)
        conv9 = self.conv2d(conv8, filters=12, kernel_size=3, batch_normalization=True)

        # Part 5
        conv10 = self.conv2d(inputs, filters=12, kernel_size=3, batch_normalization=True)
        conv11 = self.conv2d(conv10, filters=12, kernel_size=3, batch_normalization=True)
        conv12 = self.conv2d(conv11, filters=12, kernel_size=3, batch_normalization=True)

        add2 = tf.keras.layers.add([conv9, conv12])
        conv13 = self.conv2d(add2, filters=1, kernel_size=3, batch_normalization=False, activation='sigmoid')

        return tf.keras.models.Model(inputs=inputs, outputs=conv13)


def train(h5_dataset_file, output_folder):
    X_t = HDF5Matrix(h5_dataset_file, 'X_train')
    y_t = HDF5Matrix(h5_dataset_file, 'y_train')
    X_v = HDF5Matrix(h5_dataset_file, 'X_val')
    y_v = HDF5Matrix(h5_dataset_file, 'y_val')

    path = output_folder + '/equalization_model-{epoch:03d}-{mean_squared_error:03f}-{val_mean_squared_error:03f}.h5'

    model_checkpoint = callbacks.ModelCheckpoint(path, verbose=1, monitor='val_loss', save_best_only=False, mode='auto')
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

    network = COVID19EqualizationNetwork(target_image_size=(1024,1024))
    network.build_model(True, optimizer=tf.keras.optimizers.Adam(lr=1e-6),
                        loss_function='mae', additional_metrics=['mse'],
                        pretrained_weights_file_path=None)
    model = network.model
    history = model.fit(X_t, y_t, shuffle=False, batch_size=3, epochs=100, validation_data=(X_v, y_v),
                        callbacks=[model_checkpoint, earlyStopping], verbose=1, max_queue_size=100)

    final_path = output_folder + 'equalization-last_model.h5'
    model.save(final_path, overwrite=True)

    acc = history.history['mean_squared_error']
    val_acc = history.history['val_mean_squared_error']
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


def test(h5_dataset_file, model_path, output_file):
    network = COVID19EqualizationNetwork(target_image_size=(1024, 1024))
    network.build_model(True, optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                        loss_function='mae', additional_metrics=['mse'],
                        pretrained_weights_file_path=model_path)
    model = network.model

    val_images = HDF5Matrix(h5_dataset_file, 'X_val')
    pred = model.predict(val_images, batch_size=1)

    original_val = HDF5Matrix(h5_dataset_file, 'y_val')[:]
    print('MAE: ', mean_absolute_error(original_val.flatten(), pred.flatten()))

    nrrd.write(output_file, pred.squeeze())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect potential signs of COVID-19 for epidemiological control using '
                                                 'chest X-ray (CXR) as a screening tool. Last model training.')
    parser.add_argument('--operation', help="TRAIN / TEST, etc.", type=str, required=True,
                        choices=['TRAIN', 'TEST'])
    parser.add_argument('--i', help="Input h5 dataset file", type=str, required=True)
    parser.add_argument('--o', help="Output folder", type=str, required=False)
    parser.add_argument('--m', help="Model file to use for validation/testing",type=str)
    parser.add_argument('--o_val', help="Output file (.nrrd) to save validation images",type=str)

    args = parser.parse_args()
    if args.operation == 'TRAIN':
        train(args.i, args.o)
    elif args.operation == 'TEST':
        test(args.i,args.m,args.o_val)
