import tensorflow as tf

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

    def conv2d(self, input_layer, filters=32, kernel_size=3, batch_normalization=True):
        layer = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=1,
                                       activation='relu', padding='same')(input_layer)
        if batch_normalization:
            layer = tf.keras.layers.BatchNormalization()(layer)
        return layer

    def _build_model_(self):
        inputs = tf.keras.layers.Input(self.target_image_size)

        # Part 1
        conv1 = self.conv2d(inputs, filters=32, kernel_size=3, batch_normalization=True)
        conv2 = self.conv2d(conv1, filters=32, kernel_size=3, batch_normalization=True)
        conv3 = self.conv2d(conv2, filters=32, kernel_size=3, batch_normalization=True)

        # Part 2 (Down-sampling)
        pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)
        conv4 = self.conv2d(pool1, filters=64, kernel_size=3, batch_normalization=True)
        conv5 = self.conv2d(conv4, filters=64, kernel_size=3, batch_normalization=True)
        conv6 = self.conv2d(conv5, filters=64, kernel_size=3, batch_normalization=True)

        # Part 3 (Up-sampling)
        up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
        add1 = tf.keras.layers.concatenate([up1, conv3], axis=-1)

        # Part 4
        conv7 = self.conv2d(add1, filters=32, kernel_size=3, batch_normalization=True)
        conv8 = self.conv2d(conv7, filters=32, kernel_size=3, batch_normalization=True)
        conv9 = self.conv2d(conv8, filters=32, kernel_size=3, batch_normalization=True)

        # Part 5
        conv10 = self.conv2d(inputs, filters=32, kernel_size=3, batch_normalization=True)
        conv11 = self.conv2d(conv10, filters=32, kernel_size=3, batch_normalization=True)
        conv12 = self.conv2d(conv11, filters=32, kernel_size=3, batch_normalization=True)

        add2 = tf.keras.layers.concatenate([conv9, conv12], axis=-1)

        return tf.keras.models.Model(inputs=inputs, outputs=add2)

