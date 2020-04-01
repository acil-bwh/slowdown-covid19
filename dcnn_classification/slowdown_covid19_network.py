import tensorflow as tf

from tensorflow.keras.applications.densenet import DenseNet121
from cip_python.dcnn.logic import Network, Utils


class SlowdownCOVID19Network(Network):
    """
                 Class that will handle the architecture of the network.
    """
    def __init__(self, target_image_size=(224, 224), use_imagenet_weights=False, net_type='TBNet'):
        """
        Constructor
        :param image_size: int
        :param nb_input_channels: int
        """
        # Calculate input/output sizes based on the patch sizes
        # Assume isometric patch size
        self.target_image_size = (target_image_size[0], target_image_size[1], 3)

        xs_sizes = ((target_image_size[0], target_image_size[1], 3),)  # image size
        ys_sizes = ((3,),)  # 3 classes (normal, mild, moderate-severe)

        # Use parent constructor
        Network.__init__(self, xs_sizes, ys_sizes)

        self.use_imagenet_weights = use_imagenet_weights
        self._expected_input_values_range_ = (0.0, 1.0)  # TODO: change these values
        self.net_type = net_type

    def conv_block(self, input_layer, n_filters, length=2, pool=True, stride=1):
        layer = input_layer
        for i in range(length):
            layer = tf.keras.layers.Conv2D(n_filters, (3, 3), strides=stride, padding='same',
                                           kernel_initializer=tf.keras.initializers.he_normal())(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.ReLU()(layer)

        parallel = tf.keras.layers.Conv2D(n_filters, (1, 1), strides=stride ** length, padding='same',
                                          kernel_initializer=tf.keras.initializers.he_normal())(input_layer)
        parallel = tf.keras.layers.BatchNormalization()(parallel)
        parallel = tf.keras.layers.ReLU()(parallel)

        output = tf.keras.layers.add([layer, parallel])
        # divide_layer = tf.keras.layers.Conv2D(n_filters, (1, 1), strides=1, padding='same', use_bias=False,
        #                                       kernel_initializer=tf.keras.initializers.Constant(value=0.5),
        #                                       trainable=False)(output)
        if pool:
            output = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(output)

        return output

    def build_TBNET_model(self):
        width = 1
        inputs = tf.keras.layers.Input(self.target_image_size)

        output = self.conv_block(inputs, n_filters=16 * width, stride=2)
        output = self.conv_block(output, n_filters=32 * width)
        output = self.conv_block(output, n_filters=48 * width)
        output = self.conv_block(output, n_filters=64 * width)

        output = self.conv_block(output, n_filters=80 * width, pool=False)

        # Global Average Pooling
        output = tf.keras.layers.GlobalMaxPooling2D()(output)

        # Dense
        output = tf.keras.layers.Dense(512, activation='relu')(output)
        output = tf.keras.layers.Dropout(0.5)(output)
        output = tf.keras.layers.Dense(3, activation='softmax')(output)

        return tf.keras.models.Model(outputs=output, inputs=inputs)

    def _build_model_(self):
        if self.net_type == 'TBNet':
            model = self.build_TBNET_model()
        else:
            model = self.build_CheXNet_model_()

        return model

    def build_CheXNet_model_(self):
        """
        Network built as in Rajpurkar, et al.
        "CheXnet: Radiologist-level pneumonia detection on chest x-rays with deep learning."
        arXiv preprint arXiv:1711.05225 (2017).
        :return:
        """
        if self.use_imagenet_weights:
            base_weights = 'imagenet'
        else:
            base_weights = None
        densenet_121 = DenseNet121(weights=base_weights, include_top=False, pooling='avg')
        x = densenet_121.output
        output_layer = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(x)
        return tf.keras.models.Model(inputs=densenet_121.input, outputs=output_layer)
