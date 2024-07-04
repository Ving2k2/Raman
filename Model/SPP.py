import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

#   Define SpatialPyramidPooling
class SpatialPyramidPooling(Layer):

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {'channels_last', 'channels_first'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
            self.nb_channels = input_shape[3]


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        input_shape = K.shape(x)


        num_rows = input_shape[1]
        num_cols = input_shape[2]

        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []


        for pool_num, num_pool_regions in enumerate(self.pool_list):

            for ix in range(num_pool_regions):
                for iy in range(num_pool_regions):
                    x1 = ix * col_length[pool_num]
                    x2 = ix * col_length[pool_num] + col_length[pool_num]
                    y1 = iy * row_length[pool_num]
                    y2 = iy * row_length[pool_num] + row_length[pool_num]

                    x1 = K.cast(K.round(x1), 'int32')
                    x2 = K.cast(K.round(x2), 'int32')
                    y1 = K.cast(K.round(y1), 'int32')
                    y2 = K.cast(K.round(y2), 'int32')

                    new_shape = [input_shape[0], y2 - y1,
                                 x2 - x1, input_shape[3]]
                    x_crop = x[:, y1:y2, x1:x2, :]
                    xm = K.reshape(x_crop, new_shape)
                    pooled_val = K.max(xm, axis=(1, 2))
                    outputs.append(pooled_val)

        if self.dim_ordering == 'channels_first':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'channels_last':
            outputs = K.concatenate(outputs,axis = 0)
            outputs = K.reshape(outputs,(self.num_outputs_per_channel,input_shape[0], self.nb_channels))
            outputs = K.permute_dimensions(outputs,(1,0,2))
            outputs = K.reshape(outputs,(input_shape[0], self.num_outputs_per_channel * self.nb_channels))
        return outputs

class ExpandDimsLayer(Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, -1)

def SSPmodel(input_shape=(2, None, 1)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    inputA = inputs[:, 0, :]
    inputB = inputs[:, 1, :]

    convA1 = tf.keras.layers.Conv1D(32, kernel_size=(7), strides=(1), padding='same', kernel_initializer='he_normal')(inputA)
    convA1 = tf.keras.layers.BatchNormalization()(convA1)
    convA1 = tf.keras.layers.Activation('relu')(convA1)
    poolA1 = tf.keras.layers.MaxPooling1D(3)(convA1)

    convB1 = tf.keras.layers.Conv1D(32, kernel_size=(7), strides=(1), padding='same', kernel_initializer='he_normal')(inputB)
    convB1 = tf.keras.layers.BatchNormalization()(convB1)
    convB1 = tf.keras.layers.Activation('relu')(convB1)
    poolB1 = tf.keras.layers.MaxPooling1D(3)(convB1)

    con = tf.keras.layers.concatenate([poolA1, poolB1], axis=2)
    con = ExpandDimsLayer()(con)

    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal')(con)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1)

    spp = SpatialPyramidPooling([1, 2, 3, 4])(conv1)

    full1 = tf.keras.layers.Dense(1024, activation='relu')(spp)
    drop1 = tf.keras.layers.Dropout(0.5)(full1)

    outputs = tf.keras.layers.Dense(2, activation='sigmoid')(drop1)

    model = tf.keras.models.Model(inputs, outputs)
    return model


