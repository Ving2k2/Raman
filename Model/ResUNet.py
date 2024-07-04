import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Conv1D, PReLU, BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
class BasicConv(tf.keras.layers.Layer):
    def __init__(self, channels_out, batch_norm, **kwargs):
        super(BasicConv, self).__init__(**kwargs)
        self.conv = Conv1D(channels_out, kernel_size=3, strides=1, padding='same', use_bias=True)
        self.activation = PReLU()
        if batch_norm:
            self.batch_norm = BatchNormalization()
        else:
            self.batch_norm = None

    def call(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return x


class ResUNetConv(tf.keras.layers.Layer):
    def __init__(self, num_convs, channels, batch_norm):
        super(ResUNetConv, self).__init__()
        self.convs = [tf.keras.layers.Conv1D(channels, kernel_size=3, strides=1, padding='same', use_bias=True)
                      for _ in range(num_convs)]
        self.activation = tf.keras.layers.PReLU()
        if batch_norm:
            self.batch_norm = tf.keras.layers.BatchNormalization()
        else:
            self.batch_norm = None

    def call(self, x):
        x_init = x
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
            if self.batch_norm:
                x = self.batch_norm(x)
        x += x_init
        return x


class UNetLinear(tf.keras.layers.Layer):
    def __init__(self, repeats, channels_out):
        super(UNetLinear, self).__init__()
        self.layers = [tf.keras.layers.Dense(channels_out) for _ in range(repeats)]
        self.activation = tf.keras.layers.PReLU()

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x

def ResUNet_model(num_convs=3, batch_norm=True):
    inputs = tf.keras.Input(shape=(760, 1))
    conv1 = BasicConv(64, batch_norm=batch_norm)(inputs)
    res_unet_conv1 = ResUNetConv(num_convs, 64, batch_norm=batch_norm)(conv1)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(res_unet_conv1)

    conv2 = BasicConv(128, batch_norm=batch_norm)(pool1)
    res_unet_conv2 = ResUNetConv(num_convs, 128, batch_norm=batch_norm)(conv2)
    pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(res_unet_conv2)

    conv3 = BasicConv(256, batch_norm=batch_norm)(pool2)
    res_unet_conv3 = ResUNetConv(num_convs, 256, batch_norm=batch_norm)(conv3)
    pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(res_unet_conv3)

    conv4 = BasicConv(512, batch_norm=batch_norm)(pool3)
    res_unet_conv4 = ResUNetConv(num_convs, 512, batch_norm=batch_norm)(conv4)
    pool4 = tf.keras.layers.MaxPooling1D(pool_size=2)(res_unet_conv4)

    conv5 = BasicConv(1024, batch_norm=batch_norm)(pool4)
    res_unet_conv5 = ResUNetConv(num_convs, 1024, batch_norm=batch_norm)(conv5)
    conv5 = BasicConv(512, batch_norm=batch_norm)(res_unet_conv5)
    up5 = tf.keras.layers.UpSampling1D(size=2)(conv5)

    concat5 = tf.keras.layers.Concatenate(axis=-1)([conv4, up5])
    conv6 = BasicConv(512, batch_norm=batch_norm)(concat5)
    res_unet_conv6 = ResUNetConv(num_convs, 512, batch_norm=batch_norm)(conv6)
    conv6 = BasicConv(256, batch_norm=batch_norm)(res_unet_conv6)
    up6 = tf.keras.layers.UpSampling1D(size=2)(conv6)

    concat6 = tf.keras.layers.Concatenate(axis=-1)([conv3, up6])
    conv7 = BasicConv(256, batch_norm=batch_norm)(concat6)
    res_unet_conv7 = ResUNetConv(num_convs, 256, batch_norm=batch_norm)(conv7)
    conv7 = BasicConv(128, batch_norm=batch_norm)(res_unet_conv7)
    up7 = tf.keras.layers.UpSampling1D(size=2)(conv7)

    concat7 = tf.keras.layers.Concatenate(axis=-1)([conv2, up7])
    conv8 = BasicConv(128, batch_norm=batch_norm)(concat7)
    res_unet_conv8 = ResUNetConv(num_convs, 128, batch_norm=batch_norm)(conv8)
    conv8 = BasicConv(1, batch_norm=batch_norm)(res_unet_conv8)
    up8 = tf.keras.layers.UpSampling1D(size=2)(conv8)

    concat8 = tf.keras.layers.Concatenate(axis=-1)([inputs, up8])
    conv9 = BasicConv(64, batch_norm=batch_norm)(concat8)
    res_unet_conv9 = ResUNetConv(num_convs, 64, batch_norm=batch_norm)(conv9)
    conv9 = BasicConv(1, batch_norm=batch_norm)(res_unet_conv9)
    conv9 = tf.keras.layers.Reshape((-1,))(conv9)

    linear10 = UNetLinear(3, inputs.shape[0])(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=linear10)
    return model

def ResUNet_model(num_convs=3, batch_norm=True):
    inputs = tf.keras.Input(shape=(880, 1))
    conv1 = BasicConv(64, batch_norm=batch_norm)(inputs)
    res_unet_conv1 = ResUNetConv(num_convs, 64, batch_norm=batch_norm)(conv1)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(res_unet_conv1)

    conv2 = BasicConv(128, batch_norm=batch_norm)(pool1)
    res_unet_conv2 = ResUNetConv(num_convs, 128, batch_norm=batch_norm)(conv2)
    pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(res_unet_conv2)

    conv3 = BasicConv(256, batch_norm=batch_norm)(pool2)
    res_unet_conv3 = ResUNetConv(num_convs, 256, batch_norm=batch_norm)(conv3)
    pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(res_unet_conv3)

    conv4 = BasicConv(512, batch_norm=batch_norm)(pool3)
    res_unet_conv4 = ResUNetConv(num_convs, 512, batch_norm=batch_norm)(conv4)
    pool4 = tf.keras.layers.MaxPooling1D(pool_size=2)(res_unet_conv4)

    conv5 = BasicConv(1024, batch_norm=batch_norm)(pool4)
    res_unet_conv5 = ResUNetConv(num_convs, 1024, batch_norm=batch_norm)(conv5)
    conv5 = BasicConv(512, batch_norm=batch_norm)(res_unet_conv5)
    up5 = tf.keras.layers.UpSampling1D(size=2)(conv5)

    concat5 = tf.keras.layers.Concatenate(axis=-1)([conv4, up5])
    conv6 = BasicConv(512, batch_norm=batch_norm)(concat5)
    res_unet_conv6 = ResUNetConv(num_convs, 512, batch_norm=batch_norm)(conv6)
    conv6 = BasicConv(256, batch_norm=batch_norm)(res_unet_conv6)
    up6 = tf.keras.layers.UpSampling1D(size=2)(conv6)

    concat6 = tf.keras.layers.Concatenate(axis=-1)([conv3, up6])
    conv7 = BasicConv(256, batch_norm=batch_norm)(concat6)
    res_unet_conv7 = ResUNetConv(num_convs, 256, batch_norm=batch_norm)(conv7)
    conv7 = BasicConv(128, batch_norm=batch_norm)(res_unet_conv7)
    up7 = tf.keras.layers.UpSampling1D(size=2)(conv7)

    concat7 = tf.keras.layers.Concatenate(axis=-1)([conv2, up7])
    conv8 = BasicConv(128, batch_norm=batch_norm)(concat7)
    res_unet_conv8 = ResUNetConv(num_convs, 128, batch_norm=batch_norm)(conv8)
    conv8 = BasicConv(1, batch_norm=batch_norm)(res_unet_conv8)
    up8 = tf.keras.layers.UpSampling1D(size=2)(conv8)

    concat8 = tf.keras.layers.Concatenate(axis=-1)([inputs, up8])
    conv9 = BasicConv(64, batch_norm=batch_norm)(concat8)
    res_unet_conv9 = ResUNetConv(num_convs, 64, batch_norm=batch_norm)(conv9)
    conv9 = BasicConv(1, batch_norm=batch_norm)(res_unet_conv9)
    conv9 = tf.keras.layers.Reshape((-1,))(conv9)

    linear10 = UNetLinear(3, 880)(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=linear10)
    return model


# class ResUNet(tf.keras.Model):
#     def __init__(self, num_convs, batch_norm):
#         super(ResUNet, self).__init__()
#         self.conv1 = BasicConv(128, batch_norm)
#         self.res_unet_conv1 = ResUNetConv(num_convs, 128, batch_norm)
#         self.pool1 = tf.keras.layers.MaxPool1D(pool_size=2)
#
#         self.conv2 = BasicConv(256, batch_norm)
#         self.res_unet_conv2 = ResUNetConv(num_convs, 256, batch_norm)
#         self.pool2 = tf.keras.layers.MaxPool1D(pool_size=2)
#
#         self.conv3 = BasicConv(512, batch_norm)
#         self.res_unet_conv3 = ResUNetConv(num_convs, 512, batch_norm)
#         self.pool3 = tf.keras.layers.MaxPool1D(pool_size=2)
#
#         self.conv4 = BasicConv(1024, batch_norm)
#         self.res_unet_conv4 = ResUNetConv(num_convs, 1024, batch_norm)
#         self.up4 = tf.keras.layers.UpSampling1D(size=2)
#
#         self.conv5 = BasicConv(512, batch_norm)
#         self.res_unet_conv5 = ResUNetConv(num_convs, 512, batch_norm)
#         self.up5 = tf.keras.layers.UpSampling1D(size=2)
#
#         self.conv6 = BasicConv(256, batch_norm)
#         self.res_unet_conv6 = ResUNetConv(num_convs, 256, batch_norm)
#         self.up6 = tf.keras.layers.UpSampling1D(size=2)
#
#         self.conv7 = BasicConv(128, batch_norm)
#         self.res_unet_conv7 = ResUNetConv(num_convs, 128, batch_norm)
#         self.up7 = tf.keras.layers.UpSampling1D(size=2)
#
#         self.conv8 = BasicConv(64, batch_norm)
#         self.res_unet_conv8 = ResUNetConv(num_convs, 64, batch_norm)
#         self.conv9 = BasicConv(1, batch_norm)
#
#         self.linear10 = UNetLinear(3, 1024)
#
#     def call(self):
#         x = tf.keras.Input(shape=(None, None, 1))
#
#         x = self.conv1(x)
#         x1 = self.pool1(x)
#
#         x2 = self.conv2(x1)
#         x2 = self.pool2(x2)
#
#         x3 = self.conv3(x2)
#         x3 = self.pool3(x3)
#
#         x4 = self.conv4(x3)
#         x4 = self.up4(x4)
#
#         x5 = tf.concat([x3, x4], axis=-1)
#         x5 = self.conv5(x5)
#         x5 = self.up5(x5)
#
#         x6 = tf.concat([x2, x5], axis=-1)
#         x6 = self.conv6(x6)
#         x6 = self.up6(x6)
#
#         x7 = tf.concat([x1, x6], axis=-1)
#         x7 = self.conv7(x7)
#         x7 = self.up7(x7)
#
#         x8 = tf.concat([x, x7], axis=-1)
#         x8 = self.conv8(x8)
#         x9 = self.conv9(x8)
#
#         out = self.linear10(tf.squeeze(x9, axis=-1))
#         model = Model(inputs=x, outputs=out)
#         return model
