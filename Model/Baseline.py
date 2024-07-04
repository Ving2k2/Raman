import tensorflow as tf

def ConvClassifica():
    input = tf.keras.layers.Input((880,))
    input_reshaped = tf.keras.layers.Reshape((880, 1))(input)  # Thêm lớp Reshape để tạo ra kích thước (None, 1024, 1)
    conv1 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation='relu')(input_reshaped)
    pool1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)(conv1)
    flatten = tf.keras.layers.Flatten()(pool1)
    fc1 = tf.keras.layers.Dense(100, activation='relu')(flatten)
    fc2 = tf.keras.layers.Dense(2, activation='sigmoid')(fc1)  # Số lượng đầu ra là 4
    model = tf.keras.Model(inputs=input, outputs=fc2)
    return model
