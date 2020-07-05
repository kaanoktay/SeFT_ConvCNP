import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys


class funcRepresentation(keras.layers.Layer):
    def __init__(self, grid, points_per_hour, num_modalities, batch_size, num_points):
        super(funcRepresentation, self).__init__()

        self.sigma = tf.Variable(
            initial_value=2*(1/points_per_hour) * tf.ones(2), 
            dtype=tf.float32,
            trainable=True,
        )
        self.num_modalities = num_modalities
        self.batch_size = batch_size
        self.grid = tf.repeat(
            grid[None,:,None], 
            repeats=batch_size, 
            axis=0
        )
        self.num_points = num_points

    def call(self, y, x, mask):
        x = x[:,:,None]
        dist = (self.grid - tf.transpose(x, perm=[0, 2, 1])) ** 2
        repeated_dist = tf.repeat(dist[:,None,...], repeats=self.num_modalities, axis=1)
        scales = self.sigma[None, None, None, None, :]
        wt = tf.exp(-0.5 * (tf.expand_dims(repeated_dist, -1) / (scales ** 2)))

        density = tf.cast(
            x=mask,
            dtype=tf.float32
        )

        y_out = tf.concat([tf.expand_dims(density, -1), tf.expand_dims(y, -1)], axis=-1)
        y = tf.expand_dims(y_out, 2) * wt
        func = tf.reduce_sum(y, -2)

        density, conv = func[..., :1], func[..., 1:]
        normalized_conv = conv / (density + 1e-8)
        func = tf.concat([density, normalized_conv], axis=-1)
        func = tf.transpose(func, perm=[0, 1, 3, 2])  
        func = tf.reshape(func, shape=[self.batch_size,-1, self.num_points])

        return tf.transpose(func, perm=[0, 2, 1])

class convCNP(keras.Model):
    def __init__(self, grid, points_per_hour, num_modalities, batch_size, num_points, 
                 kernel_size, dropout_rate_conv, dropout_rate_dense, filter_size):
        super(convCNP, self).__init__()

        self.funcLayer = funcRepresentation(grid, points_per_hour, num_modalities, batch_size, num_points)

        self.dropout_conv = keras.layers.Dropout(
            rate = dropout_rate_conv
        )
        self.dropout_dense = keras.layers.Dropout(
            rate = dropout_rate_dense
        )
        self.conv_1 = keras.layers.Conv1D(
            filters=filter_size,
            kernel_size=kernel_size,
            padding="same"
        )
        self.conv_2 = keras.layers.Conv1D(
            filters=filter_size,
            kernel_size=kernel_size,
            padding="same"
        )
        self.conv_3 = keras.layers.Conv1D(
            filters=filter_size*2,
            kernel_size=kernel_size,
            padding="same"
        )
        self.conv_4 = keras.layers.Conv1D(
            filters=filter_size*2,
            kernel_size=kernel_size,
            padding="same"
        )

        self.dense_1 = keras.layers.Dense(
            units=256
        )
        self.dense_2 = keras.layers.Dense(
            units=1
        )

        self.max_pool = keras.layers.MaxPooling1D(
            pool_size=2
        )
        self.flatten = keras.layers.Flatten()

        self.relu = keras.layers.Activation(keras.activations.relu)
        self.sigmoid = keras.layers.Activation(keras.activations.sigmoid)

        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.bn_5 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = inputs[1]
        y = tf.transpose(inputs[2], perm=[0, 2, 1])
        mask = tf.transpose(inputs[3], perm=[0, 2, 1])

        # Functional representation
        func = self.funcLayer(y, x, mask)
        # First conv layer
        z = self.conv_1(func)
        z = self.bn_1(z)
        z = self.relu(z)
        # Second conv layer
        z = self.dropout_conv(z)
        z = self.conv_2(z)
        z = self.bn_2(z)
        z = self.relu(z)
        z = self.max_pool(z)
        # Third conv layer 
        z = self.dropout_conv(z)
        z = self.conv_3(z)
        z = self.bn_3(z)
        z = self.relu(z)
        # Fourth conv layer
        z = self.dropout_conv(z)
        z = self.conv_4(z)
        z = self.bn_4(z)
        z = self.relu(z)
        z = self.max_pool(z)
        # Flatten
        z = self.flatten(z)
        # First dense layer
        z = self.dropout_dense(z)
        z = self.dense_1(z)
        z = self.bn_5(z)
        z = self.relu(z)
        # Second dense layer
        z = self.dropout_dense(z)
        z = self.dense_2(z)
        out = self.sigmoid(z)

        return out