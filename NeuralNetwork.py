from keras import Model
from keras.models import load_model
from keras.layers import *
from keras.optimizers import Adam
import numpy as np
from Board import Board
from functools import reduce

convol_args = {"filters": 256,
               "kernel_size": 3,
               "padding": "same",
               "data_format": "channels_first"}

action_size = 9 * 9


def convolution_block(x):
    return construct_layers(x, [
        Conv2D(**convol_args),
        BatchNormalization(axis=1),
        Activation('relu')
    ])


def residual_block(x):
    return construct_layers(x, [
        Conv2D(**convol_args),
        BatchNormalization(axis=1),
        Activation('relu'),
        Conv2D(**convol_args),
        BatchNormalization(axis=1),
        lambda i: Add()([x, i]),
        Activation('relu')
    ])


def policy_block(x):
    return construct_layers(x, [
        Flatten(data_format="channels_first"),
        BatchNormalization(axis=1),
        Activation('relu'),
        Dense(action_size, activation='sigmoid'),
        Reshape((9, 9), name='pi')
    ])


def value_block(x):
    return construct_layers(x, [
        Flatten(data_format="channels_first"),
        BatchNormalization(axis=1),
        Activation('relu'),
        Dense(256, activation='relu'),
        Dense(1, activation='tanh', name='v')
    ])


def construct_layers(x, layers):
    return reduce(lambda output, layer: layer(output), layers, x)


class NeuralNetwork:
    input_shape = (6, 9, 9)
    action_size = 9 * 9
    learning_rate = 0.001

    def __init__(self, filename=None):
        if filename is not None:
            self.model = load_model(filename)
            self.input = self.model.get_layer("input")
            return
        # Input (6, 9, 9)
        self.input = Input(shape=NeuralNetwork.input_shape, name="input")

        tower_output = construct_layers(self.input, [convolution_block] + [residual_block] * 5)

        policy_output = policy_block(tower_output)
        value_output = value_block(tower_output)

        self.model = Model(inputs=self.input, outputs=[policy_output, value_output])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                           optimizer=Adam(NeuralNetwork.learning_rate))

    def save_model(self, filename):
        self.model.save(filename)

    def predict(self, board):
        return self.model.predict(np.array([board.to_numpy()]))

    def train(self, data):
        self.model.fit(data['input'], [data['output_p'], data['output_v']], batch_size=100)


if __name__ == '__main__':
    nn = NeuralNetwork()
    bs_data = np.load('bootstrap.npz')
    print(np.any(np.isnan(bs_data['input'])))
    # print(np.all(0 <= bs_data['output_p']))
    # print(np.all(1 >= bs_data['output_p']))
    # print(np.all(np.sum(bs_data['output_p'], axis=(1, 2)) == 1))
    # print(np.all(-1 <= bs_data['output_v']))
    # print(np.all(1 >= bs_data['output_v']))

    # print(bs_data.shape)
    # print(bs_data[0][0].shape)

    # print(bs_data[1].shape)

    # print(bs_data[:, 0].shape)
    nn.train(bs_data)
