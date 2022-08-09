import tensorflow as tf
from tensorflow import keras
import numpy as np

class EncodingMapper:

    def __init__(self,input_dim,output_dim=None,hidden_dims=[], activation=keras.activations.softsign,optimizer="adam", loss="mae"):
        if output_dim is None:
            output_dim = input_dim
        input = keras.layers.Input(input_dim)
        ll = input
        # ll = keras.layers.Activation(activation)(input)
        for d in hidden_dims:
            ll = keras.layers.Dense(d, activation=activation)(ll)
        output = keras.layers.Dense(output_dim, activation=keras.activations.linear)(ll)
        self.model = keras.Model(input, output)
        self.model.compile(optimizer=optimizer, loss=loss)
