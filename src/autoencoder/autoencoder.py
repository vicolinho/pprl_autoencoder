import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .custom_functions import dice_loss, soft_relu, OperatorNormConstraint
from typing import Callable, Iterable, Optional, Union, List
import os

class Autoencoder:
    """
    | class for creating, training and evaluating autoencoders using keras/tensorflow.
    |
    | There are two ways of initializing Autoencoder objects:
    |     (1) from saved models:
    |         keyword arguments:
    |             from_saved -- bool, must be 'True'
    |             path -- str, directory containing the model folders (default: "./__saved_models")
    |     (2) by giving model parameters:
    |         positional arguments:
    |             dims_enc -- list, dimensions of encoding layers, starting with
    |                               input dimension, ending with encoding dimension
    |             dims_dec -- list, dimensions of decoding layers, starting with
    |                               first decoding dimension, ending with output dimension
    |                               (should be the same as input dimension)
    |         keyword arguments:
    |             activation -- str, activation function used for all but the final layer (default: "relu")
    |             activation_final -- str (optional), activation function to be used for the final
    |                                                 layer; if 'None', 'activation' will be used (default: None)
    |             optimizer -- str, Optimizer used for training (default: "adam")
    |             loss -- Callable or str, loss function used for training (default: dice_loss)
    |             decoder_weights_constraint -- float (optional), maximum absolute value for weights
    |                                           in decoder layers (default: None)
    |             initial_weights -- keras.initializers.Initializer (optional), initial weight distribution
                                     (default: None)
    """

    def __init__(self,*args,**kwargs) -> None:
        if "from_saved" in kwargs and kwargs["from_saved"] == True:
            self._load(kwargs["path"])
        elif len(args)>=2:
            self._build(*args,**kwargs)


    def _load(self,path: str="./__saved_models") -> None:
        """loads autoencoder,encoder,decoder models from path"""

        # create custom object scope for mapping saved values to objects
        custom_objects={"dice_loss":dice_loss,"soft_relu":soft_relu,"OperatorNormConstraint":OperatorNormConstraint}
        with keras.utils.custom_object_scope(custom_objects):
            try:
                self.autoencoder = keras.models.load_model(path.rstrip("/")+"/autoencoder")
            except OSError as e:
                print(e)
                print("autoencoder model could not be retrieved")
            try:
                self.encoder = keras.models.load_model(path.rstrip("/")+"/encoder")
            except OSError as e:
                print(e)
                print("encoder model could not be retrieved")
            try:
                self.decoder = keras.models.load_model(path.rstrip("/")+"/decoder")
            except OSError as e:
                print(e)
                print("decoder model could not be retrieved")
            try:
                f = open(path.rstrip("/")+"/meta.txt")
                self.name = f.read()
            except OSError as e:
                print(e)
                print("no metadata found")

    def _build(self,
               name: str,
               dims_enc: list,
               dims_dec: list,
               activation: str="relu",
               activation_final: Optional[str]=None,
               optimizer: str="adam",
               loss:Union[Callable[[tf.Tensor,tf.Tensor],tf.Tensor],str]=dice_loss,
               decoder_weights_constraint:Optional[float]=None,
               initial_weights:Optional[keras.initializers.Initializer]=None) -> None:
        """builds autoencoder, encoder, decoder models with given parameters"""

        self.name = name

        enc = [] # array for encoding layers
        dec = [] # array for decoding layers

        if not activation_final:
            activation_final = activation

        if decoder_weights_constraint is not None:
            constraint=OperatorNormConstraint(decoder_weights_constraint)
        else:
            constraint=None

        # encoding layers
        enc.append(layers.InputLayer(input_shape=(dims_enc[0],)))
        for d in dims_enc[1:-1]:
            enc.append(layers.Dense(d,activation=activation,kernel_initializer=initial_weights))
        # encoder output layer without activation function
        enc.append(layers.Dense(dims_enc[-1],kernel_initializer=initial_weights))

        # activation for encoder output
        dec.append(layers.Activation(activation))
        # decoding layers
        if len(dims_dec)>1:
            dec.append(layers.Dense(dims_dec[0], activation=activation,kernel_constraint=constraint,kernel_initializer=initial_weights))
            for d in dims_dec[1:-1]:
                dec.append(layers.Dense(d,activation=activation,kernel_constraint=constraint,kernel_initializer=initial_weights))
        # output layer, can have different activation function
        dec.append(layers.Dense(dims_dec[-1],activation=activation_final,kernel_constraint=constraint,kernel_initializer=initial_weights))

        # build encoder, decoder, autoencoder models
        self.encoder = keras.models.Sequential()
        self.autoencoder = keras.models.Sequential()
        self.decoder = keras.models.Sequential()

        for layer in enc:
            self.encoder.add(layer)
            self.autoencoder.add(layer)

        self.decoder.add(layers.InputLayer(input_shape=(dims_enc[-1],)))

        for layer in dec:
            self.autoencoder.add(layer)
            self.decoder.add(layer)

        # compile autoencoder for training (encoder, decoder will not be trained, therefore don't need compilation)
        self.autoencoder.compile(optimizer=optimizer,loss=loss)

    def fit(self,
            x_train: Iterable,
            x_test: Optional[Iterable]=None,
            validation_split:Optional[float]=None,
            epochs: int=20,batch_size: int=1024,
            y_in_x: bool=False,
            save: bool=False,
            path: Optional[str]=None,
            callbacks: Optional[List[keras.callbacks.Callback]]=None) -> None:
        """
        | model training
        |
        | positional arguments:
        |     x_train -- Iterable, training data
        | keyword arguments:
        |     x_test -- Iterable (optional), testing/evaluation data (default: None)
        |     epochs -- int, training epochs (default: 20)
        |     batch_size -- int, batch size used in training (default: 1024)
        |     y_in_x -- bool, whether y labels are contained in x or not (default: False)
        |     save -- bool, whether the fitted model should be saved or not (default: False)
        |     path -- str (optional), path used for saving the model,
        |                             default in Autoencoder.save() will be used
        |                             if 'None'. (default: None)
        |     callbacks -- list(keras.callbacks.Callback), callbacks to be run during training
        """

        if y_in_x:
            self.autoencoder.fit(x_train,
                    epochs=epochs,
                    shuffle=False,
                    validation_data=x_test,
                    validation_split=validation_split,
                    callbacks=callbacks)
        else:
            self.autoencoder.fit(x_train,x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=False,
                    validation_data=x_test,
                    validation_split=validation_split,
                    callbacks=callbacks)
        if save:
            self.save(path) if path else self.save()

    def save(self,path: str="./__saved_models") -> bool:
        """save autoencoder, encoder, decoder models"""

        if not os.path.isdir(path):
            try:
                os.mkdir(path)
            except OSError as e:
                print("path does not exist and can not be created")
                return False
        try:
            self.autoencoder.save(path.rstrip("/")+"/autoencoder")
            self.encoder.save(path.rstrip("/")+"/encoder")
            self.decoder.save(path.rstrip("/")+"/decoder")
            with open(path.rstrip("/")+"/meta.txt","w") as f:
                f.write(self.name)
        except Exception as e:
            print(e)
            return False
        return True

    def encode(self,arg:Iterable) -> Iterable:
        """passes arguments through encoder network, returns results"""

        return self.encoder.predict(arg)

    def decode(self,arg:Iterable) -> Iterable:
        """passes arguments through decoder network, returns results"""

        return self.decoder.predict(arg)

    def encode_decode(self,arg:Iterable) -> Iterable:
        """passes arguments through autoencoder network, returns results"""

        return self.autoencoder.predict(arg)
