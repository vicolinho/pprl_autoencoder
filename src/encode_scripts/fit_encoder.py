import argparse
import json
import numpy as np
import os
import tensorflow as tf
from autoencoder import Autoencoder, function_mapper, TimeLogger
from data_processing import get_bf_dataset, DATA_PATHS, load_data_memory
from keras.callbacks import CSVLogger
from termcolor import cprint
from typing import Callable,List,Optional,Union


def fit_encoder(name:str,
                input_dimension:int,
                encoder_hidden_dimensions:List[int],
                encoding_dimension:int,
                decoder_hidden_dimensions:List[int],
                output_dimension:int,
                activation_function:str,
                training_data:str,
                dataset:str,
                validation_split:float,
                loss_function:str,
                training_epochs:int,
                batch_size:int,
                save_path:str,
                log_file:str) -> None:
    data = load_data_memory(DATA_PATHS[training_data][dataset], split=False)
    enc = [input_dimension]+encoder_hidden_dimensions+[encoding_dimension]
    dec = decoder_hidden_dimensions+[output_dimension]
    activation = function_mapper(activation_function)
    loss = function_mapper(loss_function)
    epochs = training_epochs

    if not os.path.isdir(save_path):
        a = Autoencoder(name,enc,dec,activation=activation,loss=loss)
        csv_logger = CSVLogger(log_file,append=False,separator=";")
        time_logger = TimeLogger("times_"+log_file)
        a.fit(data,epochs=epochs, validation_split=validation_split,batch_size=batch_size, save=True, path=save_path,callbacks=[csv_logger,time_logger])
    else:
        cprint("\""+save_path+"\" already exists. Either delete the existing model or choose a different path!","red")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", type=str, default="config.json", help="configuration file path")
    args = parser.parse_args()

    config = json.load(open(args.cf))
    os.chdir(args.cf.rstrip(".json"))
    fit_encoder("model_a",
                config["autoencoder"]["input_dimension"],
                config["autoencoder"]["encoder_hidden_dimensions"],
                config["autoencoder"]["encoding_dimension"],
                config["autoencoder"]["decoder_hidden_dimensions"],
                config["autoencoder"]["output_dimension"],
                config["autoencoder"]["activation_function"],
                config["autoencoder"]["training_dataset"],
                "a",
                config["autoencoder"]["validation_split"],
                config["autoencoder"]["loss_function"],
                config["autoencoder"]["training_epochs"],
                config["autoencoder"]["batch_size"],
                config["autoencoder"]["save_path_a"],
                config["autoencoder"]["log_file_a"])
