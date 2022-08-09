import argparse
import json
import os
import pickle as pkl
import numpy as np
import tensorflow as tf
from tensorflow import keras
from linking import EncodingMapper
from autoencoder import Autoencoder,soft_relu, matrix_root, function_mapper,TimeLogger
from data_processing import load_data_memory,DATA_PATHS
from keras.callbacks import CSVLogger, ModelCheckpoint
from typing import List, Optional


def fit_mapper(random_data_path:str,
               encoded_data_path:str,
               encoding_dimension:int,
               mapper_hidden_dimensions:List[int],
               mapper_loss:str,
               validation_split:float,
               epochs:int,
               batch_size:int,
               save_path:str,
               logger_path:str) -> None:
    # load training data
    with open(random_data_path,"rb") as f:
        data = pkl.load(f)
    with open(encoded_data_path,"rb") as f:
        decoded_encoded = pkl.load(f)

    # train mapper
    mapper = EncodingMapper(encoding_dimension,hidden_dims=mapper_hidden_dimensions,loss=function_mapper(mapper_loss))
    csv_logger = CSVLogger(logger_path,append=False,separator=";")
    time_logger = TimeLogger("times_"+logger_path)
    checkpoint = ModelCheckpoint(filepath=save_path, save_best_only=True)
    mapper.model.fit(data,decoded_encoded,validation_split=validation_split,epochs=epochs,batch_size=batch_size,callbacks=[csv_logger,time_logger,checkpoint])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", type=str, default="config.json", help="configuration file path")
    args = parser.parse_args()

    config = json.load(open(args.cf))
    os.chdir(args.cf.rstrip(".json"))
    fit_mapper(config["linking"]["random_data_path"],
               config["linking"]["encoded_data_path"],
               config["autoencoder"]["encoding_dimension"],
               config["mapper"]["hidden_dimensions"],
               config["mapper"]["loss_function"],
               config["mapper"]["validation_split"],
               config["mapper"]["training_epochs"],
               config["mapper"]["batch_size"],
               config["mapper"]["save_path"],
               config["mapper"]["logger_path"])
