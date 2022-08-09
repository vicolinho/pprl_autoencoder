import argparse
import os
import numpy as np
import pickle as pkl
import json

def generate_data(data_length:int, dimension:int, save_path:str) -> None:
    data = np.random.normal(size=(data_length, dimension))
    with open(save_path,"wb") as f:
        pkl.dump(data,f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", type=str, default="config.json", help="configuration file path")
    args = parser.parse_args()

    config = json.load(open(args.cf))
    os.chdir(args.cf.rstrip(".json"))
    generate_data(config["linking"]["training_data_length"],
                  config["autoencoder"]["enocding_dimension"],
                  config["linking"]["random_data_path"])