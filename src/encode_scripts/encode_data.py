import argparse
import json
import os
from autoencoder import Autoencoder, soft_relu, matrix_root, normalize
from data_processing import load_data_memory, DATA_PATHS
import pickle as pkl
import numpy as np



def encode(encoding_data:str,
           dataset: str,
           encoder_path:str,
           encoded_save_path: str,
           meta_save_path: str,
           normal: bool = True)->None:
    keys,data = load_data_memory(DATA_PATHS[encoding_data][dataset], with_keys=True)

    ae = Autoencoder(from_saved=True, path=encoder_path)

    encoded = ae.encode(data)
    if normal:
        encoded, mean_cov = normalize(encoded, return_mean_cov=True)

    with open(encoded_save_path,"wb") as f:
        pkl.dump((keys, encoded), f)

    if normal:
        with open(meta_save_path,"wb") as f:
            pkl.dump(mean_cov, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", type=str, default="config.json", help="configuration file path")
    args = parser.parse_args()

    config = json.load(open(args.cf))
    os.chdir(args.cf.rstrip(".json"))
    encode(config["encoder"]["encoding_dataset"],
           "a",
           config["autoencoder"]["save_path_a"],
           config["encoder"]["path_encoded_a"],
           config["encoder"]["path_meta_a"])
