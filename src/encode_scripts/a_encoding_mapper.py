import os
import argparse
import json
from autoencoder import Autoencoder, soft_relu, matrix_root, normalize
from data_processing import load_data_memory, DATA_PATHS
from bitarray import bitarray
import pickle as pkl
import numpy as np



def encode_decoded_random(ae_path:str,
                          decoded_data_path:str,
                          meta_path:str,
                          encoded_save_path:str
                         ) -> None:

    a = Autoencoder(from_saved=True,path=ae_path)

    with open(decoded_data_path,"rb") as f:
        decoded_ba = pkl.load(f)
    decoded = np.array([[np.float32(x) for x in y.tolist()] for y in decoded_ba])
    encoded = a.encoder.predict(decoded)

    with open(meta_path,"rb") as f:
        mean_a,cov_a = pkl.load(f)
    encoded_transformed = normalize(encoded,(mean_a,cov_a))

    with open(encoded_save_path,"wb") as f:
        pkl.dump(encoded_transformed,f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", type=str, default="config.json", help="configuration file path")
    args = parser.parse_args()

    config = json.load(open(args.cf))
    os.chdir(args.cf.rstrip(".json"))
    encode_decoded_random(config["autoencoder"]["save_path_a"],
                          config["linking"]["decoded_data_path"],
                          config["encoder"]["path_meta_a"],
                          config["linking"]["encoded_data_path"])
