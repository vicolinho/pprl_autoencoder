import argparse
import os
from autoencoder import Autoencoder, soft_relu, matrix_root, normalize
from data_processing import load_data_memory, DATA_PATHS
from bitarray import bitarray
import pickle as pkl
import numpy as np
import json

def decode_random(ae_path:str,
                  random_data_path:str,
                  meta_path:str,
                  decoded_save_path:str
                  ):
    b = Autoencoder(from_saved=True,path=ae_path)

    with open(random_data_path,"rb") as f:
        data = pkl.load(f)

    with open(meta_path,"rb") as f:
        mean_b,cov_b = pkl.load(f)

    data = normalize(data,(mean_b, cov_b),inverse=True)

    decoded = b.decoder.predict(data)
    decoded = np.array(list(map(lambda x: 0 if x <.5 else 1, decoded.flatten()))).reshape(decoded.shape)
    decoded_ba = [bitarray(list(x)) for x in decoded]
    with open(decoded_save_path,"wb") as f:
        pkl.dump(decoded_ba,f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", type=str, default="config.json", help="configuration file path")
    args = parser.parse_args()

    config = json.load(open(args.cf))
    os.chdir(args.cf.rstrip(".json"))
    decode_random(config["autoencoder"]["save_path_b"],
                  config["linking"]["random_data_path"],
                  config["encoder"]["path_meta_b"],
                  config["linking"]["decoded_data_path"])
