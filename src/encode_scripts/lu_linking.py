import argparse
import json
import os
from autoencoder import matrix_root, cumulative_normal, normalize
import numpy as np
import pickle as pkl
from scipy import spatial
from tensorflow import keras
from linking import buildIndex,find_matches
from data_processing import load_data_memory,DATA_PATHS
from typing import List


def link(path_encoded_a:str,
         path_encoded_b:str,
         path_meta_a:str,
         path_meta_b:str,
         path_mapper:str,
         encoding_dimension:int,
         linking_thresholds:List[float],
         result_logger_path:str,
         density_transformation:bool=False,
        ) -> None:
    with open(path_encoded_a,"rb") as f:
        keys_a,encoded_a = pkl.load(f)
    with open(path_encoded_b,"rb") as f:
        keys_b,encoded_b = pkl.load(f)


    if path_mapper is not None:
        with open(path_meta_a,"rb") as f:
            mean_a,cov_a = pkl.load(f)
        with open(path_meta_b,"rb") as f:
            mean_b,cov_b = pkl.load(f)
        mapper = keras.models.load_model(path_mapper)
        encoded_b_mapped = mapper.predict(encoded_b)
    else:
        ab = normalize(np.concatenate((encoded_a,encoded_b),axis=0))
        encoded_a = ab[:encoded_a.shape[0]]
        encoded_b_mapped = ab[encoded_a.shape[0]:]

    if density_transformation:
        encoded_a = cumulative_normal(encoded_a)
        encoded_b_mapped = cumulative_normal(encoded_b_mapped)

    aind = buildIndex(encoding_dimension, enumerate(encoded_a), 200)
    matches = {}

    if os.path.isfile(result_logger_path):
        os.remove(result_logger_path)
    for i, linking_threshold in enumerate(linking_thresholds):
        print("linking with "+str(linking_threshold)+" threshold.")
        matches[i] = find_matches(aind, enumerate(encoded_b_mapped), threshold=linking_threshold, threshold_avg=False,
                                  avg_len=encoding_dimension, metric=spatial.distance.cosine)
        n = 0
        keypairs = []
        for match in matches[i]:
            keypair = (keys_a[match[0]],keys_b[match[1]])
            keypairs.append(keypair)
            if keypair[0]==keypair[1]:
                n+=1
        with open(result_logger_path,"a") as f:
            f.write("linking result:\n")
            f.write("linking threshold: "+str(linking_threshold)+"\n")
            f.write("found matches: "+str(len(matches[i]))+"\n")
            f.write("correct matches: "+str(n)+"\n\n")
            # f.write("matched keys:\n")
            # for keypair in keypairs:
            #     f.write(keypair[0]+", "+keypair[1]+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", type=str, default="config.json", help="configuration file path")
    args = parser.parse_args()

    config = json.load(open(args.cf))
    os.chdir(args.cf.rstrip(".json"))
    link(config["encoder"]["path_encoded_a"],
         config["encoder"]["path_encoded_b"],
         config["encoder"]["path_meta_a"],
         config["encoder"]["path_meta_b"],
         config["mapper"]["save_path"],
         config["autoencoder"]["encoding_dimension"],
         config["linking"]["linking_thresholds"],
         config["linking"]["result_logger_path"])
