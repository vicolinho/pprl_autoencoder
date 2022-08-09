import unittest
from autoencoder import Autoencoder, matrix_root
import numpy as np
import pickle as pkl
from tensorflow import keras
from linking import buildIndex,find_matches
from data_processing import load_data_memory,DATA_PATHS

class test_encoderSplitLinking(unittest.TestCase):

    def test_linking(self):
        with open("__encoded_a.pkl","rb") as f:
            keys_a,encoded_a = pkl.load(f)
        with open("__encoded_b.pkl","rb") as f:
            keys_b,encoded_b = pkl.load(f)

        with open("__mean_covar_a.pkl","rb") as f:
            mean_a,cov_a = pkl.load(f)
        with open("__mean_covar_b.pkl","rb") as f:
            mean_b,cov_b = pkl.load(f)

        for d in encoded_b:
            d += -mean_b
        t = matrix_root(cov_b,inverse=True)
        encoded_b = np.matmul(encoded_b,t)

        for d in encoded_a:
            d += -mean_a
        t = matrix_root(cov_a,inverse=True)
        encoded_a = np.matmul(encoded_a,t)

        mapper = keras.models.load_model("__mapper")
        encoded_b_mapped = mapper.predict(encoded_b)
        aind = buildIndex(256,enumerate(encoded_a),50)
        matches = find_matches(aind,enumerate(encoded_b_mapped),threshold=.4,threshold_avg=True,avg_len=256)
        n = 0
        for match in matches:
            correct_match = (keys_a[match[0]]==keys_b[match[1]])
            print(correct_match)
            if correct_match:
                n+=1
        print(n)
        print(len(matches))
