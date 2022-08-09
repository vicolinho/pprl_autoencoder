import unittest
from autoencoder import Autoencoder
from data_processing import get_bf_dataset, DATA_PATHS, load_data_memory
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
np.set_printoptions(threshold=np.inf)


class TestSplitLoadEval(unittest.TestCase):

    def test_encode(self):
        a = Autoencoder(from_saved=True,path="./__saved_models_a_3")
        b = Autoencoder(from_saved=True,path="./__saved_models_b_3")
        keys,data=load_data_memory(DATA_PATHS["NCVR_bf"],with_keys=True)
        encoded_a = a.encode(data[:50000])
        encoded_b = b.encode(data[50000:])
        print(len(encoded_a))
        print(len(encoded_b))
        print(encoded_a.shape)
        with open("__encoded_a.pkl","wb") as f:
            pkl.dump((keys[:50000],encoded_a), f)
        with open("__encoded_b.pkl","wb") as f:
            pkl.dump((keys[50000:],encoded_b), f)

        with open("__mean_covar_a.pkl","wb") as f:
            val = (np.mean(encoded_a,axis=0),np.cov(encoded_a,rowvar=False))
            pkl.dump(val, f)

        with open("__mean_covar_b.pkl","wb") as f:
            val = (np.mean(encoded_b,axis=0),np.cov(encoded_b,rowvar=False))
            pkl.dump(val, f)
