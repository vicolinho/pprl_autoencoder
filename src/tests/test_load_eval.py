import unittest
from autoencoder import Autoencoder
from data_processing import get_bf_dataset, DATA_PATHS, load_data_memory
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
np.set_printoptions(threshold=np.inf)


class TestLoadEval(unittest.TestCase):

    # def test_load_eval(self):
    #     a = Autoencoder(from_saved=True,path="./__saved_models_2")
    #     data = get_bf_dataset(DATA_PATHS["NCVR_bf"],batch_size=4)
    #     for d,_ in data.take(2):
    #         for bf in d.numpy():
    #             ed = a.encode_decode(np.array([bf]))
    #             print(np.linalg.norm(bf-ed[0],ord=1))
    #             plt.bar(list(range(len(bf))),bf.tolist(),alpha=.5)
    #             plt.bar(list(range(len(bf))),ed.tolist()[0],alpha=.5)
    #             plt.show()

    def test_encode(self):
        a = Autoencoder(from_saved=True,path="./__saved_models_2")
        for layer_ws in a.autoencoder.get_weights():
            print(layer_ws.shape)
        keys,data=load_data_memory(DATA_PATHS["NCVR_bf"],with_keys=True)
        encoded = a.encode(data)
        print(len(encoded))
        print(encoded.shape)
        with open("__encoded.pkl","wb") as f:
            pkl.dump((keys,encoded), f)
