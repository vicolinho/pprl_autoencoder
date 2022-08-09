import unittest
import pickle as pkl
import numpy as np
import tensorflow as tf
from tensorflow import keras
from linking import EncodingMapper
from autoencoder import Autoencoder,soft_relu, matrix_root
from data_processing import load_data_memory,DATA_PATHS
from scipy.stats import skew,kurtosis
from matplotlib import pyplot as plt
from keras.callbacks import CSVLogger, ModelCheckpoint


np.set_printoptions(threshold=np.inf)


class TestModel(unittest.TestCase):

    def test_encoding_mapper(self):
        with open("__random_data.pkl","rb") as f:
            data = pkl.load(f)
        with open("__random_encoded.pkl","rb") as f:
            decoded_encoded = pkl.load(f)

        with open("__mean_covar_a.pkl","rb") as f:
            mean_a,cov_a = pkl.load(f)
        with open("__mean_covar_b.pkl","rb") as f:
            mean_b,cov_b = pkl.load(f)

        # t = matrix_root(cov_b)
        # data = np.matmul(data,t)
        # for d in data:
        #     d += mean_b
        #
        # t = matrix_root(cov_a)
        # decoded_encoded = np.matmul(decoded_encoded,t)
        # for d in decoded_encoded:
        #     d += mean_a






        a = Autoencoder(from_saved=True, path="__saved_models_a_3")
        b = Autoencoder(from_saved=True, path="__saved_models_b_3")
        _,bfs = load_data_memory(DATA_PATHS["NCVR_bf"],split=True)
        print("encoders work:")
        print(np.testing.assert_array_equal(a.decode(a.encode(bfs[0:1])),a.encode_decode(bfs[0:1])))
        print(np.testing.assert_array_equal(b.decode(b.encode(bfs[0:1])),b.encode_decode(bfs[0:1])))

        test_a = a.encode(bfs)
        test_b = b.encode(bfs)

        for d in test_b:
            d += -mean_b
        t = matrix_root(cov_b,inverse=True)
        test_b = np.matmul(test_b,t)

        for d in test_a:
            d += -mean_a
        t = matrix_root(cov_a,inverse=True)
        test_a = np.matmul(test_a,t)


        stats = test_b.copy()
        print()
        # print("Skew:")
        # print(skew(stats))
        # print("Kurtosis:")
        # print(kurtosis(stats))
        # print()
        # print("min:")
        # print(np.min(stats,axis=0))
        # print("max:")
        # print(np.max(stats,axis=0))
        # print(np.cov(stats,rowvar=False))
        plt.hist(stats[:,42])
        plt.savefig("__histogram.png")


        mapper = EncodingMapper(256,hidden_dims=[2048,4096,2048],loss="mae")
        cb = CSVLogger("fit_mapper.csv",append=False,separator=";")
        checkpoint = ModelCheckpoint(filepath="./__mapper", save_best_only=True)
        mapper.model.fit(data,decoded_encoded,validation_data=(test_b,test_a),epochs=20,batch_size=1024,callbacks=[cb, checkpoint])
        # mapper.model.fit(test_b[:50000],test_a[:50000],validation_data=(test_b[50000:],test_a[50000:]),epochs=20,batch_size=1024,callbacks=[cb,checkpoint])
