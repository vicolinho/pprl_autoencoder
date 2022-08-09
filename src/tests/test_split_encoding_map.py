import unittest
import pickle as pkl
from autoencoder import Autoencoder,matrix_root
import numpy as np
np.set_printoptions(threshold=np.inf)


class TestSplitMap(unittest.TestCase):

    def test_generate_data(self):
        n,d = 500000, 256
        data = np.random.normal(size=(n,d))
        with open("__random_data.pkl","wb") as f:
            pkl.dump(data,f)

        with open("__mean_covar_b.pkl","rb") as f:
            mean_b,cov_b = pkl.load(f)
        t = matrix_root(cov_b)
        with open("__random_data.pkl","rb") as f:
            data = pkl.load(f)
        data = np.matmul(data,t)
        for d in data:
            d += mean_b
        print(data[0])
        b = Autoencoder(from_saved=True,path="./__saved_models_b_3")
        decoded = b.decoder.predict(data)
        decoded = np.array(list(map(lambda x: 0. if x <.5 else 1., decoded.flatten()))).reshape(decoded.shape)
        with open("__random_decoded.pkl","wb") as f:
            pkl.dump(decoded,f)

        with open("__random_decoded.pkl","rb") as f:
            decoded = pkl.load(f)
        a = Autoencoder(from_saved=True,path="./__saved_models_a_3")
        encoded = a.encoder.predict(decoded)
        with open("__mean_covar_a.pkl","rb") as f:
            mean_a,cov_a = pkl.load(f)
        t = matrix_root(cov_a,inverse=True)
        for d in encoded:
            d += -mean_a
        encoded_transformed = np.matmul(encoded,t)
        with open("__random_encoded.pkl","wb") as f:
            pkl.dump(encoded_transformed,f)
