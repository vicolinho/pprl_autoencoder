import unittest
from autoencoder import Autoencoder, soft_relu
from data_processing import get_bf_dataset, DATA_PATHS, load_data_memory
import tensorflow as tf
import numpy as np
from keras.callbacks import CSVLogger


class TestSplitFitting(unittest.TestCase):

    def test_fitting_a_b(self):
        data = load_data_memory(DATA_PATHS["NCVR_bf"],split=False)
        da = data[:50000]
        db = data[50000:]
        enc,dec = [1024,512,256],[512,1024]
        av = lambda x: soft_relu(x,.2)
        loss = "mae"
        cba = CSVLogger("fit_a.csv",append=False,separator=";")
        cbb = CSVLogger("fit_b.csv",append=False,separator=";")
        a = Autoencoder(enc,dec,activation=av,loss=loss)
        b = Autoencoder(enc,dec,activation=av,loss=loss)
        a.fit(da,epochs=800,batch_size=4096,save=True,path="./__saved_models_a_3",callbacks=[cba])
        b.fit(db,epochs=800,batch_size=4096,save=True,path="./__saved_models_b_3",callbacks=[cbb])

if __name__ == "__main__":
    unittest.main()
