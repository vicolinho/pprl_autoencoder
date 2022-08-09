import unittest
from autoencoder import Autoencoder, soft_relu
from data_processing import get_bf_dataset, DATA_PATHS, load_data_memory
import tensorflow as tf
import numpy as np


class TestFitting(unittest.TestCase):

    # def test_fitting(self):
    #     a = Autoencoder([1024,256],[1024])
    #     # a = Autoencoder(from_saved=True,path="./__saved_models_2")
    #     data = get_bf_dataset(DATA_PATHS["NCVR_bf"],batch_size=4096)
    #     a.fit(data,epochs=10,y_in_x=True,save=True,path="./__saved_models_2")

    def test_fitting_memory(self):
        data,test_data = load_data_memory(DATA_PATHS["NCVR_bf"],split=True)
        a = Autoencoder([1024,1024,512,512,128],[512,1024],activation_final=lambda x: soft_relu(x,.2), decoder_weights_constraint=1.5,optimizer=tf.keras.optimizers.Adam(lr=.0005))
        # a = Autoencoder(from_saved=True,path="./__saved_models_3")
        a.fit(data,(test_data,test_data),epochs=250,batch_size=4096,save=True,path="./__saved_models_2")

if __name__ == "__main__":
    unittest.main()
