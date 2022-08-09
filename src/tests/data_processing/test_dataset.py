import unittest
from data_processing import get_bf_dataset, DATA_PATHS
import tensorflow as tf
import numpy as np


class TestDataGenerator(unittest.TestCase):

    def test_data_generator(self):
        ds = get_bf_dataset(DATA_PATHS["NCVR_bf"])
        for batch in ds.take(2):
             print(batch)
