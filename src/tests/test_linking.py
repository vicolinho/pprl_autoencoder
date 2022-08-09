import unittest
from data_processing import DATA_PATHS, get_bf_dataset, load_data_memory
from linking import buildIndex, find_matches
from autoencoder import Autoencoder, matrix_root
from random import random
import numpy as np
import pickle as pkl



class TestLinking(unittest.TestCase):

    def test_linking_encoded(self):
        data_a, data_b = [],[]
        keys_a, keys_b = [],[]
        duplicates = {}
        with open("__encoded.pkl","rb") as f:
            keys,data = pkl.load(f)
            data_cov = np.cov(data.T)
            print(data_cov)
            m = matrix_root(data_cov,inverse=True)
            data = np.matmul(data,m)
            for record_key,val_np in zip(keys,data):
                if record_key in keys_a:
                    i = keys_a.index(record_key)
                    j = len(keys_b)
                    data_b.append(val_np)
                    keys_b.append(record_key)
                    duplicates[record_key] = (i,j)
                elif record_key in keys_b:
                    i = len(keys_a)
                    j = keys_b.index(record_key)
                    data_a.append(val_np)
                    keys_a.append(record_key)
                    duplicates[record_key] = (i,j)
                else:
                    if random()<.5:
                        data_a.append(val_np)
                        keys_a.append(record_key)
                    else:
                        data_b.append(val_np)
                        keys_b.append(record_key)

        print(len(data_a))
        ai = buildIndex(len(data_a[0]),enumerate(data_a),20,metric="euclidean",on_disk_build=True)
        matches = find_matches(ai,enumerate(data_b),threshold=.01)
        counter = 0
        found_matches = []
        for match in matches:
            rids = keys_a[match[0]], keys_b[match[1]]
            # print(rids)
            try:
                self.assertEqual(*rids)
                found_matches.append(rids[0])
            except AssertionError as e:
                # print(e)
                counter += 1
                # print(list(zip(data_a[match[0]],data_b[match[1]])))
                # print(np.linalg.norm(data_a[match[0]]-data_b[match[1]])/np.linalg.norm(data_b[match[1]],ord=1))
        # for key in duplicates:
        #     if not key in found_matches:
        #         print(key)
        #         print(list(zip(data_a[duplicates[key][0]],data_b[duplicates[key][1]])))
        #         print(np.linalg.norm(data_a[duplicates[key][0]]-data_b[duplicates[key][1]],ord=1)/np.linalg.norm(data_b[duplicates[key][1]],ord=1))
        print("found matches:",len(matches))
        print("duplicates in data:",len(duplicates))
        print("wrong matches:",counter)
