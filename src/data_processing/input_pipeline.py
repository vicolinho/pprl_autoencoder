import base64

import numpy as np
from bitarray import bitarray
from base64 import b64decode
import tensorflow as tf
from typing import Optional, Union, Tuple
import random


def b64_to_array(b64_str: str, n:Optional[int]=None) -> Optional[np.array]:
    """maps base64-encoded bloom filter to corresponding float-array"""

    hash_bytes = b64decode(b64_str)
    ba = bitarray(endian='little')
    ba.frombytes(hash_bytes)
    array = [np.float32(x) for x in ba.tolist()]

    # some encoded bloom filters seem to have wrong length (usually one byte missing)
    if n is not None and len(array)!=n:
        if len(array)<n:
            array.extend([np.float32(0.) for _ in range(n-len(array))])

        else:
            return None
    return np.array(array)

def decode(base_string, length):
    bf_array = bitarray(length, endian='little')
    bf_array.setall(0)
    if isinstance(base_string, str):
        bf_string = base64.b64decode(base_string.strip())
        bf = list()
        for index, bit in enumerate(bf_string.strip()):
            bytes_little = bit.to_bytes(1, 'little')
            array = [access_bit(bytes_little, i) for i in range(len(bytes_little) * 8)]
            bf.extend(array)
        non_zero = np.nonzero(np.asarray(bf))
        for i in non_zero[0]:
            bf_array[int(i)] = 1
    return bf_array


def access_bit(data, num):
    base = int(num // 8)
    shift = int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def decode_tensor(b64_str_tensor: tf.Tensor,n:int=1024) -> tf.Tensor:
    """wrapper to apply 'b64_to_array' to Tensors"""

    return tf.py_function(lambda x: tf.stack([z for z in [b64_to_array(y,n) for y in x.numpy()] if z is not None]),[b64_str_tensor],[tf.float32])


def get_bf_dataset(path:str, batch_size:int=4096, num_epochs:int=1,n:int=1024)->tf.data.Dataset:
    """input pipeline for bloom filter datasets"""

    dataset = tf.data.experimental.make_csv_dataset([path,], column_names=["key","bloom_filter"],select_columns=["bloom_filter",], batch_size=batch_size,sloppy=True,num_epochs=num_epochs)
    dataset = dataset.map(lambda x: x["bloom_filter"])
    dataset = dataset.map(lambda x: decode_tensor(x,n))
    # data will be needed as (feature,label) pair
    dataset = dataset.map(lambda x: (x,x))
    dataset = dataset.prefetch(10)
    return dataset

def load_data_memory(path:str,n:int=1024,split:bool=False,test_split:float=.15,with_keys:bool=False,shuffle:bool=True)-> Union[np.array,Tuple[np.array,np.array]]:
    """returns data as numpy array in memory"""
    print(path)
    data = []
    data_keys = []
    test = []
    test_keys = []
    with open(path, "r") as csv:
        for row in csv:
            if with_keys:
                rec_key = row.split(",")[0]
            bf_base64 = row.rstrip("\n").split(",")[1]
            bf = b64_to_array(bf_base64, n)
            bf = bf.astype(int)
            if bf is not None:
                if split:
                    if random.random()<test_split:
                        test.append(bf)
                        if with_keys:
                            test_keys.append(rec_key)
                    else:
                        data.append(bf)
                        if with_keys:
                            data_keys.append(rec_key)
                else:
                    data.append(bf)
                    if with_keys:
                        data_keys.append(rec_key)
    if shuffle:
        # generate random seed in order to obtain the same permutation twice
        seed = np.random.randint(0,np.iinfo(np.int32).max)
        rng_data = np.random.default_rng(seed)
        data = rng_data.permutation(data)
        if with_keys:
            rng_keys = np.random.default_rng(seed)
            data_keys = rng_keys.permutation(data_keys)
        if split:
            seed = np.random.randint(0,np.iinfo(np.int32).max)
            rng_test = np.random.default_rng(seed)
            test = rng_test.permutation(test)
            if with_keys:
                rng_tkeys = np.random.default_rng(seed)
                test_keys = rng_tkeys.permutation(test_keys)


    if split:
        if with_keys:
            return (np.array(data_keys),np.array(data)),(np.array(test_keys),np.array(test))
        else:
            return np.array(data),np.array(test)
    else:
        if with_keys:
            return np.array(data_keys), np.array(data)
        else:
            return np.array(data)
