import argparse
import pickle

import tensorflow.keras as keras
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda, Conv2DTranspose, Conv1D, MaxPooling1D
import tensorflow.keras.backend as K
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import base64
from bitarray import bitarray

def Conv1DTranspose(input_tensor, filters, kernel_size, activation,
                    name=None, strides=2, padding='valid'):
    """
    Define a 1D deconvolution layer
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1),
                        padding=padding, activation=activation)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2), name=name)(x)
    return x

'''
adapted implementation from https://github.com/UmeanNever/CNN_DCNN_text_classification/blob/master/DCNN.py
'''

class CNN_DCNNAutoencoder:


    def __init__(self, hidden_size):
        self.model = None
        self.hidden_size = hidden_size

    def build(self, seq_length, optimizer, filter_size, strides, window_size):
        embedded_input = Input(shape=(seq_length, 1),
                               dtype='float32', name='embedded_input')
        cnn1 = Conv1D(filters=filter_size, kernel_size=window_size,
                      strides=strides, activation='relu')(embedded_input)
        max_pooling = MaxPooling1D(pool_size=cnn1.shape[1])(cnn1)
        hidden = Conv1D(filters=self.hidden_size, kernel_size=max_pooling.shape[1],
                        strides=strides, activation='relu', name='encoder')(max_pooling)
        DCNN1 = Conv1DTranspose(input_tensor=hidden, filters=self.hidden_size,
                                     kernel_size=cnn1.shape[1],
                                strides=strides,
                                activation='relu')
        reconstruction_output = Conv1DTranspose(input_tensor=DCNN1, filters=1,
                                                     kernel_size=window_size,
                                                strides=strides,
                                                activation='relu', name='reconstruction_output')
        self.model = Model(inputs=embedded_input, outputs=[reconstruction_output])
        self.model.compile(
                      loss={'reconstruction_output': 'mse'},
                      optimizer=optimizer
                     )

    def train(self, data, model_fname, epochs, batch_size):
        sequences = data
        checkpoint = ModelCheckpoint(filepath=model_fname, save_best_only=True)
        self.model.fit(sequences, sequences, epochs=epochs,
                    batch_size=batch_size, verbose=1,
                    validation_split=.15, callbacks=[checkpoint])
        self.model.save(model_fname)

    def generate_random_bf(self, size=1000, number=5000):
        bloom_filters = np.zeros((number, size))
        np.random.seed(1337)
        for i in range(number):
            bf = np.random.randint(2, size=size)
            bloom_filters[i] = bf
        #bloom_filters = bloom_filters.reshape(-1, 1000, 1)
        np.savetxt("random_bf.csv", bloom_filters, delimiter="\t")
        return bloom_filters

    @staticmethod
    def decode(base_string, length):
        bf_string = base64.b64decode(base_string.strip())
        ba = bitarray()
        ba.frombytes(bf_string)
        bf = [np.float32(x) for x in ba.tolist()]
        if len(bf)<length:
            bf.extend(np.array([np.float32(0.) for _ in range(length-len(bf))]))
        return bf

    def read_bloom_filter_from_one_file(self, bf_file, length=1024):
        data = pd.read_csv(bf_file, delimiter=",")
        #print(data)
        data['source'] = data.apply(lambda row: row['rec_src'].split('-')[2], axis=1)
        data['rec_id'] = data.apply(lambda row: int(row['rec_src'].split('-')[1]), axis=1)
        data['bf'] = data.apply(lambda row: CNN_DCNNAutoencoder.decode(row['base64_bf'], length), axis=1)
        grouped = dict(tuple(data.groupby('source')))
        return grouped



    def generate_compact_rep(self, model_file, hidden_file, bfs, ):
        model = load_model(model_file)
        #encoder = model.get_layer('encoder')
        encoder = Model(model.input, model.get_layer("encoder").output)
        hidden = encoder.predict(bfs)
        hidden = hidden.reshape((hidden.shape[0], hidden.shape[2]))
        print(hidden.shape)
        np.savetxt(hidden_file, hidden, delimiter="\t")
        return hidden



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dim', type=str, default=1024, help='Embeddings dimension')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=20, help='Num epochs')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--seq_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--model_fname', type=str, default='test_encoder.h5',
                        help='Model filename')
    parser.add_argument('--bloom_filter_file', type=str, default='E:/data/bloomfilter_primat/northcarolina_autoencoding.csv',
                        help='file consisting of bloom filters in base64 format')
    parser.add_argument('--f', type=int, default=256,
                        help='filter size for cnn')
    parser.add_argument('--ws', type=int, default=16,
                        help='window cnn')
    parser.add_argument('--strides', type=int, default=1,
                        help='window cnn')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    args = parser.parse_args()
    auto_encoder = CNN_DCNNAutoencoder(args.hidden_size)
    auto_encoder.build(args.seq_length, args.optimizer, args.f, args.strides, args.ws)
    #data = auto_encoder.generate_random_bf(500, 10000)
    data_source_dict = auto_encoder.read_bloom_filter_from_one_file(args.bloom_filter_file)
    complete_array = []
    alice_array = []
    bob_array = []
    alice_ids = []
    bob_ids = []
    f = open("ids.csv", "w")
    f.write("source\tid\n")
    print(len(data_source_dict['org']))
    for key, data in data_source_dict['org'].iterrows():
        f.write("org" + "\t" + str(data['rec_id']) + "\n")
        complete_array.append(data['bf'])
        alice_ids.append(data['rec_id'])
        alice_array.append(data['bf'])
    alice_array = np.asarray(alice_array)
    alice_array = alice_array.reshape(-1, alice_array.shape[1], 1)
    print(len(data_source_dict['dup']))
    for key, data in data_source_dict['dup'].iterrows():
        f.write("dup" + "\t" + str(data['rec_id']) + "\n")
        complete_array.append(data['bf'])
        bob_ids.append(data['rec_id'])
        bob_array.append(data['bf'])
    f.close()
    data_source_dict['dup'].to_csv("dup_bloom_filter.csv", "w")
    data_source_dict['org']["bob_file"] = open("org_bloom_filter.csv", "w")

    complete_array = np.asarray(complete_array)
    bob_array = np.asarray(bob_array)
    bob_array = bob_array.reshape(-1, bob_array.shape[1], 1)
    np.savetxt('original_bf.csv', complete_array, delimiter="\t")
    complete_array = complete_array.reshape(-1, complete_array.shape[1], 1)
    print(complete_array.shape)
    auto_encoder.model.summary()
    keras.utils.plot_model(auto_encoder.model,to_file="ae_model.png",show_shapes=True)
    auto_encoder.train(complete_array, args.model_fname, args.n_epochs, args.batch_size)
    # #auto_encoder.generate_compact_rep(args.model_fname, "random_bf_compact.csv", alice_array)
    #
    alice_encode = auto_encoder.generate_compact_rep(args.model_fname, "alice_compact.csv", alice_array)
    bob_encode = auto_encoder.generate_compact_rep(args.model_fname, "bob_compact.csv", bob_array)
