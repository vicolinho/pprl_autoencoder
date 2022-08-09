import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Union, Callable
import time
from scipy import stats

def dice_loss(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """generalized dice-loss implementation for non-negative integer arrays x,y"""

    dice = tf.norm(x-y,ord=1) / (keras.backend.sum(x) + keras.backend.sum(y))
    return dice

def soft_relu(x:tf.Tensor, alpha:float=.2, max_value=1)-> tf.Tensor:
    """identity between 0 and max_value; continuously extended with slope alpha outside of [0,max_value]"""
    return (1-alpha)*keras.activations.relu(x,max_value=max_value)+alpha*keras.activations.linear(x)

class OperatorNormConstraint(keras.constraints.Constraint):
    """Constrains weights-matrix in the operator-norm"""

    def __init__(self,max_norm:float=1.) -> None:
        self.max_norm = max_norm

    def __call__(self, w:tf.Tensor) -> tf.Tensor:
        norms = tf.norm(w,axis=0,ord=1)
        norms = tf.expand_dims(norms, axis=0)
        multiplyers = tf.map_fn(lambda x: tf.math.minimum(np.float32(1.),self.max_norm/x), norms)
        return tf.multiply(w,multiplyers)

    def get_config(self):
        return {"max_norm":self.max_norm}

def matrix_root(m: np.array, inverse:bool=False) -> np.array:
    eig, base = np.linalg.eig(m)
    if not all(e > 0 for e in eig):
        raise ValueError("Matrix is not positive definite")
    if not inverse:
        root_matrix = np.matmul(np.matmul(base, np.diag(np.sqrt([e for e in eig]))), base.T)
    else:
        root_matrix = np.matmul(np.matmul(base,np.diag(np.sqrt([1/e for e in eig]))), base.T)
    return root_matrix

def function_mapper(fname:str) -> Union[str,Callable]:
    known_functions = {
        "soft_relu": soft_relu
    }

    if fname in known_functions:
        return known_functions[fname]
    else:
        return fname


class TimeLogger(keras.callbacks.Callback):
    def __init__(self,save_path):
        self.save_path = save_path
        self.epoch = 0

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
        self.epoch +=1

    def on_epoch_end(self, batch, logs={}):
        with open(self.save_path,"a") as f:
            f.write(str(self.epoch)+","+str(time.time() - self.epoch_time_start)+"\n")


def normalize(data,mean_cov=None,inverse=False,return_mean_cov=False):
    if mean_cov is not None:
        mean, cov = mean_cov
    else:
        mean, cov = (np.mean(data, axis=0), np.cov(data, rowvar=False))
    if not inverse:
        data = data-mean
        t = matrix_root(cov,inverse=True)
        data = np.matmul(data,t)
    else:
        t = matrix_root(cov, inverse=False)
        data = np.matmul(data, t)
        data = data+mean
    if return_mean_cov:
        return data, (mean, cov)
    else:
        return data

def cumulative_normal(x):
    return stats.norm(0, 1).cdf(x)
