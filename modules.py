import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv1D, Input, Dropout, Conv1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.activations import softmax
from utils import my_softmax

def MLP(n_in, n_hid, n_out, do_prob=0.):

    input = Input(shape=(n_in,))
    x = Dense(n_hid, activation='relu', use_bias=True)(input)
    x = Dropout(do_prob)(x)
    x = Dense(n_out, activation='relu', use_bias=True)(x)
    x = BatchNormalization()(x)

    return Model(input, x)

def CNN(n_in, n_hid, n_out, do_prob=0.):

    input = Input(shape=(n_in,))
    x = Conv1D(filters=n_hid, kernel_size=5, stride=1, activation='relu', padding='same', use_bias=True)(input)
    x = BatchNormalization()(x)
    x = Dropout(do_prob)(x)
    x = Conv1D(filters=n_hid, kernel_size=5, stride=1, activation='relu', padding='same', use_bias=True)(x)
    x = BatchNormalization()(x)

    pred = Conv1D(filters=n_out, kernel_size=1, stride=1, padding='same', use_bias=True)(x)
    attention = my_softmax(Conv1D(1, kernel_size=1, stride=1, padding='same', use_bias=True)(x), [2,1,0])
    edge_prob = tf.math.reduce_mean(pred * attention, axis=2)

    return Model(input, edge_prob)

def MLPencoder(n_in, n_hid, n_out, do_prob=0., factor = True):

    def edge2node():
        return

    def node2edge():
        return

    input = Input(n_in)
    x = MLP(n_in, n_in, n_in, do_prob=do_prob)

    if factor:
        pass
    return
