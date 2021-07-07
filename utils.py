from itertools import permutations
import tensorflow as tf 
from tensorflow.keras.activations import softmax

def my_softmax(input, permutations):

    trans_input = tf.transpose(input, perm=permutations)
    softmax1d = softmax(trans_input)
    return  tf.transpose(softmax1d, perm=permutations)