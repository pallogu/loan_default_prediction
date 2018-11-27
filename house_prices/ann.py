import tensorflow as tf
import math

MIN_LAYER_SIZE = 16

def getLayerSize(a):
    return max(MIN_LAYER_SIZE, math.ceil(a))

class ANN(object):
    def __init__(self, numberOfFeatures, reg_lambda):
        self.numberOfFeatures = numberOfFeatures
        self.input_x = tf.placeholder(tf.float32, [None, self.numberOfFeatures], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")

        nn_arch = {
            "w1": tf.Variable(tf.truncated_normal([self.numberOfFeatures, getLayerSize(self.numberOfFeatures / 8)], stddev=0.1), dtype=tf.float32),
            "w2": tf.Variable(tf.truncated_normal([getLayerSize(self.numberOfFeatures / 8), getLayerSize(self.numberOfFeatures / 16)], stddev=0.1), dtype=tf.float32),
            "wOut": tf.Variable(tf.truncated_normal([getLayerSize(self.numberOfFeatures / 16), 1], stddev=0.1), dtype=tf.float32),
            "b1": tf.Variable(tf.truncated_normal([getLayerSize(self.numberOfFeatures / 8)], stddev=0.1), dtype=tf.float32),
            "b2": tf.Variable(tf.truncated_normal([getLayerSize(self.numberOfFeatures / 16)], stddev=0.1), dtype=tf.float32),
            "bOut": tf.Variable(tf.truncated_normal([1], stddev=0.1)),
        }

        _reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)

        l1 = tf.nn.relu(tf.nn.xw_plus_b(self.input_x, nn_arch["w1"], nn_arch["b1"]))
        l2 = tf.nn.relu(tf.nn.xw_plus_b(l1, nn_arch["w2"], nn_arch["b2"]))
        self.pred = tf.nn.xw_plus_b(l2, nn_arch["wOut"], nn_arch["bOut"])

        l2_loss = tf.constant(0.0, dtype=tf.float32)
        for key, value in nn_arch.items():
            l2_loss += tf.nn.l2_loss(value)

        self.loss = tf.reduce_mean(tf.squared_difference(self.pred, self.input_y), name="loss") + _reg_lambda*l2_loss






