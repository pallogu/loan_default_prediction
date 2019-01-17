import tensorflow as tf
import math

MIN_LAYER_SIZE = 128


def getLayerSize(a):
    return max(MIN_LAYER_SIZE, math.ceil(a))


class ANN(object):
    def __init__(self, number_of_cardinal_features, number_of_categorical_features, reg_lambda, graph=None):

        self.input_x_car = tf.placeholder(tf.float32, [None, number_of_cardinal_features], name="input_x_car")
        self.input_x_cat = tf.placeholder(tf.float32, [None, number_of_categorical_features], name="input_x_cat")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")

        if graph:
            nn_arch = {
                "w1_car": graph.get_operation_by_name("w1_car").outputs[0],
                # "w2_car": graph.get_operation_by_name("w2_car").outputs[0],
                # "w3_car": graph.get_operation_by_name("w3_car").outputs[0],
                "wOut_car": graph.get_operation_by_name("wOut_car").outputs[0],

                "b1_car": graph.get_operation_by_name("b1_car").outputs[0],
                # "b2_car": graph.get_operation_by_name("b2_car").outputs[0],
                # "b3_car": graph.get_operation_by_name("b3_car").outputs[0],
                "bOut_car": graph.get_operation_by_name("bOut_car").outputs[0],

                "w1_cat": graph.get_operation_by_name("w1_cat").outputs[0],
                "b1_cat": graph.get_operation_by_name("b1_cat").outputs[0],
                "bOut_cat": graph.get_operation_by_name("bOut_cat").outputs[0],
                "wOut_cat": graph.get_operation_by_name("wOut_cat").outputs[0],
            }

        else:
            nn_arch = {
                "w1_car": tf.Variable(
                    tf.truncated_normal([number_of_cardinal_features, getLayerSize(number_of_cardinal_features / 2)], stddev=0.01),
                    dtype=tf.float32,
                    name="w1_car"
                ),
                # "w2_car": tf.Variable(
                #     tf.truncated_normal([number_of_cardinal_features, getLayerSize(number_of_cardinal_features / 2)], stddev=0.01),
                #     dtype=tf.float32,
                #     name="w2_car"
                # ),
                # "w3_car": tf.Variable(
                #     tf.truncated_normal([getLayerSize(number_of_cardinal_features / 2), getLayerSize(number_of_cardinal_features / 4)], stddev=0.01),
                #     dtype=tf.float32,
                #     name="w3_car"
                # ),
                "wOut_car": tf.Variable(
                    tf.truncated_normal([getLayerSize(number_of_cardinal_features / 2), 1], stddev=0.01),
                    dtype=tf.float32,
                    name="wOut_car"
                ),

                "b1_car": tf.Variable(
                    tf.truncated_normal([getLayerSize(number_of_cardinal_features / 2)], stddev=0.01),
                    dtype=tf.float32,
                    name="b1_car"
                ),
                # "b2_car": tf.Variable(
                #     tf.truncated_normal([getLayerSize(number_of_cardinal_features / 2)], stddev=0.01),
                #     dtype=tf.float32,
                #     name="b2_car"
                # ),
                # "b3_car": tf.Variable(
                #     tf.truncated_normal([getLayerSize(number_of_cardinal_features / 4)], stddev=0.01),
                #     dtype=tf.float32,
                #     name="b3_car"
                # ),
                "bOut_car": tf.Variable(tf.truncated_normal([1], stddev=0.01), name="bOut_car"),

                "w1_cat": tf.Variable(
                    tf.truncated_normal([number_of_categorical_features, number_of_categorical_features], stddev=0.01),
                    dtype=tf.float32,
                    name="w1_cat"
                ),
                "wOut_cat": tf.Variable(
                    tf.truncated_normal([number_of_categorical_features, 1], stddev=0.01),
                    dtype=tf.float32,
                    name="wOut_cat"
                ),
                "b1_cat": tf.Variable(
                    tf.truncated_normal([number_of_categorical_features], stddev=0.01),
                    dtype=tf.float32,
                    name="b1_cat"
                ),
                "bOut_cat": tf.Variable(
                    tf.truncated_normal([1], stddev=0.01),
                    dtype=tf.float32,
                    name="bOut_cat"
                ),
            }

        _reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)

        l1_car = tf.nn.relu(tf.nn.xw_plus_b(self.input_x_car, nn_arch["w1_car"], nn_arch["b1_car"]), name="l1_car")
        # l2_car = tf.nn.relu(tf.nn.xw_plus_b(l1_car, nn_arch["w2_car"], nn_arch["b2_car"]), name="l2_car")
        # l3_car = tf.nn.relu(tf.nn.xw_plus_b(l2_car, nn_arch["w3_car"], nn_arch["b3_car"]), name="l3_car")
        out_car = tf.nn.xw_plus_b(l1_car, nn_arch["wOut_car"], nn_arch["bOut_car"], name="pred_cardicanal")

        l1_cat = tf.nn.sigmoid(tf.nn.xw_plus_b(self.input_x_cat, nn_arch["w1_cat"], nn_arch["b1_cat"]), name="l1_cat")
        out_cat = tf.nn.xw_plus_b(l1_cat, nn_arch["wOut_cat"], nn_arch["bOut_cat"], name="pred_categorical")

        self.pred = tf.add(out_car, out_cat, name="prediction")

        l2_loss = tf.constant(0.0, dtype=tf.float32)
        for key, value in nn_arch.items():
            l2_loss += tf.nn.l2_loss(value)

        self.loss = tf.reduce_mean(tf.squared_difference(self.pred, self.input_y), name="loss") + _reg_lambda*l2_loss






