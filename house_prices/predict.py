import tensorflow as tf
import numpy as np
import os

import csv
from tensorflow.python.framework import ops
import warnings

from sklearn.model_selection import train_test_split

import loadCleanup

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string("checkpointDir", "layer3min16reg1", "Checkpoint directory")
FLAGS = tf.flags.FLAGS

train, unknown = loadCleanup.getTrainTestDFs()
Y = loadCleanup.getValueColumn()
Y = np.array(Y.values).reshape((-1, 1))
trainX, testX, trainY, testY = train_test_split(train, Y, test_size=0.3, random_state=43)

out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.checkpointDir, "checkpoints"))
checkpoint_file = tf.train.latest_checkpoint(out_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        w1 = graph.get_operation_by_name("W1").outputs[0]
        w2 = graph.get_operation_by_name("W2").outputs[0]
        wOut = graph.get_operation_by_name("WOut").outputs[0]

        b1 = graph.get_operation_by_name("b1").outputs[0]
        b2 = graph.get_operation_by_name("b2").outputs[0]
        bOut = graph.get_operation_by_name("bOut").outputs[0]

        l1 = tf.nn.relu(tf.nn.xw_plus_b(tf.convert_to_tensor(unknown, dtype=tf.float32), w1, b1), name="l1")
        l2 = tf.nn.relu(tf.nn.xw_plus_b(l1, w2, b2), name="l2")
        pred = tf.nn.xw_plus_b(l2, wOut, bOut, name="prediction")

        predictions = sess.run(pred)

        # predictions = np.column_stack((sess.run(pred)))

        out_path = os.path.join(os.path.curdir, "submit.csv")
        print("Saving evaluation to {0}".format(out_path))
        with open(out_path, 'w') as f:
            csv.writer(f).writerows(predictions)
