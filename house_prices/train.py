import tensorflow as tf
import numpy as np
import os
import warnings
from tensorflow.python.framework import ops
import time
import datetime
from sklearn.model_selection import train_test_split

import loadCleanup
from ann import ANN

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

tf.flags.DEFINE_integer("number_of_epochs", 10000, "Evaluate model on dev set after this many steps (default: 100)")tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("reg_lambda", 0.01, "Regularisation parameter")
tf.flags.DEFINE_string("description", "layer12", "Run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

train, unknown = loadCleanup.getTrainTestDFs()
Y = loadCleanup.getValueColumn()
Y = np.array(Y.values).reshape((-1, 1))
trainX, testX, trainY, testY = train_test_split(train, Y, test_size=0.3, random_state=43)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        ann = ANN(numberOfFeatures=train.shape[1], reg_lambda=FLAGS.reg_lambda)

        loss_summary = tf.summary.scalar("Loss", ann.loss)
        merge_op = tf.summary.merge([loss_summary])

        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.description))
        print("Writing to {}\n".format(out_dir))

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000001, name="Optimizer")

        for epoch in range(FLAGS.number_of_epochs):
            trainOp = optimizer.minimize(ann.loss)

            sess.run(trainOp, feed_dict={
                ann.input_x: trainX,
                ann.input_y: trainY
            })
            loss, summaries = sess.run([ann.loss, merge_op], feed_dict={
                ann.input_x: trainX,
                ann.input_y: trainY
            })
            train_summary_writer.add_summary(summaries, epoch)

            print("epoch: {} \t loss {}, ".format(epoch, loss))

            if epoch % FLAGS.evaluate_every == 0:
                loss, summaries = sess.run([ann.loss, merge_op], feed_dict={
                    ann.input_x: testX,
                    ann.input_y: testY
                })
                test_summary_writer.add_summary(summaries, epoch)

            if epoch % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=epoch)


