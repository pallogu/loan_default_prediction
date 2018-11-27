import tensorflow as tf
import numpy as np
import os
import time
import datetime
from sklearn.model_selection import train_test_split

import loadCleanup
from ann import ANN

numberOfEpochs = 10000


train, unknown = loadCleanup.getTrainTestDFs()
Y = loadCleanup.getValueColumn()
Y = np.reshape(Y, (-1, 1))
trainX, testX, trainY, testY = train_test_split(train, Y, test_size=0.7, random_state=43)


with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        ann = ANN(numberOfFeatures=train.shape[1], reg_lambda=0.001)

        loss_sum = tf.summary.scalar("Loss", ann.loss)
        summary_merge_op = tf.summary.merge([loss_sum])

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000001, name="Optimizer")

        for epoch in range(numberOfEpochs):
            trainOp = optimizer.minimize(ann.loss)

            sess.run(trainOp, feed_dict={
                ann.input_x: trainX,
                ann.input_y: trainY
            })
            loss, summaries = sess.run([ann.loss, summary_merge_op], feed_dict={
                ann.input_x: trainX,
                ann.input_y: trainY
            })

            train_summary_writer.add_summary(summaries, epoch)
            # if epoch % 1000 == 0:
            print("epoch: {} \t loss {}, ".format(epoch, loss))
