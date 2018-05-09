from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import logging


# tf.enable_eager_execution()

import titanic_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=50000, type=int,
                    help='number of training steps')

def input_fn(dataset):
    dataset = dataset.shuffle(700).repeat().batch(70)
    # dataset = dataset.repeat().batch(70)
    return dataset

def input_test_fn(dataset):
    dataset = dataset.batch(50)
    return dataset
def main(argv):
    args = parser.parse_args(argv[1:])
    
    estimator = tf.estimator.DNNClassifier(
        feature_columns=titanic_data.getFeatureDefs(),
        hidden_units=[8,4],
        n_classes=2)

    estimator.train(input_fn = lambda:input_fn(titanic_data.getTrainingSet()), steps=args.train_steps)

    eval_result_train = estimator.evaluate(input_fn = lambda:input_test_fn(titanic_data.getTrainingSet()))
    eval_result = estimator.evaluate(input_fn = lambda:input_test_fn(titanic_data.getTestSet()))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result_train))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    tf.app.run(main)