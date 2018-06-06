from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import logging
import pandas as pd


import titanic_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=5000, type=int,
                    help='number of training steps')

def input_fn(dataset):
    dataset = dataset.shuffle(700).repeat().batch(30)
    # dataset = dataset.repeat().batch(70)
    return dataset

def input_test_fn(dataset):
    dataset = dataset.batch(10)
    return dataset

def main(argv):
    args = parser.parse_args(argv[1:])

    # 'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'
    randomForest = tf.contrib.tensor_forest.client.random_forest()

    estimator = tf.estimator.TensorForestEstimator(
        feature_columns=titanic_data.getFeatureDefs(),
        optimizer=tf.train.FtrlOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.0001
        ),
        # hidden_units=[12],
        # optimizer='Adam',
        # activation_fn=tf.nn.relu,
        # loss_reduction=tf.losses.Reduction.SUM,
        n_classes=2)

    estimator.train(input_fn = lambda:input_fn(titanic_data.getTrainingSet()), steps=args.train_steps)

    eval_result_train = estimator.evaluate(input_fn = lambda:input_test_fn(titanic_data.getTrainingSet()))
    eval_result = estimator.evaluate(input_fn = lambda:input_test_fn(titanic_data.getTestSet()))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result_train))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    predict_result = estimator.predict(input_fn = lambda:input_test_fn(titanic_data.getPredictSet()))
    asList = [prediction['class_ids'][0] for prediction in predict_result]
  
    predict = titanic_data.getPreditIds()
    result = pd.concat([predict,pd.Series(asList, name="Survived")], axis=1)
    result.to_csv('submission.csv', columns=["PassengerId", "Survived"], header=True, index=False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    tf.app.run(main)