import argparse
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tf.enable_eager_execution()

import titanic_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


# def loss(model, x, y):
#   y_ = model(x)
#   return tf.losses.mean_squared_error(labels=y, predictions=y_)
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
  with tfe.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

def main(argv):
    args = parser.parse_args(argv[1:])

    (features, labels) = titanic_data.getTrainingSet()
    X = tf.convert_to_tensor(features, np.float32)
    Y = tf.convert_to_tensor(labels, np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    # dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(32)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(35, activation="relu", input_shape=(49,)),
        tf.keras.layers.Dense(35, activation="relu"),
        tf.keras.layers.Dense(35, activation="relu"),
        tf.keras.layers.Dense(2)
    ])

    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 501

    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        # Training loop - using batches of 32
        for (x, y) in tfe.Iterator(dataset):
            # Optimize the model
            grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.variables),
                                    global_step=tf.train.get_or_create_global_step())

            # Track progress
            epoch_loss_avg(loss(model, x, y))  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

        # end epoch

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)