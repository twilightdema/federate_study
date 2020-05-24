import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import os
import shutil

tf.compat.v1.enable_v2_behavior()

np.random.seed(0)

# Loaded federated version of MNIST dataset (key by writter of each data entry to represent i.i.d)
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

# create function to preprocess dataset (flatten and convert from 'pixels','label' to 'x', 'y')
NUM_CLIENTS = 3
NUM_EPOCHS = 1
BATCH_SIZE = 5
SHUFFLE_BUFFER = 10
PREFETCH_BUFFER=10

def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 784]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

# try getting 1 data sample from writer 1
example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])
    
# try getting and printing 1 batch of data
preprocessed_example_dataset = preprocess(example_dataset)

# helper function to create list of dataset for each fenerated node
def make_federated_data(client_data, client_ids):
  return [
      preprocess(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]

# call the function to create federated data (only for subset of writter key for experimental)
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)

print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))

# function to create simple keras model
def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(784,)),
      tf.keras.layers.Dense(10, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])

# wrap it inside the function to convert keras model to TFF model
def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# created federated training operation
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

# invoke initialization to create initial model and optimizer state on server
state = iterative_process.initialize()

NUM_ROUNDS = 10

logdir = "tboard_logs/mnist_federated"
if os.path.exists(logdir):
  shutil.rmtree(logdir)

summary_writer = tf.summary.create_file_writer(logdir)

# Run federated training
with summary_writer.as_default():
  for round_num in range(1, NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {}, metrics={}'.format(str(round_num), metrics))
    for name, value in metrics._asdict().items():
      tf.summary.scalar(name, value, step=round_num)

print('Finished')