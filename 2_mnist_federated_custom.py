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

# Create a custom model
MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')

# Function to created model variables and accumulated statistics, all weights need to have lambda initializer
def create_mnist_variables():
  return MnistVariables(
      weights=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
          name='weights',
          trainable=True),
      bias=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(10)),
          name='bias',
          trainable=True),
      num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
      loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
      accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))

# Implementation of forward pass
def mnist_forward_pass(variables, batch):
  y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
  predictions = tf.cast(tf.argmax(y, 1), tf.int32)

  flat_labels = tf.reshape(batch['y'], [-1])
  loss = -tf.reduce_mean(
      tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(predictions, flat_labels), tf.float32))

  num_examples = tf.cast(tf.size(batch['y']), tf.float32)

  variables.num_examples.assign_add(num_examples)
  variables.loss_sum.assign_add(loss * num_examples)
  variables.accuracy_sum.assign_add(accuracy * num_examples)

  return loss, predictions

# Implement local metrics
def get_local_mnist_metrics(variables):
  return collections.OrderedDict(
      num_examples=variables.num_examples,
      loss=variables.loss_sum / variables.num_examples,
      accuracy=variables.accuracy_sum / variables.num_examples)

# Create global metrics. Note that this part is implement using TFF instruction, not TF
@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
  return collections.OrderedDict(
      num_examples=tff.federated_sum(metrics.num_examples),
      loss=tff.federated_mean(metrics.loss, metrics.num_examples),
      accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples))

# Because we are not converting from Keras Model anymore, we have to create TFF Model class
class MnistModel(tff.learning.Model):

  def __init__(self):
    self._variables = create_mnist_variables()

  @property
  def trainable_variables(self):
    return [self._variables.weights, self._variables.bias]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def local_variables(self):
    return [
        self._variables.num_examples, self._variables.loss_sum,
        self._variables.accuracy_sum
    ]

  @property
  def input_spec(self):
    return collections.OrderedDict(
        x=tf.TensorSpec([None, 784], tf.float32),
        y=tf.TensorSpec([None, 1], tf.int32))

  @tf.function
  def forward_pass(self, batch, training=True):
    del training
    loss, predictions = mnist_forward_pass(self._variables, batch)
    num_exmaples = tf.shape(batch['x'])[0]
    return tff.learning.BatchOutput(
        loss=loss, predictions=predictions, num_examples=num_exmaples)

  @tf.function
  def report_local_outputs(self):
    return get_local_mnist_metrics(self._variables)

  @property
  def federated_output_computation(self):
    return aggregate_mnist_metrics_across_clients

# created federated training operation
iterative_process = tff.learning.build_federated_averaging_process(
    MnistModel,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02))

# invoke initialization to create initial model and optimizer state on server
state = iterative_process.initialize()

NUM_ROUNDS = 10

logdir = "tboard_logs/mnist_federated_custom"
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

print('Finished Training')

# Try evaluating the trained model
evaluation = tff.learning.build_federated_evaluation(MnistModel)
eval_metrics = evaluation(state.model, federated_train_data)
print('Eval metrics={}'.format(eval_metrics))

# Try testing the trained model
federated_test_data = make_federated_data(emnist_test, sample_clients)
test_metrics = evaluation(state.model, federated_test_data)
print('Test metrics={}'.format(test_metrics))
