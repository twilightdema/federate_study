import collections
import functools
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

tf.compat.v1.enable_v2_behavior()

np.random.seed(0)

# A fixed vocabularly of ASCII chars that occur in the works of Shakespeare and Dickens:
vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# load pretrained model
def load_model(batch_size):
  urls = {
      1: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch1.kerasmodel',
      8: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch8.kerasmodel'}
  assert batch_size in urls, 'batch_size must be in ' + str(urls.keys())
  url = urls[batch_size]
  local_file = tf.keras.utils.get_file(os.path.basename(url), origin=url)  
  return tf.keras.models.load_model(local_file, compile=False)

# function for using model to generate text
def generate_text(model, start_string):
  # From https://www.tensorflow.org/tutorials/sequences/text_generation
  num_generate = 200
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  temperature = 1.0

  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(
        predictions, num_samples=1)[-1, 0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

# Try generating using pretrained model
# Text generation requires a batch_size=1 model.
keras_model_batch1 = load_model(batch_size=1)
print(generate_text(keras_model_batch1, 'What of TensorFlow Federated, you ask? '))


# Load Shakespear data from TFF, partitioned by the person who speak the sentence in play
train_data, test_data = tff.simulation.datasets.shakespeare.load_data()

# Here the play is "The Tragedy of King Lear" and the character is "King".
raw_example_dataset = train_data.create_tf_dataset_for_client(
    'THE_TRAGEDY_OF_KING_LEAR_KING')
# To allow for future extensions, each entry x
# is an OrderedDict with a single key 'snippets' which contains the text.
for x in raw_example_dataset.take(2):
  print(x['snippets'])

# Input pre-processing parameters
SEQ_LENGTH = 100
BATCH_SIZE = 8
BUFFER_SIZE = 10000  # For dataset shuffling

# Construct a lookup table to map string chars to indexes,
# using the vocab loaded above:
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=vocab, values=tf.constant(list(range(len(vocab))),
                                       dtype=tf.int64)),
    default_value=0)

# Function to convert character to ID (in TF)
def to_ids(x):
  s = tf.reshape(x['snippets'], shape=[1])
  chars = tf.strings.bytes_split(s).values
  ids = table.lookup(chars)
  return ids

# Function to split sequence of char to input and label (for next char prediction)
def split_input_target(chunk):
  input_text = tf.map_fn(lambda x: x[:-1], chunk)
  target_text = tf.map_fn(lambda x: x[1:], chunk)
  return (input_text, target_text)

# Function to create model input from dataset
def preprocess(dataset):
  return (
      # Map ASCII chars to int64 indexes using the vocab
      dataset.map(to_ids)
      # Split into individual chars
      .unbatch()
      # Form example sequences of SEQ_LENGTH +1
      .batch(SEQ_LENGTH + 1, drop_remainder=True)
      # Shuffle and form minibatches
      .shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
      # And finally split into (input, target) tuples,
      # each of length SEQ_LENGTH.
      .map(split_input_target))

# preprocess the dataset
example_dataset = preprocess(raw_example_dataset)
print(example_dataset.element_spec)

# Create metric function for character based prediction
# (Original Keras SparseCategoricalAccuracy takes rank 2 tensor, so we need to flatten our rank 3 tensor [BATCH X LEN X CH])
class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

  def __init__(self, name='accuracy', dtype=tf.float32):
    super().__init__(name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, [-1, 1])
    y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
    return super().update_state(y_true, y_pred, sample_weight)

BATCH_SIZE = 8  # The training and eval batch size for the rest of this tutorial.
keras_model = load_model(batch_size=BATCH_SIZE)
keras_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[FlattenedCategoricalAccuracy()])

# Confirm that loss is much lower on Shakespeare than on random data
loss, accuracy = keras_model.evaluate(example_dataset.take(5), verbose=0)
print(
    'Evaluating on an example Shakespeare character => Accuracy: {a:3f}'.format(a=accuracy))

# As a sanity check, we can construct some completely random data, where we expect
# the accuracy to be essentially random:
random_guessed_accuracy = 1.0 / len(vocab)
print('Expected accuracy for random guessing: {a:.3f}'.format(
    a=random_guessed_accuracy))
random_indexes = np.random.randint(
    low=0, high=len(vocab), size=1 * BATCH_SIZE * (SEQ_LENGTH + 1))
data = collections.OrderedDict(
    snippets=tf.constant(
        ''.join(np.array(vocab)[random_indexes]), shape=[1, 1]))
random_dataset = preprocess(tf.data.Dataset.from_tensor_slices(data))
loss, accuracy = keras_model.evaluate(random_dataset, steps=10, verbose=0)
print('Evaluating on completely random data  => Accuracy: {a:.3f}'.format(a=accuracy))

# Clone the keras_model inside `create_tff_model()`, which TFF will
# call to produce a new copy of the model inside the graph that it will 
# serialize. Note: we want to construct all the necessary objects we'll need 
# _inside_ this method.
def create_tff_model():
  # TFF uses an `input_spec` so it knows the types and shapes
  # that your model expects.
  input_spec = example_dataset.element_spec
  keras_model_clone = tf.keras.models.clone_model(keras_model)
  return tff.learning.from_keras_model(
      keras_model_clone,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[FlattenedCategoricalAccuracy()])

# This command builds all the TensorFlow graphs and serializes them: 
fed_avg = tff.learning.build_federated_averaging_process(
    model_fn=create_tff_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=0.5))

# For simulation, pick data from 3 clients and construct train and test dataset
# Test data will be evaluate in centralized machine, so data from 3 clients are batched together
def data(client, source=train_data):
  return preprocess(
      source.create_tf_dataset_for_client(client)).take(5)

clients = ['ALL_S_WELL_THAT_ENDS_WELL_CELIA',
           'MUCH_ADO_ABOUT_NOTHING_OTHELLO',
           'THE_TRAGEDY_OF_KING_LEAR_KING']

train_datasets = [data(client) for client in clients]

# We concatenate the test datasets for evaluation with Keras.
test_dataset = functools.reduce(
    lambda d1, d2: d1.concatenate(d2),
    [data(client, test_data) for client in clients])

NUM_ROUNDS = 5

# Initialize the model on server. Note that the model weights are initialized with random function
state = fed_avg.initialize()

# Load pretrained weights here
state = tff.learning.state_with_new_model_weights(
    state,
    trainable_weights=[v.numpy() for v in keras_model.trainable_weights],
    non_trainable_weights=[
        v.numpy() for v in keras_model.non_trainable_weights
    ])

# Function to evaluate model on centralized server
def keras_evaluate(state, round_num):
  # Take our global model weights and push them back into a Keras model to
  # use its standard `.evaluate()` method.
  keras_model = load_model(batch_size=BATCH_SIZE)
  keras_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[FlattenedCategoricalAccuracy()])
  tff.learning.assign_weights_to_keras_model(keras_model, state.model)
  loss, accuracy = keras_model.evaluate(example_dataset, steps=2, verbose=0)
  print('\tEval: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))

# Train the model, for each round, do evaluate then train model in 1 FL batch
for round_num in range(NUM_ROUNDS):
  print('Round {r}'.format(r=round_num))
  keras_evaluate(state, round_num)
  state, metrics = fed_avg.next(state, train_datasets)
  print('\tTrain: loss={l:.3f}, accuracy={a:.3f}'.format(
      l=metrics.loss, a=metrics.accuracy))

# Do final evaluation
keras_evaluate(state, NUM_ROUNDS + 1)

# Set our newly trained weights back in the originally created model.
keras_model_batch1.set_weights([v.numpy() for v in keras_model.weights])
# Text generation requires batch_size=1
print(generate_text(keras_model_batch1, 'What of TensorFlow Federated, you ask? '))

'''
Suggested extensions
This tutorial is just the first step! Here are some ideas for how you might try extending this notebook:

Write a more realistic training loop where you sample clients to train on randomly.
Use ".repeat(NUM_EPOCHS)" on the client datasets to try multiple epochs of local training (e.g., as in McMahan et. al.). See also Federated Learning for Image Classification which does this.
Change the compile() command to experiment with using different optimization algorithms on the client.
Try the server_optimizer argument to build_federated_averaging_process to try different algorithms for applying the model updates on the server.
Try the client_weight_fn argument to to build_federated_averaging_process to try different weightings of the clients. The default weights client updates by the number of examples on the client, but you can do e.g. client_weight_fn=lambda _: tf.constant(1.0).
'''
