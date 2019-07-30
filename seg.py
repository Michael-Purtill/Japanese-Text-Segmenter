from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time

file = 'trainingset.txt'
text = open(file, 'rb').read().decode(encoding='utf-8')
print ('Length of text: {} characters'.format(len(text)))
print(text[:250])
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

seq_length = 50
examples_per_epoch = len(text)//seq_length

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE

BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)

embedding_dim = 256

rnn_units = 1024

rnn = tf.keras.layers.CuDNNLSTM

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                              batch_input_shape=[batch_size, None]),
                            rnn(rnn_units,
        return_sequences=True, 
        recurrent_initializer='lecun_uniform',
        stateful=True),
        tf.keras.layers.Dense(vocab_size)
  ])
    return model




model = build_model(
  vocab_size = len(vocab), 
  embedding_dim=embedding_dim, 
  rnn_units=rnn_units, 
  batch_size=BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1): 
    example_batch_predictions = model(input_example_batch)

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)

model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = loss)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=50

history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))


def generate_text(model, sentence):
    num_generate = 1

    input_eval = [char2idx[s] for s in sentence]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 0.6

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
      
        input_eval = tf.expand_dims([predicted_id], 0)
      
        text_generated.append(idx2char[predicted_id])

    return text_generated[0]

s = input()
segmented = ""

while s != "exit":
    for i in range(len(s)):
        segmented += s[i]
        if (generate_text(model, segmented)) == "|":
            segmented += "|"
    print(segmented)
    s = input()
    segmented = ""


