import numpy as np
import keras as K
import tensorflow as tf
import os
from keras.models import load_model
import warnings

def main():

  np.random.seed(1)
  tf.set_random_seed(1)

  # loading data into memory with maximum word count
  max_words = 20000
  # save np.load to get around new version of tensorflow
  np_load_old = np.load

  # modify the default parameters of np.load
  # call load_data with allow_pickle implicitly set to true
  np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

  (train_x, train_y), (test_x, test_y) = \
      K.datasets.imdb.load_data(seed=1, num_words=max_words)

  # restore np.load for future normal usage
  np.load = np_load_old

  max_sentence_len = 80
  train_x = K.preprocessing.sequence.pad_sequences(train_x,
                                                   truncating='pre', padding='pre', maxlen=max_sentence_len)
  test_x = K.preprocessing.sequence.pad_sequences(test_x,
                                                  truncating='pre', padding='pre', maxlen=max_sentence_len)
  # Uncomment when wanting to create
  """
  #define and compile LSTM model

  ent_init = K.initializers.RandomUniform(-0.01, 0.01, seed=1)
  init = K.initializers.glorot_uniform(seed=1)
  simple_adam = K.optimizers.Adam()
  embed_vec_len = 32
  model = K.models.Sequential()
  model.add(K.layers.embeddings.Embedding(input_dim=max_words,
                                          output_dim=embed_vec_len, embeddings_initializer=ent_init,
                                          mask_zero=True))
  model.add(K.layers.LSTM(units=100, kernel_initializer=init,
                          dropout=0.2, recurrent_dropout=0.2))  # 100 memory
  model.add(K.layers.Dense(units=1, kernel_initializer=init,
                           activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer=simple_adam,
                metrics=['acc'])

  #train model
  bat_size = 32
  max_epochs = 3
  print("\nStarting training ")
  model.fit(train_x, train_y, epochs=max_epochs,
            batch_size=bat_size, shuffle=True, verbose=1)
  print("Training complete \n")

  #evaluate model
  loss_acc = model.evaluate(test_x, test_y, verbose=0)
  
  # 5. save model
  print("Saving model to disk \n")
  mp = ".\\Models\\imdb_model.h5"
  model.save(mp)
  """

  #predicts sentiment
  #use model to make a prediction
  mp = ".\\Models\\imdb_model.h5"
  model = load_model(mp)
  print("New review: \'A contradictory statement is one that says two things that cannot both be true\'")
  d = K.datasets.imdb.get_word_index()
  word_or_sentence = "a contradictory statement is one that says two things that cannot both be true"
  words = word_or_sentence.split()
  word_array = []
  for word in words:
      if word not in d:
          word_array.append(2)
      else:
          word_array.append(d[word] + 3)

  predata = K.preprocessing.sequence.pad_sequences([word_array],
                                                  truncating='pre', padding='pre', maxlen=max_sentence_len)
  prediction = model.predict(predata)
  print("Prediction (0 = negative, 1 = positive) = ", end="")
  print("%0.4f" % prediction[0][0])
if __name__=="__main__":
  main()
