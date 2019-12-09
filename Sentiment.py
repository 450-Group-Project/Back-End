#working_Code/Training
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.externals import joblib
import keras as K
import keras.preprocessing.text
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence

"""
# lines 28-207 is reading from the training data and then creates a model, trains it, then
# -fits it and saves the new model
filepath = 'training_data.txt'

df_list = []

df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
df_list.append(df)
df = pd.concat(df_list)

# next part is using the actual data from the text file to train properly
sentences = df['sentence'].values
label = df['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, label, test_size=0.25, random_state=1000)

# This is for future testing and training
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, verbose=False, validation_data=(X_test, y_test), batch_size=10)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
plot_history(history)

# Wording Embedding
#catches warnings that are thrown from older version of tensorflow

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)
    vocab_size = len(tokenizer.word_index) + 1

maxlen = 100
# using padding
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

###################################################
###################################################
# This is the main part of the project
def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.save('Models/TryModel.h5')
    return model

param_grid = dict(num_filters=[32, 64, 128],
                  kernel_size=[3, 5, 7],
                  vocab_size=[5000],
                  embedding_dim=[50],
                  maxlen=[100])

epochs = 20
embedding_dim = 50
maxlen = 100
output_file = 'dataoutput.txt'

# Run grid search

sentences = df['sentence'].values
y = df['label'].values

# Train-test split
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

# Tokenize words
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
with open('tokenizer.pcl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

# Pad sequences with zeros
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Parameter grid for grid search
param_grid = dict(num_filters=[32, 64, 128],
                    kernel_size=[3, 5, 7],
                    vocab_size=[vocab_size],
                    embedding_dim=[embedding_dim],
                    maxlen=[maxlen])
model = KerasClassifier(build_fn=create_model,
                        epochs=epochs, batch_size=10,
                        verbose=False)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                            cv=4, verbose=1, n_iter=5)
grid_result = grid.fit(X_train, y_train)


# Evaluate testing set
test_accuracy = grid.score(X_test, y_test)

# Save and evaluate results
with open(output_file, 'a') as f:
    s = ('Running {} data set\nBest Accuracy : '
            '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
    output_string = s.format(
        "Data",
        grid_result.best_score_,
        grid_result.best_params_,
        test_accuracy)
    print(output_string)
    f.write(output_string)
"""

#Using to predict/checking prediction
text = "today was a good day"
result_value = text_to_word_sequence(text)
model = load_model('Models/TryModel.h5')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#created array for the tokens to be assigned to the words and place inside the array
Sentence_Index = []
with open('tokenizer.pcl', 'rb') as handle:
    tokenizer = pickle.load(handle)
    word_index = tokenizer.word_index
    for word in result_value:
        try:
            index = word_index[word]
            Sentence_Index.append(index)
        except:
            Sentence_Index.append(0)
predata = K.preprocessing.sequence.pad_sequences([Sentence_Index],
                                                  truncating='pre', padding='pre', maxlen=100)
#this function predicts the results
prediction = model.predict(predata)
print("Prediction (0 = negative, 1 = positive) = ", end="")
print("%0.4f" % prediction[0][0])
