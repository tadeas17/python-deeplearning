from keras.datasets import reuters
from keras import models, layers
import numpy as np
from keras.utils.np_utils import to_categorical
from graph_lib.draw_graph import draw_graph


def vectorize_data(sequnces, dimension=10000):
    result = np.zeros((len(sequnces), dimension))
    for i, sequnce in enumerate(sequnces):
        result[i, sequnce] = 1.
    return result

#Load and vectorize data
(training_data, training_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

x_train = vectorize_data(training_data)
x_test = vectorize_data(test_data)
y_train = to_categorical(training_labels)
y_test = to_categorical(test_labels)


x_val = x_train[:1000]
x_partial = x_train[1000:]
y_val = y_train[:1000]
y_partial = y_train[1000:]
#Create network
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_partial, y_partial, epochs=9, batch_size=512, validation_data=(x_val, y_val))

draw_graph(history)
