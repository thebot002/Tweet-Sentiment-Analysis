from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import random
import numpy as np
from tqdm import tqdm


class MyLSTM:
    def __init__(self, vocab_len, training_data, max_input_size):
        embedding_len = 32
        input_node_len = 3
        output_dimensionality = 1

        number_epochs = 3
        batch_size = 128

        self.max_input_size = max_input_size
        padded_data = self.prepare_data(training_data[0], vocab_len)

        # creation of the model
        self.model = Sequential()

        # Addition of the layer that converts the sentences to embeddings
        self.model.add(Embedding(vocab_len, embedding_len, input_length=max_input_size))

        # Addition of the lstm layer
        self.model.add(LSTM(input_node_len))

        # Addition of NN classifier output layer
        self.model.add(Dense(output_dimensionality))

        # Compilation of the model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print('Training...')
        # Training of the model

        batches = self.gen_batch(padded_data, batch_size, training_data[1])

        self.model.fit_generator(batches, epochs=number_epochs, steps_per_epoch=(len(training_data[0])/batch_size))#, use_multiprocessing=True)

    def score(self, test_data):
        return self.model.evaluate(test_data[0], test_data[1])

    def prepare_data(self, data, vocab_len):
        # Hashing and the padding the data to prepare it for the embedding layer
        x_data = []
        for sentence in data:
            x_data.append(' '.join(sentence))
        encoded_data = []

        print('Encoding data...')
        t = tqdm(total=len(x_data))
        for s in x_data:
            encoded_data.append(one_hot(s, vocab_len))
            t.update()
        t.close()

        print('Padding data...')
        return pad_sequences(encoded_data, maxlen=self.max_input_size, padding='post')

    def gen_batch(self, data, batch_size, y_data=None, shuffle=False):
        """
        Generates a batch iterator for a dataset.
        """
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        # Shuffle the data at each epoch
        if shuffle:
            if y_data is not None:
                data = list(zip(data.tolist(), y_data))
            random.shuffle(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if y_data is None:
                yield np.array(data[start_index:end_index])
            else:
                if not shuffle:
                    yield data[start_index:end_index], y_data[start_index:end_index]
                else:
                    newdata = []
                    newdata[:], y_data[:] = zip(*data)
                    yield np.array(newdata[start_index:end_index]), np.array(y_data[start_index:end_index])
