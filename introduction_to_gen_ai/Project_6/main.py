import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense
import nltk
import random

# Download dataset
nltk.download('gutenberg')
from nltk.corpus import gutenberg

text = gutenberg.raw('shakespeare-hamlet.txt')

# Preprocessing: Convert to lowercase & tokenize words
text = text.lower().replace('\n', ' ')
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])


sequence_data = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1


sequence_length = 50  
input_sequences = []
for i in range(len(sequence_data) - sequence_length):
    input_sequences.append(sequence_data[i:i + sequence_length + 1])

input_sequences = np.array(input_sequences)
X, y = input_sequences[:, :-1], input_sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

model = Sequential([
    Embedding(vocab_size, 100, input_length=sequence_length),
    LSTM(150, return_sequences=True),
    LSTM(150),
    Dense(150, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


model.fit(X, y, epochs=30, batch_size=128)


model.save("lstm_text_generator.h5")


model = tf.keras.models.load_model("lstm_text_generator.h5")


def generate_text(seed_text, next_words=50, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding='pre')

        predictions = model.predict(token_list, verbose=0)[0]
        predictions = np.log(predictions) / temperature  # Adjust temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        predicted_index = np.random.choice(range(len(predictions)), p=predictions)
        output_word = tokenizer.index_word.get(predicted_index, "")

        seed_text += " " + output_word
    return seed_text


print(generate_text("To be or not to be", next_words=50, temperature=0.8))
