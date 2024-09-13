# Text Generation with LSTM

## Introduction
This project implements a text generation model using LSTM (Long Short-Term Memory) in TensorFlow/Keras. The model takes a sequence of words and predicts the next word in the sequence, which can be used for generating text based on input sentences.

## Dependencies
- TensorFlow (>=2.0)
- Keras (included in TensorFlow)
- NumPy (for numerical computations)

## Installation
To install the required dependencies, run the following command:
```bash
pip install tensorflow numpy
```


Memory updated
Here's a README.md file for your text generation model using LSTM in TensorFlow/Keras:

md
Copy code
# Text Generation with LSTM

## Introduction
This project implements a text generation model using LSTM (Long Short-Term Memory) in TensorFlow/Keras. The model takes a sequence of words and predicts the next word in the sequence, which can be used for generating text based on input sentences.

## Dependencies
- TensorFlow (>=2.0)
- Keras (included in TensorFlow)
- NumPy (for numerical computations)

## Installation
To install the required dependencies, run the following command:
```bash
pip install tensorflow numpy
```
## Model Architecture
The model follows a sequential structure with the following layers:

### Embedding Layer: Maps each word to a vector of 100 dimensions.
### LSTM Layer: Processes the input sequence to learn temporal dependencies.
### Dense Layer: Outputs a probability distribution over the vocabulary for predicting the next word.
### Usage
Data Preparation
The input dataset consists of FAQ text, where sentences are split and tokenized using the Keras Tokenizer. N-grams are created from each sentence, which are then padded to a fixed length using pad_sequences.

```bash
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer.fit_on_texts([faqs])

input_sentence = []

for sentence in faqs.split('\n'):
  tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

  for i in range(1, len(tokenized_sentence)):
    n_gram = tokenized_sentence[:i+1]
    input_sentence.append(n_gram)

padded_input_sentence = pad_sequences(input_sentence, maxlen=max_len, padding='pre')

```

## Model Definition
The model consists of an embedding layer, an LSTM layer, and a Dense output layer with softmax activation.
```bash
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(283, 100, input_length=56))
model.add(LSTM(150))
model.add(Dense(283, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
## Training the Model
The model is trained for 100 epochs using the fit method.
```bash
model.fit(X, y, epochs=100)
```

## Text Prediction
The model can be used to generate text based on an input sequence. Here's an example to predict the next word based on a given word "mail":
```bash
import numpy as np

text = "mail"

for i in range(5):
  token_text = tokenizer.texts_to_sequences([text])[0]
  padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
  pos = np.argmax(model.predict(padded_token_text))
  
  for word, index in tokenizer.word_index.items():
    if index == pos:
      text = text + " " + word
      print(text)
```

## Predicting on Unseen Text
You can also generate text for input that was not in the original dataset:
```bash
text = "hi how are you"

for i in range(10):
  token_text = tokenizer.texts_to_sequences([text])[0]
  padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
  pos = np.argmax(model.predict(padded_token_text))
  
  for word, index in tokenizer.word_index.items():
    if index == pos:
      text = text + " " + word
      print(text)
```
## Further Exploration
Hyperparameters: You can experiment with different LSTM sizes, embedding dimensions, and input sequence lengths to improve the model's performance.
Data: Train the model on larger datasets for better text generation results.

