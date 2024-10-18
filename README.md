# Text Prediction Model

This repository contains the implementation of a text prediction model using an LSTM (Long Short-Term Memory) neural network. The model is trained on multiple datasets to predict the next word in a sequence of text, similar to the autocomplete feature in applications like Gmail and Gboard.

## Table of Contents
- [Introduction](#introduction)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Training](#training)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction

This project implements a text prediction model that can suggest the next word given an input sequence of text. It uses an LSTM-based neural network to learn the patterns in the text and generate the most probable next word. 

The model is trained on a combination of various text datasets to improve its ability to generalize across different types of text data, including English and Bangla languages, and synthetic conversations.

## Datasets

The following datasets were used for training the model:
1. **Fake News Detection**: [Pooja Jain's Fake News Detection Dataset](https://www.kaggle.com/jainpooja/fake-news-detection)
2. **Next Word Prediction (English)**: [Ronik Dedhia's Next Word Prediction](https://www.kaggle.com/ronikdedhia/next-word-prediction)
3. **Next Word Prediction (Bangla)**: [Asif Mahmud's Next Word Prediction - Bangla](https://www.kaggle.com/asifmahmudcste/next-word-prediction-bangla)
4. **Synthetic Phone Call Conversations**: [Zeyad Sayed Abdullah's Phone Call Conversation Dataset](https://www.kaggle.com/zeyadsayedadbullah/synthetic-phone-call-conversation)

These datasets provide diverse text data that helps the model learn language patterns across different contexts and domains.

## Model Architecture

The model uses an LSTM layer to capture sequential dependencies in the input text. It is followed by a Dense layer and a softmax activation to output probabilities for each possible next word.

### Model Architecture:
```python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

# Build the model
model = Sequential()
model.add(LSTM(128, input_shape=(WORD_LENGTH, len(unique_words))))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))

# Compile the model
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history
```

- **LSTM layer**: Captures the sequential nature of text and learns long-term dependencies.
- **Dense layer**: Maps the LSTM output to a word prediction.
- **Softmax activation**: Produces a probability distribution over the possible next words.

## Requirements

To run the code, install the following Python packages:
```bash
pip install numpy pandas keras tensorflow scikit-learn
```

Additionally, you will need the Kaggle CLI to download the datasets:
```bash
pip install kaggle
```

## Training

To train the model, download the datasets from Kaggle, preprocess the text data, and feed it into the LSTM model for training.

```bash
# Download datasets
kaggle datasets download -d jainpooja/fake-news-detection
kaggle datasets download -d ronikdedhia/next-word-prediction
kaggle datasets download -d asifmahmudcste/next-word-prediction-bangla
kaggle datasets download -d zeyadsayedadbullah/synthetic-phone-call-conversation
```

Run the training script after setting up your environment:
```python
python train_model.py
```

The model is trained for 2 epochs with a batch size of 128. It uses the RMSprop optimizer and categorical cross-entropy as the loss function.



Special thanks to the authors of the datasets:
- **Pooja Jain** for the Fake News Detection dataset
- **Ronik Dedhia** for the English Next Word Prediction dataset
- **Asif Mahmud** for the Bangla Next Word Prediction dataset
- **Zeyad Sayed Abdullah** for the Synthetic Phone Call Conversation dataset
