# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:20:12 2025

@author: yokut
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Input, Dense, Conv1D, MaxPool1D, ZeroPadding1D
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#===============DATA SELECTION===========
st.write("-------------------")
st.write("Data Selection")
st.write("-----------------------")
dataframe = pd.read_csv("C:/project/work/spam_text_message.csv")
st.write(dataframe.head(10))

#===============PREPROCESSING==============
#cheking missing values
st.write("----------------------")
st.write("Handling missing values")
st.write("----------------------")
st.write(dataframe.isnull().sum())

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataframe['ham'])

X = tokenizer.texts_to_sequences(dataframe['ham'])
Y = dataframe['ham']

# Pad the sequences
max_length = 200
padded_X = pad_sequences(X, maxlen=max_length)

# Label encoding
label_encoder = LabelEncoder()
padded_X = np.array(padded_X)
Y = label_encoder.fit_transform(Y)

# Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(padded_X, Y, test_size=0.25, random_state=2)

st.write("-----------------")
st.write("Data splitting")
st.write("-----------------")
st.write("Total No of input data :", dataframe.shape[0])
st.write("Total No of training data :", X_train.shape[0])
st.write("Total no of testing data :", X_test.shape[0])

#==========CLASSIFICATION===============
st.write("-----------------------------------")
st.write("Convolutional Neural Network")
st.write("-------------------------------")

#== CNN LAYERS ==
# Create the CNN model
inp = Input(shape=(max_length, 1))
x = ZeroPadding1D(padding=1)(inp)  # Add zero padding
conv = Conv1D(filters=2, kernel_size=2)(x)
pool = MaxPool1D(pool_size=2)(conv)
flat = Flatten()(pool)
dense = Dense(1, activation='sigmoid')(flat)  # Add sigmoid activation
model = Model(inputs=inp, outputs=dense)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ==SUMMARIZE==
st.write(model.summary())

# Reshape the data for the CNN
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

# ==FIT THE LAYERS==
history = model.fit(X_train_cnn, Y_train, epochs=10, batch_size=15, verbose=1, validation_split=0.2)

# Evaluate the model
Eval = model.evaluate(X_test_cnn, Y_test, verbose=1)[1] * 100
his = history.history['accuracy']
acc_cnn = max(his) * 100  # Multiply by 100
loss_cnn = min(his)  # Use min instead of max

st.write("---------------------------")
st.write("Performance Analysis")
st.write("-------------------------")
st.write("1. Accuracy=", acc_cnn, '%')
st.write("2. Error Value=", loss_cnn)

# =====VALIDATION GRAPH===
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Validation Graph')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Visualize the data
sns.countplot(x='ham', data=dataframe)
plt.show()

# Decision Tree Classifier
X_train_dt = X_train
X_test_dt = X_test
dt = DecisionTreeClassifier()
dt = dt.fit(X_train_dt, Y_train)

# Make predictions
y_pred = dt.predict(X_test_dt)

# Print the predictions
st.write("-----------------------------")
st.write("Prediction")
st.write("-----------------------")
st.write("Accuracy:", accuracy_score(Y_test, y_pred))
st.write("Classification Report:")
st.write(classification_report(Y_test, y_pred))


