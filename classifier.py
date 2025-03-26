# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:10:51 2025

@author: yokut
"""

import streamlit as st
import nltk
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#download the required NLTK data if you haven't already
nltk.download('names')
nltk.download('wordnet')

#Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

#Define a function to preprocess the text data
def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

#Load the spam and not spam message datasets
spam_messages = ["Win a free iphone","your account has been compromised","make money fast","your free ringtone buy later",]
not_spam_messages = ["THanks for your subscription","order confirmation","password reset","welcome email""No calls,message,missed calls","Dont worry.i guess he's busy",]

#Create a list of labeled messages
labeled_messages = [(message, 'spam') for message in spam_messages] + [(message, 'not spam') for message in not_spam_messages]

#Preprocess the text data
preprocessed_messages = [(preprocess_text(message), label) for message, label in labeled_messages]

#Split the preprocessed messages into training and testing sets
train_messages, test_messages, train_labels, test_labels = train_test_split([message for message, label in preprocessed_messages], [label for message, label in preprocessed_messages], test_size=0.2, random_state=42)

#Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

#Fit the vectorizer to the training messages and transform both the training and testing messages
X_train = vectorizer.fit_transform(train_messages)
y_train = train_labels
X_test = vectorizer.transform(test_messages)
y_test = test_labels

#Train a Multinomial Naive Bayes classifier on the training data
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

#Make predictions on the testing data
y_pred = classifier.predict(X_test)

#Evaluate the classifier performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:')
print(classification_report(y_test, y_pred))

#Create a Streamlit app
st.title('Spam or Not Spam Message Classifier')

#Create a text input field for the user to enter their message
message = st.text_input('Enter your message')

#Create a button to classify the message
if st.button('Classify Message'):
    # Preprocess the user's message
    preprocessed_message = preprocess_text(message)

    # Transform the preprocessed message into a vector
    message_vector = vectorizer.transform([preprocessed_message])

    # Make a prediction on the user's message
    prediction = classifier.predict(message_vector)

    # Display the prediction
    if prediction[0] == 'spam':
        st.write('This message is likely spam.')
        import subprocess
        subprocess.run(['streamlit','run','reporteddata.py'])
    else:
        st.write('This message is likely not spam.')
        
    

