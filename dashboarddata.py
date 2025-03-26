# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:13:31 2025

@author: yokut
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Create a dashboard title
st.title("Spam Message Dashboard")

# Create a sidebar with option

st.sidebar.title("Options")
option = st.sidebar.selectbox("Select an option", ["View Spam Messages", "View Non-Spam Messages", "View Statistics"])

# Dataset for the CSV file
data = {
    "text": [
        "Win a free iphone",
        "your account has been compromised",
        "make money fast",
        " free Order delivaries",
        "credit card offers",
        "your free ringtone buy later",
        "THanks for your subscription",
        "order confirmation",
        "password reset",
        "welcome email"
        "No calls,message,missed calls",
        "Dont worry.i guess he's busy",
        ],
    "target": [
        "Spam",
        "Spam",
        "Spam",
        "Spam",
        "Spam",
        "spam",
        "Not Spam",
        "Not Spam",
        "Not Spam",
        "Not spam",
        "Not spam",
        ],
}

# Create a DataFrame from the dataset
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("C:/project/work/cleaned_sms_spam.csv", index=False)

# Load the dataset from the CSV file
try:
    df = pd.read_csv("C:/project/work/cleaned_sms_spam.csv")
except FileNotFoundError:
    st.write("Error: The CSV file 'C:/project/work/cleaned_sms_spam.csv' was not found.") 
    exit()

# Separate spam and non-spam messages
spam_messages = df[df['target'].str.lower() == 'spam']
non_spam_messages = df[df['target'].str.lower() == 'not spam']



# View spam messages
if option == "View Spam Messages":
    st.write("Spam Messages:")
    st.write(spam_messages)

# View non-spam messages
elif option == "View Non-Spam Messages":
    st.write("Non-Spam Messages:")
    st.write(non_spam_messages)

# View statistics
elif option == "View Statistics":
    st.write("Statistics:")
    st.write("Total Spam Messages:", len(spam_messages))
    st.write("Total Non-Spam Messages:", len(non_spam_messages))

    # Create a pie chart
    labels = ['Spam Messages', 'Non-Spam Messages']
    sizes = [len(spam_messages), len(non_spam_messages)]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    st.pyplot()
    # Create a submit button
    if st.button('Submit'):
        # Code to be executed when the submit button is clicked
       st.write('Submit button clicked!')
       import subprocess
       subprocess.run(['streamlit','run','logout.py'])