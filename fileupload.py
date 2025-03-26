# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:26:19 2025

@author: yokut
"""

import streamlit as st
import pandas as pd

# Create a Streamlit app
st.title('Upload Dataset')

# Create a file uploader
uploaded_file = st.file_uploader('C:/project/work/cleaned_sms_spam.csv', type='csv')

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    # Display the uploaded dataset
    st.write('Uploaded Dataset:')
    st.write(df)

    # Display the shape of the dataset
    st.write('Shape of the Dataset:')
    st.write(df.shape)

    # Display the columns of the dataset
    st.write('Columns of the Dataset:')
    st.write(df.columns)

    # Display the summary statistics of the dataset
    st.write('Summary Statistics of the Dataset:')
    st.write(df.describe())
    
    # Create a submit button
    if st.button('Submit'):
        # Code to be executed when the submit button is clicked
        st.write('Submit button clicked!')

        import subprocess
        subprocess.run(['streamlit','run','classifier.py'])