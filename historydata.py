# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:05:31 2025

@author: yokut
"""

import streamlit as st
import pandas as pd

def spam_message_history_page():
    st.title("Spam Message History")

    # Read the spam message history data from a CSV file
    try:
        df = pd.read_csv("C:/project/work/cleaned_sms_spam.csv")
    except FileNotFoundError:
        st.write("No spam message history data found.")
        return

    # Display the DataFrame in a table
    st.table(df)

    # Create a submit button
    if st.button('Submit'):
        # Code to be executed when the submit button is clicked
       st.write('Submit button clicked!')

    
       import subprocess
       subprocess.run(['streamlit','run','dashboarddata.py'])
def main():
    spam_message_history_page()

if __name__ == "__main__":
    main()

