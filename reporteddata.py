# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:58:43 2025

@author: yokut
"""

import streamlit as st
import pandas as pd
import csv

def spam_message_report_page():
    st.title("Spam Message Report")

    # Create a form for the user to report a spam message
    with st.form("report_spam_message"):
        message = st.text_area("Message", height=100)
        sender = st.text_input("Sender")
        reason = st.text_input("Reason")

        # Create a submit button
        submitted = st.form_submit_button("Report Spam Message")

    if submitted:
        # Create a dictionary to store the reported spam message
        report = {
            "Message": [message],
            "Sender": [sender],
            "Reason": [reason]
        }

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(report)

        # Try to read the existing CSV file
        try:
            existing_df = pd.read_csv("C:/project/work/cleaned_sms_spam.csv")
            combined_df = pd.concat([existing_df, df])
            combined_df.to_csv("C:/project/work/cleaned_sms_spam.csv", index=False)
        except FileNotFoundError:
            # If the file doesn't exist, create a new one
            df.to_csv("spam_reports.csv", index=False)

        # Show a confirmation message to the user
        st.success("Spam message reported successfully!")

       
        import subprocess
        subprocess.run(['streamlit','run','historydata.py'])

def main():
    spam_message_report_page()

if __name__ == "__main__":
    main()


