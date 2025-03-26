# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:42:34 2025

@author: yokut
"""


import streamlit as st

# Create a title for the welcome page
st.title("SMART FINDER: A SPAM DETECTION METHOD IN MESSAGE USING AI!.......")

# Add a brief introduction
st.write("Welcome To Our Project!!!!......")


# Add a button to navigate to the next page
if st.button("Get Started"):
    # Code to navigate to the next page

   import subprocess
   subprocess.run(['streamlit','run','int.py'])
# Add a background image (optional)
# st.markdown(
#     """
#     <style>
#     .reportview-container {
#         background: url("background.jpg")
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
