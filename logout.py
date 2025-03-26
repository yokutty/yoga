# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:56:54 2025

@author: yokut
"""

import streamlit as st

# Create a session state to store the login status
if 'login_status' not in st.session_state:
    st.session_state.login_status = True

# Function to handle logout
def handle_logout():
    st.session_state.login_status = False

# Logout page
st.title('Logout Page')
st.write('You are currently logged in.')
if st.button('Logout'):
    handle_logout()
    st.write('You have been logged out.')

    import subprocess
    subprocess.run(['streamlit','run','int.py'])