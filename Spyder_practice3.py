#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 19:07:58 2025

@author: sylvestercleveland
"""

import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go
#import pandas as pd
import matplotlib.pyplot as plt
#%config InlineBackend.figure_formats='svg'

import os
from apikey import apikey
from langchain import llms
from langchain.llms import OpenAI
from langchain import agents #creat_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv 

# os.environ['OPEN_API_KEY'] = apikey
# load_dotenv(find_dotenv())

# #Initialize the key in session state if 'clicked' not in st.session_state:
# if 'clicked' not in st.session_state:
#     st.session_state.clicked={1: False}

# #Function to update the value in session state:
# def clicked(button):
#     st.session_state.clicked[button]=True
# st.button("Let's get started", on_click = clicked, args=[1])
# if st.session_state.clicked[1]:
#     st.header('Exploratory Data Analysis Part')
#     st.subheader('Solution')

#     user_csv = st.file_uploader("Upload your file here", type="csv")

#     if user_csv != None:
#         user_csv.seek(0)
#         df = pd.read_csv(user_csv, low_memory=False)

# llm model
# llm = OpenAI(temperature=0)
st.title("Author ResearcherCleveS 👑")
if st.button("**Abstract**"):
    st.header("Hinton & ResearcherCleveS are Successes!")
    st.subheader("*KC [Special Member♥️💠]*")
    st.success("**Hinton & ResearcherCleveS, KC are Successes!**")
    st.caption("*<p style = 'text:align-center'> Made w/ ♥️ by Vester Et als </p>*", unsafe_allow_html=True)
    st.divider()
with st.sidebar:
    st.title("**AI plus Partial Differential Eqns ie Physics Informed Neural Network (PINN) demonstration.**")
    st.write("**We'll explore implementing a PINN to optimize the efficiency and application of the Black Scholes Formula.**")
    # st.caption("**Fri June 6 2:54 PM [ Brain on Sir C🐫rter 🍳 like drugs ] Sike!**")
    st.caption("**<p style='text:align-center'>Fri June 6 2:54 PM NYC Time</p>**", unsafe_allow_html=True) 
    st.caption("*<p style='text:align-center'> KC [Special Memeber♥️💠] </p>*", unsafe_allow_html=True)
    #Initialize the key in session state if 'clicked' not in st.session_state:
    if 'clicked' not in st.session_state:
        st.session_state.clicked={1: False}

    #Function to update the value in session state:
    def clicked(button):
        st.session_state.clicked[button]=True
    st.button("*Add File*", on_click = clicked, args=[1])
    if st.session_state.clicked[1]:
        st.header('Stock Market Data Analysis')
        st.subheader('Solution')

        user_csv = st.file_uploader("Upload your file here", type="csv")

        if user_csv != None:
            user_csv.seek(0)
            df = pd.read_csv(user_csv, low_memory=False)
            st.subheader("*Data Description*")
            st.write(df.describe())
            # Doesn't work w/ out custom function and programming:
            #     st.file_uploader("Upload your file here", type="csv")

    # Doesn't work w/ out custom function and programming: if st.button("*Add File*"):
    #     user_csv = st.file_uploader("Upload your file here", type="csv")
    # st.file_uploader("Upload your file here", type="csv")
    # st.subheader("*Data Description*")
    # st.write(df.describe())
    st.divider()
    with st.expander("*Mermaid Cup AKA Expander Name*"):
        st.divider()
        st.write("*KC [Mood♥️💠]*")
    with st.expander("*KC [Mood♥️💠]*"):
        st.success("*K. Corera AKA KC [Mood♥️💠]*")
    with st.expander("What are the steps of EDA"):
        st.write('*What are the steps of EDA*')
    
# if st.button("Abstract"):
# st.success("**Hinton & ResearcherCleveS, KC are Successes!**")
# st.divider()

x = np.linspace(-20, 20, 256)
f = np.sin(x)/(1e-8+x)

fig = plt.figure()
plt.plot(x, f)
plt.title('Sinc function', x=.15, y=.88, color='tab:blue')
st.pyplot(fig)

frqz = np.zeros(len(f))

for j, i in enumerate(f):
    if j%9 == 0:
        frqz[j] = i

# plt.plot(x, frqz, color='tab:red', linewidth=0.75)
st.line_chart(frqz)
plt.title('Sinc function & select values', x=.23, y=.88, color='tab:red', fontsize=10.5)
st.pyplot(fig)
# import streamlit as st
#import yfinance as yf
# import pandas as pd
# import numpy as np
# from datetime import timedelta
#from scipy.stats import norm
#from scipy.optimize import brentq
#from scipy.interpolate import griddata
#import plotly.graph_objects as go

# st.title('Implied Volatility Surface')
#st.title('Time Fractional Black - Scholes Partial Differential Equation (tfBSPDE)')
st.subheader('MATLAB Wave Data')
st.success('ResearcherCleveS')
st.warning('Debugging')
st.error('🌀 Helene')
data = pd.read_csv('Closed Loop Forecast Data.csv')
data = pd.DataFrame(data)
fig, ax = plt.subplots(3, 1, figsize=(12, 10))
#for i in range(0, data.columns.size):
#    ax[i].plot(data[i+1].values);
#plt.plot(data[0].values);
datalist_1, datalist_2, datalist_3 = [], [], []
datalist = [datalist_1, datalist_2, datalist_3]
for i in range(0, len(data.columns)):
    for j in data.values[:, i]:
        datalist[i].append(float(j))
    datalist[i] = pd.DataFrame({f"Channel_{i+1}": datalist[i]})
    ax[i].plot(datalist[i]);
    ax[i].set_ylabel(f"Channel {i+1}", rotation=0, fontweight='bold')
    # ax[i].set_yticks([])
st.pyplot(fig)
# [st.line_chart(datalist[i]) for i in range(0, 3)]
for i in range(0, np.size(datalist, 0)):
    st.line_chart(datalist[i])
    plt.ylabel(f"Channel {i}", rotation=0, fontweight="bold")
st.success(np.shape(datalist))
st.success(np.size(datalist, 0))
np.random.seed(42)
st.write(np.random.randint(-11, 11, np.size(datalist, 0)))
np.random.seed(0)
st.write(np.random.randint(-11, 11, np.size(datalist, 0)))
df = np.zeros((len(datalist[0]), 3))
for i in range(0, 3):
    df[:, i:i+1] = datalist[i]
    
df = pd.DataFrame({"Channel_1": df[:, 0], "Channel_2": df[:, 1], "Channel_3": df[:, 2]})
st.info(f"**Testing my deep learning app! Here's some sample wave data.**")
with st.expander('Wave Data'):
    st.write('**Raw wave data samples**')
    df
