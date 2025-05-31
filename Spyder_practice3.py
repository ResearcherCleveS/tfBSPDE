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
import matplotlib.pyplot as plt


x = np.linspace(-20, 20, 256)
f = np.sin(x)/x
fig = plt.figure()
plt.plot(x, f)
st.pyplot(fig)

frqz = np.zeros(len(f))

for j, i in enumerate(f):
    if j%9 == 0:
        frqz[j] = i

plt.plot(x, frqz, color='tab:red', linewidth=0.75)
st.pyplot(fig)
st.title("Author Sylvester Cleve")
