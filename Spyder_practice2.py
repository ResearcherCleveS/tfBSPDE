#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 19:07:58 2025

@author: sylvestercleveland
"""

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#%config InlineBackend.figure_formats='svg'

x = np.linspace(-20, 20, 256)
f = np.sin(x)/x
plt.plot(x, f)

frqz = np.zeros(len(f))

for j, i in enumerate(f):
    if j%9 == 0:
        frqz[j] = i

plt.plot(x, frqz, color='tab:red', linewidth=0.75)        
import streamlit as st
#import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
#from scipy.stats import norm
#from scipy.optimize import brentq
#from scipy.interpolate import griddata
#import plotly.graph_objects as go

# st.title('Implied Volatility Surface')
#st.title('Time Fractional Black - Scholes Partial Differential Equation (tfBSPDE)')

# data = pd.read_csv('Closed Loop Forecast Data.csv')
data = pd.DataFrame(ClosedLoopForecastDatacsv)
fig, ax = plt.subplots(3, 1, figsize=(12, 10))
#for i in range(0, data.columns.size):
#    ax[i].plot(data[i+1].values);
plt.plot(data[0].values);
datalist = []
for i in data[0].values[1:-1]:
    datalist.append(float(i))
print(datalist)
plt.plot(datalist);
channel_1 = pd.Series(datalist)
print(channel_1)
plt.plot(channel_1, color='b');
# %%
import streamlit as st
st.title("Author Sylvester Cleve")