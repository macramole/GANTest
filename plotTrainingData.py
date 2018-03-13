#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 12:04:51 2018

@author: leandro
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from bokeh.plotting import figure, output_file, show

df = pd.read_csv("dcgan/saved_model/log.csv")
df

plt.plot( df.d_loss )
plt.plot( df.d_acc )
plt.plot( df.g_loss )
plt.legend()

sns.tsplot( df.d_loss )

p = figure(title="hola", plot_width = 1000)
#p.line( df.epoch, pd.rolling_mean(df.d_loss,1000) )
#p.line( df.epoch, pd.rolling_mean(df.d_acc,1000) )
#p.line( df.epoch, pd.rolling_mean(df.g_loss,1000) )
p.multi_line( [df.epoch,df.epoch,df.epoch], 
             [ pd.rolling_mean(df.d_loss,1000), 
              pd.rolling_mean(df.d_acc,1000), 
              pd.rolling_mean(df.g_loss,1000) ] )
show(p)

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot, init_notebook_mode, iplot

init_notebook_mode()

data = [
    go.Scatter(
        x = df.epoch,
        y = pd.rolling_mean(df.d_loss,1000),
        name = "d_loss"
    ),
    go.Scatter(
        x = df.epoch,
        y = pd.rolling_mean(df.d_acc,1000),
        name = "d_acc"
    ),
    go.Scatter(
        x = df.epoch,
        y = pd.rolling_mean(df.g_loss,1000),
        name = "g_loss"
    ),
]

iplot(data)



