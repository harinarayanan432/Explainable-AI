from pycebox.ice import ice, ice_plot
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestRegressor
from pdpbox import pdp

df=pd.read_csv("H:\sem 6\XAI\winequality-red.csv")

# Split features and target variable
x = df.drop('quality', axis=1)
y = df['quality']

# Train RandomForestRegressor model
model = RandomForestRegressor()
model.fit(x, y)
interest='citric acid'
ice_data = ice(x, interest, model.predict)
ice_plot(ice_data, linewidth=0.5, plot_points=False, color_by=None)
plt.title(f"ICE Plots for {interest}")
plt.xlabel(interest)
plt.ylabel('Predicted Outcome')
plt.show()
