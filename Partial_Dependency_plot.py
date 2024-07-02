import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt


data = pd.read_csv('H:\\sem 6\\XAI\\fifa.csv')
y = (data['Man of the Match'] == "Yes") # Convert from string "Yes"/"No" to binary


feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=100, random_state=0).fit(train_X, train_y)

features = [0,1,2,3,5,6]
target_class = 'Yes'
display = PartialDependenceDisplay.from_estimator(my_model, X, features, target=target_class, feature_names = val_X.columns.tolist())

fig, ax = plt.subplots(figsize=(10,8))
display.plot(ax=ax)

plt.show()
