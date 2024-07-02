import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import eli5 
from eli5.sklearn import PermutationImportance

data = pd.read_csv('H:\\sem 6\\XAI\\fifa.csv')
y = (data['Man of the Match'] == "Yes") # Convert from string "Yes"/"No" to binary


feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=100, random_state=0).fit(train_X, train_y)

#print(X)

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y) 
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
#iris dataset
"""
iris = load_iris()
X, y = iris.data, iris.target

y_binary = (y == 1).astype(int)

clf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y_binary)

perm_importance = PermutationImportance(clf, random_state=0).fit(X, y_binary)

eli5.show_weights(perm_importance, feature_names=iris.feature_names)"""
"
