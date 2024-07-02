import shap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model) # you can use deep explainer,Kernel Explainer as well based on model usage

shap_values = explainer.shap_values(X_test[0])

print("SHAP Values:", shap_values)

shap.summary_plot(shap_values, X_test[0], feature_names=iris.feature_names, class_names=iris.target_names)
