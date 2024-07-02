import numpy as np
from anchor import anchor_tabular

# Generate some synthetic data for demonstration
np.random.seed(0)
X = np.random.rand(1000, 5)
y = (X[:, 0] > 0.5).astype(int)  # Binary classification based on first feature

# Train a simple classifier (e.g., decision tree) on the synthetic data
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(X, y)

# Define feature names
feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']

# Define class names
class_names = ['Class 0', 'Class 1']

# Initialize AnchorTabularExplainer
explainer = anchor_tabular.AnchorTabularExplainer(
    class_names=class_names,
    feature_names=feature_names,
    train_data=X
)

# Choose an instance to explain (e.g., the first instance)
instance_to_explain = X[0]

# Generate anchor explanation
explanation = explainer.explain_instance(
    instance_to_explain,
    classifier.predict,
    threshold=0.95  # Set the required precision of the anchor
)

# Print the anchor explanation
print('Anchor Explanation:')
print(explanation)

# Accessing attributes of the AnchorExplanation object
print("Anchor Features:", explanation.names())
print("Precision:", explanation.precision())
print("Coverage:", explanation.coverage())
print("Anchor:", explanation.names())
            
