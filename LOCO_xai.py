
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
iris_df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                      names=["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"])

# Shuffle the dataset
iris_df = iris_df.sample(frac=1).reset_index(drop=True)

# Encode the target labels
label_encoder = LabelEncoder()
iris_df["class"] = label_encoder.fit_transform(iris_df["class"])

# Define features and target column
features = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
target = "class"

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(iris_df[features], iris_df[target])

# Predict on the whole dataset
original_predictions = rf_model.predict(iris_df[features])

# Function to calculate LOCO scores
def calculate_loco_scores(model, data, features, target):
    loco_scores = {}
    for feature in features:
        loco_predictions = []
        for i in range(len(data)):
            modified_data = data.drop([feature], axis=1)  # Removing the current feature
            loco_predictions.append(model.predict([modified_data.iloc[i]]))
        loco_scores[feature] = accuracy_score(data[target], loco_predictions)
    return loco_scores

# Calculate LOCO scores
loco_scores = calculate_loco_scores(rf_model, iris_df, features, target)

# Print LOCO scores
print("LOCO Scores:")
for feature, score in loco_scores.items():
    print(f"{feature}: {score}")


