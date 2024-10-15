import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load the wine dataset
wine_data = load_wine()
X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
y = pd.Series(wine_data.target)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Evaluate each model using cross-validation
accuracy_results = {}

for model_name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    accuracy_results[model_name] = scores

# Create a DataFrame for easy plotting
results_df = pd.DataFrame(accuracy_results)

# Plot boxplot for model comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df)
plt.title('Model Comparison: Accuracy')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Identify the best model based on mean accuracy
mean_accuracies = results_df.mean().sort_values(ascending=False)
best_model_name = mean_accuracies.idxmax()
best_model_accuracy = mean_accuracies.max()

print(f"The best model is: {best_model_name} with an accuracy of {best_model_accuracy:.4f}")
