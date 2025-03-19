import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create a Small Dataset
data = {
    'Temperature': [30, 25, 20, 15, 10, 35, 40, 22],
    'Humidity': [80, 60, 50, 70, 90, 40, 30, 55],
    'WindSpeed': [10, 15, 5, 7, 12, 20, 25, 8],
    'Precipitation': [1, 0, 0, 1, 1, 0, 0, 1],  # 1 = Rain, 0 = No Rain
    'Weather': ['Rainy', 'Clear', 'Clear', 'Rainy', 'Rainy', 'Clear', 'Clear', 'Rainy']
}

df = pd.DataFrame(data)

# Encode Categorical Labels
df['Weather'] = df['Weather'].map({'Clear': 0, 'Rainy': 1})


X = df[['Temperature', 'Humidity', 'WindSpeed', 'Precipitation']]
y = df['Weather']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Decision Tree Model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['Clear', 'Rainy'], yticklabels=['Clear', 'Rainy'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualizing the Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=X.columns, class_names=['Clear', 'Rainy'], filled=True)
plt.show()
