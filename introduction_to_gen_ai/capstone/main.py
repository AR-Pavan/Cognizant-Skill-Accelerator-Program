import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.models import Sequential

# User Story 1: Task Automation with Supervised Learning

np.random.seed(42)
data = {
    "hours_studied": np.random.randint(1, 10, 100),
    "past_scores": np.random.randint(40, 100, 100),
    "passed": np.random.choice([0, 1], 100)
}
df = pd.DataFrame(data)

X = df[["hours_studied", "past_scores"]]
y = df["passed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# User Story 2: Clustering for Student Grouping
# Applying K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

sns.scatterplot(x=df["hours_studied"], y=df["past_scores"], hue=df["cluster"], palette="viridis")
plt.xlabel("Hours Studied")
plt.ylabel("Past Scores")
plt.title("Student Clustering Based on Learning Styles")
plt.show()

# User Story 3: Image Generation with GANs
latent_dim = 100

def build_generator():
    model = Sequential([
        Dense(256, activation="relu", input_dim=latent_dim),
        Dense(512, activation="relu"),
        Dense(1024, activation="relu"),
        Dense(28*28, activation="tanh"),
        Reshape((28, 28))
    ])
    return model

def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation="relu"),
        Dense(256, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

discriminator.trainable = False

gan = Sequential([generator, discriminator])
gan.compile(optimizer="adam", loss="binary_crossentropy")

print("GAN model summary:")
gan.summary()
