import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("hamspam.csv")


# Encode categorical features
data['Contains Link'] = data['Contains Link'].map({'Yes': 1, 'No': 0})
data['Contains Money Words'] = data['Contains Money Words'].map({'Yes': 1, 'No': 0})
data['Length'] = data['Length'].map({'Long': 1, 'Short': 0})

# Features and target
X = data[['Contains Link', 'Contains Money Words', 'Length']]
y = data['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))

# KNN (K=2)
knn_model = KNeighborsClassifier(n_neighbors=2)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, knn_predictions))