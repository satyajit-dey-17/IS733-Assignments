import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# Load dataset
data = pd.read_csv("hamspam.csv")

# Encode categorical features
data['Contains Link'] = data['Contains Link'].map({'Yes': 1, 'No': 0})
data['Contains Money Words'] = data['Contains Money Words'].map({'Yes': 1, 'No': 0})
data['Length'] = data['Length'].map({'Long': 1, 'Short': 0})

# Features and target
X = data[['Contains Link', 'Contains Money Words', 'Length']]
y = data['Class'].map({'Spam': 1, 'Ham': 0})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_prob = nb_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()