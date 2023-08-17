import pandas as pd

# Load the dataset using absolute path
data = pd.read_csv("/Users/niraj/Python Programming/Big Data/lab 4/diabetes.csv")

print(data.head())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
knn_classifier.fit(X_train_scaled, y_train)


from sklearn.metrics import classification_report, accuracy_score

y_pred = knn_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)


import numpy as np

class CustomStandardScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
    
    def transform(self, X):
        z_scores = (X - self.mean) / self.std
        return z_scores
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# Assuming you have your dataset X_train and X_test
custom_scaler = CustomStandardScaler()

# Fit on training data and transform both training and testing data
X_train_scaled = custom_scaler.fit_transform(X_train)
X_test_scaled = custom_scaler.transform(X_test)


######################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the dataset (replace with your code)
data = pd.read_csv("/Users/niraj/Python Programming/Big Data/lab 4/diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Determine the best K value using cross-validation
k_values = list(range(1, 21))
accuracy_scores = []
for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_classifier, X_train_scaled, y_train, cv=5)
    accuracy_scores.append(np.mean(scores))

best_k = k_values[np.argmax(accuracy_scores)]
print("Best K value:", best_k)

# Visualization of accuracy scores
plt.plot(k_values, accuracy_scores, marker='o')
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Number of Neighbors")
plt.xticks(k_values)
plt.show()

# Train the model with the best K value
best_knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
best_knn_classifier.fit(X_train_scaled, y_train)

# Evaluate using confusion matrix
y_pred = best_knn_classifier.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Explain the accuracy of the model in a Markdown cell

##################################



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the dataset (replace with your code)
data = pd.read_csv("/path/to/diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Determine the best K value using cross-validation
best_k = 5  # Replace with the best K value you've found

# Train the model with the best K value
best_knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
best_knn_classifier.fit(X_train_scaled, y_train)

# Evaluate using confusion matrix
y_pred = best_knn_classifier.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Explain the accuracy of the model in a Markdown cell
markdown_text = """
### Model Accuracy

The K-Nearest Neighbors model was trained and evaluated on the Pima Indians Diabetes dataset. After performing 5-fold cross-validation, the mean accuracy was calculated to be {:.2f}% with a standard deviation of {:.2f}%. 

The confusion matrix above shows how well the model performed on the testing data. It provides insights into the true positive, true negative, false positive, and false negative predictions.

For further evaluation, the model was retrained using leave-one-out cross-validation. After conducting this analysis, the mean accuracy was found to be {:.2f}% with a standard deviation of {:.2f}%.
""".format(np.mean(cross_val_score(best_knn_classifier, X_train_scaled, y_train, cv=5)) * 100,
           np.std(cross_val_score(best_knn_classifier, X_train_scaled, y_train, cv=5)) * 100,
           np.mean(cross_val_score(best_knn_classifier, X_train_scaled, y_train, cv=LeaveOneOut())) * 100,
           np.std(cross_val_score(best_knn_classifier, X_train_scaled, y_train, cv=LeaveOneOut())) * 100)

print(markdown_text)
