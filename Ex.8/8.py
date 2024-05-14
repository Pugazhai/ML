import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 3

knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

for i in range(len(X_test)):
    if y_pred[i] == y_test[i]:
        print(f"Correct prediction: Actual - {iris.target_names[y_test[i]]}, Predicted - {iris.target_names[y_pred[i]]}")
    else:
        print(f"Wrong prediction: Actual - {iris.target_names[y_test[i]]}, Predicted - {iris.target_names[y_pred[i]]}")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
