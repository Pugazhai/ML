import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Sample documents and their corresponding labels
documents = [
    "This is a sample document about the naive Bayes classifier algorithm.",
    "Naive Bayes classifier is easy to implement and works well for text classification tasks.",
    "Text classification using the naive Bayes algorithm is popular in natural language processing.",
    "The output of the naive Bayes program depends on the input features and training data."
]
labels = [1, 1, 1, 0]  # 1 for documents about naive Bayes, 0 for others

# Convert the documents into a bag-of-words representation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Train a naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, labels)

# Test data
test_documents = [
    "This document is not related to naive Bayes.",
    "Naive Bayes algorithm is widely used for text classification."
]
true_labels = [0, 1]

# Convert test documents into bag-of-words representation
X_test = vectorizer.transform(test_documents)

# Predict labels for test documents
predicted_labels = classifier.predict(X_test)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
