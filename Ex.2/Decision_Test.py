import pandas as pd
from math import log2

data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain", "Sunny"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild", "Hot"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong", "Strong"],
    "Answer": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No", "No"]
}

df = pd.DataFrame(data)

def entropy(column):

    value_counts = column.value_counts()
    total_count = len(column)
    
    entropy = 0
    for count in value_counts:
        probability = count / total_count
        entropy -= probability * log2(probability)
    
    return entropy

def conditional_entropy(df, feature, target):
    conditional_entropy = 0
    total_count = len(df)

    for value in df[feature].unique():
        subset = df[df[feature] == value]
        probability_feature = len(subset) / total_count
        entropy_feature = entropy(subset[target])
        conditional_entropy += probability_feature * entropy_feature
    
    return conditional_entropy

answer_entropy = entropy(df['Answer'])
print("Entropy of Answer column:", answer_entropy)

outlook_conditional_entropy = conditional_entropy(df, 'Humidity', 'Answer')
print("Conditional entropy of Outlook with respect to Answer:", outlook_conditional_entropy)

information_gain_outlook = answer_entropy - outlook_conditional_entropy
print("Information gain of Outlook:", information_gain_outlook)
