import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Step 1: Load and preprocess the dataset
data = pd.read_csv('corona_dataset.csv')  # Load your dataset here
# Preprocess the dataset as per your requirements, handling missing values, encoding categorical variables, etc.

# Step 2: Define the structure of the Bayesian Network
model = BayesianModel([('Fever', 'COVID'), ('Cough', 'COVID'), ('Difficulty_Breathing', 'COVID'), 
                       ('Sore_Throat', 'COVID'), ('Fatigue', 'COVID'), ('Headache', 'COVID'), 
                       ('Age', 'COVID'), ('Sex', 'COVID'), ('Travel_History', 'COVID')])

# Step 3: Estimate parameters from the dataset
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Step 4: Perform inference
inference = VariableElimination(model)

# Example: Given symptoms, infer the probability of COVID-19
evidence = {'Fever': 'High', 'Cough': 'Yes', 'Difficulty_Breathing': 'Yes', 
            'Sore_Throat': 'No', 'Fatigue': 'Yes', 'Headache': 'No', 
            'Age': 'Adult', 'Sex': 'Male', 'Travel_History': 'No'}
probabilities = inference.query(['COVID'], evidence=evidence)
print(probabilities)
