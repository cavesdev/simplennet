from simplennet.neural_network import NeuralNetwork
import numpy as np
import pandas as pd
from sklearn import preprocessing

input_train_scaled = np.load('data/input_train_scaled.npy')
output_train_scaled = np.load('data/output_train_scaled.npy')
input_test_scaled = np.load('data/input_test_scaled.npy')
output_test_scaled = np.load('data/output_test_scaled.npy')
input_pred = np.load('data/input_pred.npy')

columns = ["Department", "Gender", "JobRole", "JobLevel", "JobSatisfaction", "MonthlyIncome", \
           "OverTime", "YearsSinceLastPromotion", "YearsInCurrentRole", "YearsAtCompany", "Attrition"]
data = pd.read_csv('data.csv', usecols=columns)

# Convertir los strings a valores numéricos
for column in data.columns:
    if data[column].dtype == 'O':
        le = preprocessing.LabelEncoder()
        data[column] = le.fit_transform(data[column])

# estandarizar los datos
data_scaled = data.copy()

for column in data_scaled.columns:
	data_scaled[column] = (data_scaled[column] - data_scaled[column].mean()) / data_scaled[column].std()
    
outputs = []
for out in data_scaled.loc[:,"Attrition"].to_list():
    outputs.append([out])

data_scaled = data_scaled.drop("Attrition", axis=1)
inputs = []

for i in range(data_scaled["Department"].size):
    new_set = []
    for column in data_scaled.columns:
        new_set.append(data_scaled[column].to_list()[i])
    inputs.append(new_set)


# print(inputs)
# print(outputs)
print(np.array(inputs).shape)
print(np.array(outputs).shape)
# print(data_scaled.to_string())

NN = NeuralNetwork()

NN.train(np.array(inputs), np.array(outputs), 10)
input_pred = np.array([1,1,2,2,3,4404,0,4,3,1])
print(NN.predict(input_pred))
NN.view_error_development()
# NN.test_evaluation(input_test_scaled, output_test_scaled)