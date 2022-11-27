from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_column = "WorkLifeBalance"
columns = ["WorkLifeBalance", "Department", "Gender","JobLevel", "JobRole", "JobSatisfaction", "MonthlyIncome", 
           "OverTime", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion"]
data = pd.read_csv('data.csv', usecols=columns)
columns_in_order = data.columns.values

print('Discretizar valores de output a 0 - 1...')
data[output_column] = pd.cut(data[output_column], bins=2, labels=False)

data[output_column].value_counts().plot(kind='bar', title="WorkLifeBalance counts")
plt.show()

print('Convertir los strings a valores numéricos...')
for column in data.columns:
    if data[column].dtype == 'O':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])

print('Estandarizar los datos...')
data_scaled = StandardScaler().fit_transform(data)

df = pd.DataFrame(data_scaled, columns=columns_in_order)

print(df[output_column])

x_train, x_test, y_train, y_test = train_test_split(
    df.drop(output_column, axis=1),
    df[output_column],
    test_size=0.2,
    random_state=10,
    stratify=df[output_column]
)

print(x_train.vallues)
print(y_train.value_counts())


# output = np.array([[x] for x in df.loc[:,output_column].to_list()])
# df = df.drop(output_column, axis=1)
# columns_in_order = np.delete(columns_in_order, np.where(columns_in_order == output_column))
# input = df[columns_in_order].values

# print(input.shape)
# print(input)
# print(output.shape)
# print(output)

# scaler = MinMaxScaler()
# input_train_scaled = scaler.fit_transform(input_train)
# output_train_scaled = scaler.fit_transform(output_train)
# input_test_scaled = scaler.fit_transform(input_test)
# output_test_scaled = scaler.fit_transform(output_test)

# np.save('data/input_train_scaled.npy', input_train_scaled)
# np.save('data/output_train_scaled.npy', output_train_scaled)
# np.save('data/input_test_scaled.npy', input_test_scaled)
# np.save('data/output_test_scaled.npy', output_test_scaled)
# np.save('data/input_pred.npy', input_pred)