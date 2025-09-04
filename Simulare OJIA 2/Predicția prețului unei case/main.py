# Importăm biblioteca pandas pentru a manipula datele
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor


# Citim fișierul CSV cu datele de antrenament și afișăm primele 5 rânduri
df_train = pd.read_csv('train_data.csv')
df_train.head()

# Citim fișierul CSV cu datele de test și afișăm primele 5 rânduri
df_test = pd.read_csv('test_data.csv')
df_test.head()


df_test['Task1'] = df_test['Square_Footage'] + df_test['Garage_Size'] + df_test['Lot_Size']

df_test['Garage_to_Room_Ratio'] = df_test['Garage_Size'] / df_test['Total_Rooms']
df_test['Task2'] = df_test['Garage_to_Room_Ratio']

df_test['Task3'] = (df_test['Solar_Exposure_Index'] - df_test['Vibration_Level']) / df_test['Magnetic_Field_Strength']

median = df_train['Square_Footage'].mean()
df_test['Task4'] = abs(df_test['Square_Footage'] - median)


# Task 5
df_train.fillna(df_train.mean())
df_test.fillna(df_test.mean())

features = ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size',
            'Garage_Size', 'Footage_to_Lot_Ratio', 'Total_Rooms', 'Age_of_House',
            'Garage_to_Footage_Ratio', 'Avg_Room_Size', 'House_Orientation_Angle',
            'Street_Alignment_Offset', 'Solar_Exposure_Index', 'Magnetic_Field_Strength', 'Vibration_Level']

x_train = df_train[features]
y_train = df_train['Price']


# RandomForestRegressor(n_estimators=69) 570
# LinearRegression() # MAE 271
# Ridge # a luat 100pct
model = Ridge(alpha=1.0)
model.fit(x_train, y_train)

x_test = df_test[features]
df_test['Task5'] = model.predict(x_test)



# Inițializăm o listă goală pentru a stoca rezultatele
result = []

# Iterăm prin fiecare rând al setului de date de test
for _, row in df_test.iterrows():
    # Iterăm prin subtasks (Task1 până la Task5)
    for subtask_id in range(1, 6):
        # Adăugăm un dicționar cu valorile corespunzătoare fiecărui subtask
        result.append({
            'subtaskID': subtask_id,  # ID-ul subtask-ului
            'datapointID': row['ID'],  # ID-ul datapoint-ului din rândul curent
            'answer': row[f'Task{subtask_id}']  # Răspunsul pentru subtask-ul curent
        })

# Creăm un DataFrame cu rezultatele obținute
df_output = pd.DataFrame(result)

# Afișăm primele 5 rânduri din DataFrame-ul rezultat
print(df_output.head())

# Salvăm rezultatele într-un fișier CSV pe care să-l putem apoi submite pe platformă
df_output.to_csv('sample_output.csv', index=False)
