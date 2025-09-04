import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')


# Construiește rezultatul final
results = []


# Subtask 1 – un singur datapoint
valoare = round(df_test['traffic_level'].mean(), 2)
results.append((1, 1,  valoare))

# Subtask 2 – un singur datapoint
valoare = round(df_test['traffic_level'].std(), 2)
results.append((2, 1, valoare))



# Subtask 3 – predicțiile pe test set
x_train = df_train[['distance_km', 'package_weight_kg', 'traffic_level']]
y_train = df_train['on_time']

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)  

x_test = df_test[['distance_km', 'package_weight_kg', 'traffic_level']]
x_test_scaled = scaler.transform(x_test)

predictions = model.predict(x_test_scaled)


for idx, row in enumerate(df_test.itertuples()):
    results.append((3, row.id, int(predictions[idx])))


# Salveaza rezultatele
df_output = pd.DataFrame(results, columns=['subtaskID', 'datapointID', 'answer'])
df_output.to_csv('final_submission.csv', index=False)
