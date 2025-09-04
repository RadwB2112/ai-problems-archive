import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')

# === PRELUCRARE DATE

val1 = len(df_train)
filter_data = df_train[df_train['Credit_Utilization_Ratio'] >= 25]
val2 = np.floor(filter_data['Monthly_Inhand_Salary'].mean())
val3 = len(df_train['Month'].unique())

end_with_20 = df_train['SSN'].str.endswith('20)')
val4 = len(df_train[end_with_20]['SSN'].unique())

# === ANTRENARE MODEL

numeric_columns1 = df_train.select_dtypes(include=['float64', 'int64']).columns
numeric_columns = [col for col in numeric_columns1 if col != 'Credit_Score']


df_train[numeric_columns] = df_train[numeric_columns].fillna(df_train[numeric_columns].median())
df_test[numeric_columns] = df_test[numeric_columns].fillna(df_test[numeric_columns].median())


x_train = df_train[numeric_columns]
y_train = df_train['Credit_Score'].astype(int)
x_test = df_test[numeric_columns]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)


# === PREDICTII (Subtask 5)
y_test_pred = model.predict(x_test)

# === CONSTRUIRE FORMAT FINAL CSV
results = []

# Subtask 1–4: cate o valoare unica per subtask
results.append((1, 1, val1))
results.append((2, 1, val2))
results.append((3, 1, val3))
results.append((4, 1, val4))

# Subtask 5: fiecare rand din test
for i, row in df_test.iterrows():
    results.append((5, row['ID'], int(y_test_pred[i])))

df_output = pd.DataFrame(results, columns=['subtaskID', 'datapointID', 'answer'])
df_output.to_csv('final_submission.csv', index=False)
