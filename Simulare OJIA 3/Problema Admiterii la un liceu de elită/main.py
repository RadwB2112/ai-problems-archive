import pandas as pd


df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')

subtask1 = 'dif_NT-MEV'
subtask2 = 'loc-MEV'
subtask3 = 'status_admitere'

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


train, validation = train_test_split(df_train, test_size=0.2, random_state=0)

y_train = train[subtask3]
x_train = train.drop(columns=['id', 'gen', 'judet', subtask3])


y_validation = validation[subtask3]
x_validation = validation.drop(columns=['id', 'gen', 'judet', subtask3])

x_train = x_train.fillna(x_train.median())
x_validation = x_validation.fillna(x_train.median())
df_test = df_test.fillna(x_train.median())

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_validation = scaler.transform(x_validation)

model = LogisticRegression(max_iter=1000, random_state=69)
model.fit(x_train, y_train)

y_pred = model.predict(x_validation)
print(accuracy_score(y_validation, y_pred))


df_test[subtask1] = round(df_test['NT'] - df_test['MEV'], 2)
df_test[subtask2] = df_test['MEV'].rank(ascending=False, method='first').astype(int) # pt 16/20 la subtask2

""" 
df_test_sorted = df_test.sort_values(by=['MEV'], ascending=False) # pt punctag maxim la subtask 2 : 20 / 20
df_test_sorted[subtask2] = range(1, len(df_test_sorted) + 1)
df_test = df_test.merge(df_test_sorted[['id', subtask2]], on='id', how='left')"""

x_test = df_test.drop(columns=['id', 'gen', 'judet', subtask1, subtask2])
x_test = scaler.transform(x_test)
df_test[subtask3] = model.predict(x_test)


output1 = pd.DataFrame({
    'subtaskID': 1,
    'datapointID': df_test['id'],
    'answer': df_test[subtask1]
})
output2 = pd.DataFrame({
    'subtaskID': 2,
    'datapointID': df_test['id'],
    'answer': df_test[subtask2]
})
output3 = pd.DataFrame({
    'subtaskID': 3,
    'datapointID': df_test['id'],
    'answer': df_test[subtask3]
})
final_output = pd.concat([output1, output2, output3])
final_output.to_csv('output.csv', index=False)
