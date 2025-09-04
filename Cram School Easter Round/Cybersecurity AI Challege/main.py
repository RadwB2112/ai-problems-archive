import pandas as pd

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split


# === Read the data
df_train = pd.read_csv("train_data.csv")
df_test = pd.read_csv("test_data.csv")


# === Subtask 1

value_subtask1 = []
for timestamp in df_test['Timestamp']:
    hour = int(timestamp[11:13])
    value_subtask1.append("PM" if hour >= 12 else "AM")

task1 = pd.DataFrame({
    'subtaskID': 1,
    'datapointID': df_test.ID,
    'answer':  value_subtask1
})


# === Subtask 2
df_train = df_train.fillna(df_train.mean(numeric_only=True))
df_test = df_test.fillna(df_test.mean(numeric_only=True))

x = df_train.drop(['ID', 'Timestamp', 'Attack Type'], axis=1)
y = df_train['Attack Type']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=69)

model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=69)
model.fit(x_train, y_train)


x_test = df_test.drop(['ID', 'Timestamp'], axis=1)
test_preds = model.predict(x_test)

task2 = pd.DataFrame({
    'subtaskID': 2,
    'datapointID': df_test.ID,
    'answer': test_preds
})

# === Final datatset
submission_df = pd.concat([task1, task2], ignore_index=True)
submission_df.to_csv("submission.csv", index=False)

