import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# === CITIRE DATE
df_train = pd.read_csv("train_data.csv")
df_test = pd.read_csv("test_data.csv")

# === SUBTASK 1
filter_barlad = df_test[(df_test["City A"] == "Barlad") & (df_test["Weather"] == "Fog")]
val1 = len(filter_barlad)

# === PRELUCRARE DATE PENTRU SUBTASK 2
categorical_cols = ["City A", "City B", "Weather"]
numeric_cols = ["Distance", "Time of Day", "Traffic", "Road Quality", "Driver Experience"]

train = df_train.copy()
test = df_test.copy()

# Label Encoding
encoder_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    encoder_dict[col] = le

# === FEATURES & TARGET
X_train = train[categorical_cols + numeric_cols]
y_train = train["deliver_time"]
X_test = test[categorical_cols + numeric_cols]

# === ANTRENARE MODEL
model = HistGradientBoostingRegressor(
    max_iter=500,
    max_depth=15,
    min_samples_leaf=3,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# === PREDICTII
y_test_pred = model.predict(X_test)

results = []
results.append((1, 1, val1)) # Subtask 1

# Subtask 2
for i, row in df_test.iterrows():
    results.append((2, row["ID"], y_test_pred[i]))

# Output
df_output = pd.DataFrame(results, columns=["subtaskID", "datapointID", "answer"])
df_output.to_csv("output.csv", index=False)
