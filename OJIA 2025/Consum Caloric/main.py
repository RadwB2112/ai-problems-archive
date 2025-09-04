import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

df_train = pd.read_csv('train_data.csv')
df_test  = pd.read_csv('test_data.csv')

# === Subtask 1-4:
subtask1 = len(df_train)
subtask2 = len(df_train.loc[df_train['Gender'].str.lower() == 'male'])
subtask3 = df_train['Duration'].mean()
subtask4 = len(df_train.loc[df_train['Age'] >= 75])

df14 = pd.DataFrame({
    'subtaskID': [1, 2, 3, 4],
    'datapointID': [1, 1, 1, 1],
    'answer': [subtask1,
               np.round(subtask2, 0), 
               np.round(subtask3, 2), 
               np.round(subtask4, 0)]
})

# === Preprocesare date
df_train.fillna(df_train.mean(numeric_only=True), inplace=True)
df_test.fillna(df_test.mean(numeric_only=True), inplace=True)
df_train['Gender'].fillna('female', inplace=True)
df_test['Gender'].fillna('female', inplace=True)

X = df_train.drop(columns=['User_ID', 'Calories'], errors='ignore')
y = df_train['Calories']

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Model Subtask 5
model5 = Pipeline([
    ('pre', preprocessor),
    ('reg', HistGradientBoostingRegressor(
        max_iter=10000,
        learning_rate=0.01,
        max_depth=8,
        l2_regularization=2.0,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=42
    ))
])

X_train5 = X[df_train['Subtask']==5] if 'Subtask' in df_train.columns else X
y_train5 = y[df_train['Subtask']==5] if 'Subtask' in df_train.columns else y
model5.fit(X_train5, y_train5)

# Subtask 6
model6 = Pipeline([
    ('pre', preprocessor),
    ('reg', HistGradientBoostingRegressor(
        max_iter=10000,
        learning_rate=0.01,
        max_depth=8,
        l2_regularization=2.0,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=99  # alt seed
    ))
])

X_train6 = X[df_train['Subtask']==6] if 'Subtask' in df_train.columns else X
y_train6 = y[df_train['Subtask']==6] if 'Subtask' in df_train.columns else y
model6.fit(X_train6, y_train6)

# Preds
outputs = []
for sub_id, model in zip([5, 6], [model5, model6]):
    test_sub = df_test[df_test['Subtask'] == sub_id]
    x_test   = test_sub.drop(columns=['User_ID', 'Subtask'], errors='ignore')
    preds    = model.predict(x_test)
    outputs.append(pd.DataFrame({
        'subtaskID':   sub_id,
        'datapointID': test_sub['User_ID'],
        'answer':      np.round(preds, 0)
    }))

df5, df6 = outputs
final = pd.concat([df14, df5, df6], ignore_index=True)
final.to_csv('output.csv', index=False)
