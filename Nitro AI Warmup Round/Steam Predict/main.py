
import pandas as pd
import numpy as np


df_train = pd.read_csv("train_data.csv")
df_test  = pd.read_csv("test_data.csv")

# Completare numerică cu media
df_train = df_train.fillna(df_train.mean(numeric_only=True))
df_test  = df_test.fillna(df_test.mean(numeric_only=True))

# Task 1
def get_avg_owners(owners):
    min, max = owners.split(' - ')
    min_i, max_i = int(min), int(max)
    return int((min_i + max_i) / 2)


df_train['Avg Owners'] = df_train['Estimated owners'].apply(get_avg_owners)
df_test['Avg Owners'] = df_test['Estimated owners'].apply(get_avg_owners)


from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

train_genres = df_train['Genres'].fillna('').str.split(r'\s*,\s*')
test_genres  = df_test ['Genres'].fillna('').str.split(r'\s*,\s*')

genres_train_df = pd.DataFrame(mlb.fit_transform(train_genres), 
                               columns=[f"genre_{g}" for g in mlb.classes_], 
                               index=df_train.index)
genres_test_df  = pd.DataFrame(mlb.transform(test_genres), 
                               columns=genres_train_df.columns, 
                               index=df_test.index)

df_train = pd.concat([df_train, genres_train_df], axis=1)
df_test  = pd.concat([df_test,  genres_test_df],  axis=1)




df_train['like_ratio']    = df_train['Positive'] / (df_train['Positive'] + df_train['Negative'] + 1)
df_test ['like_ratio']    = df_test ['Positive'] / (df_test ['Positive'] + df_test ['Negative'] + 1)

df_train['rec_per_owner'] = df_train['Recommendations'] / (df_train['Avg Owners'] + 1)
df_test ['rec_per_owner'] = df_test ['Recommendations'] / (df_test ['Avg Owners'] + 1)

df_train['likes_per_owner'] = df_train['Positive'] / (df_train['Avg Owners'] + 1)
df_test ['likes_per_owner'] = df_test ['Positive'] / (df_test ['Avg Owners'] + 1)

df_train['log_owners'] = np.log1p(df_train['Avg Owners'])
df_test ['log_owners'] = np.log1p(df_test ['Avg Owners'])

df_train['net_sentiment'] = df_train['Positive'] - df_train['Negative']
df_test ['net_sentiment'] = df_test ['Positive'] - df_test ['Negative']

df_train['owners_x_rating'] = df_train['Avg Owners'] * df_train['Metacritic score']
df_test['owners_x_rating'] = df_test['Avg Owners'] * df_test['Metacritic score']



cols_to_drop = ['AppID','Name','Release date','Estimated owners','Genres','Price','Publishers']
X = df_train.drop(columns=cols_to_drop)
y = df_train['Price'] 

# Log-transform pentru stabilitate
y = np.log1p(y)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.13, random_state=42, shuffle=True # conteaza mult si cat pun la split
)

num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(include="object").columns.tolist()

# print("Numeric features:", num_cols)
# print("Categorical features:", cat_cols)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor


# 3) Build ColumnTransformer using those lists directly:
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

""" nu cred ca merita un Stack
from sklearn.ensemble import StackingRegressor
base_models = [
    ('rf', HistGradientBoostingRegressor(max_iter=10000, learning_rate=0.05, random_state=42)),
    ('cat', CatBoostRegressor(depth=6, iterations=300, l2_leaf_reg=3, learning_rate=0.03, verbose=0))
]
stack = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

"""

model = Pipeline([
    ('pre', preprocessor),
    ('cat', CatBoostRegressor(depth=6, iterations=300, l2_leaf_reg=3, learning_rate=0.03, verbose=0)) # parametri din gridsearch


    ## ('stack', stack)
    # ('ensemble', VotingRegressor([
    #    ('cat', CatBoostRegressor(depth=6, iterations=300, l2_leaf_reg=3, learning_rate=0.03, verbose=0)) ,
    #    ('lr', LinearRegression(n_jobs=-1))
    # ]))
])
"""     CatBoostRegressor(
                iterations=4000,
                learning_rate=0.01,
                depth=6,
                l2_leaf_reg=3,
                random_strength=2,
                bagging_temperature=0.7,
                border_count=128,
                eval_metric='MAE',
                early_stopping_rounds=50,
                random_seed=42,
                verbose=0))"""
""" grid search 
from sklearn.model_selection import GridSearchCV
param_grid = {
    'cat__iterations':     [300, 500],
    'cat__depth':          [6, 10],
    'cat__learning_rate':  [0.03, 0.1],
    'cat__l2_leaf_reg':    [3, 5]
}
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
best = grid.best_estimator_
"""

model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error 

# Predicție și invers log-transform
y_pred = np.expm1(model.predict(X_val))
y_true = np.expm1(y_val)
print("MAE:",  mean_absolute_error(y_true, y_pred))

# Subtask 1

sub1 = pd.DataFrame({
    'subtaskID':   1,
    'datapointID': df_test['AppID'],
    'answer':      df_test['Avg Owners']
})


# Pregătire X_test ca mai sus:
cols_to_drop1 = ['AppID','Name','Release date','Estimated owners','Genres','Publishers']
X_test = df_test.drop(columns=cols_to_drop1)
y_test_pred = np.expm1(model.predict(X_test))

sub2 = pd.DataFrame({
    'subtaskID':   2,
    'datapointID': df_test['AppID'],
    'answer':      y_test_pred
})

final = pd.concat([sub1, sub2], ignore_index=True)

final['answer'] = final.apply(
    lambda row: f"{int(row.answer)}"
                if row.subtaskID == 1
                else f"{row.answer:.6f}",
    axis=1)

final.to_csv('submission_ai.csv', index=False)
