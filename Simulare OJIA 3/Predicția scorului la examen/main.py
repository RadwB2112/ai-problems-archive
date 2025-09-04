# Importăm biblioteca pandas pentru a manipula datele
import pandas as pd 

# Citim fișierul CSV cu datele de antrenament și afișăm primele 5 rânduri
df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')

# Adăugăm coloane noi în setul de date de test pentru fiecare subtask
med = df_train['Hours_Studied'].mean()
df_test['Subtask1'] = abs(df_test['Hours_Studied'] - med)


df_test['Subtask2'] = df_test['Sleep_Hours'].apply( # Subtask2 va avea un rezultat de tip boolean
    lambda x : True if x < 7 else False
) 


train_score = df_train['Previous_Scores']
def f1(test_score):
    return (train_score >= test_score).sum()
df_test['Subtask3'] = df_test['Previous_Scores'].apply(f1)  # Subtask3 va avea un rezultat de tip întreg


train_score1 = df_train['Motivation_Level'] 
def f2(test_score1):
    return (train_score1 == test_score1).sum()
df_test['Subtask4'] = df_test['Motivation_Level'].apply(f2)  # Subtask4 va avea un rezultat de tip întreg


# Task 5 - prediction model


df_train = df_train.fillna(df_train.mean(numeric_only=True))
df_test = df_test.fillna(df_test.mean(numeric_only=True))

from sklearn.preprocessing import LabelEncoder, StandardScaler

for column in df_train.columns:
    if df_train[column].dtype == 'object':
        le = LabelEncoder()
        df_train[column] = le.fit_transform(df_train[column].astype(str)) # only fit once 
        df_test[column] = le.transform(df_test[column].astype(str))

x = df_train.drop(columns=['ID', 'Exam_Score'])
y = df_train['Exam_Score']



from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=69)



scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

#model = RandomForestClassifier(n_estimators=100, random_state=69) # mae 1.48
#model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=69) # mae 1.81
#model = LinearRegression() # mae 1.14
#model = LogisticRegression(class_weight='balanced') # mae >= 2.0
model1 = GradientBoostingRegressor(random_state=69)


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(model1, param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(x_train_scaled, y_train)
model = grid_search.best_estimator_



x_test = df_test[x_train.columns]
x_test_scaled = scaler.transform(x_test)
df_test['Subtask5'] = model.predict(x_test_scaled) # Subtask5 va avea un rezultat de tip float (aici vom adăuga predicțiile)



# Inițializăm o listă goală pentru a stoca rezultatele
result = []

# Iterăm prin fiecare rând al setului de date de test
for _, row in df_test.iterrows():
    # Iterăm prin subtasks (Subtask1 până la Subtask5)
    for subtask_id in range(1, 6):
        # Adăugăm un dicționar cu valorile corespunzătoare fiecărui subtask
        result.append({
            'subtaskID': subtask_id,  # ID-ul subtask-ului
            'datapointID': row['ID'],  # ID-ul datapoint-ului din rândul curent
            'answer': row[f'Subtask{subtask_id}']  # Răspunsul pentru subtask-ul curent
        })

# Creăm un DataFrame cu rezultatele obținute
df_output = pd.DataFrame(result)

# Afișăm primele 5 rânduri din DataFrame-ul rezultat
df_output.head()

# Salvăm rezultatele într-un fișier CSV pe care să-l putem apoi submite pe platformă
df_output.to_csv('submission.csv', index=False)
