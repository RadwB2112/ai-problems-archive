# Importăm biblioteca pandas pentru a manipula datele
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score


# Citim fișierul CSV cu datele de antrenament și afișăm primele 5 rânduri
df_train = pd.read_csv('train_data.csv')
df_train.head()

# Citim fișierul CSV cu datele de test și afișăm primele 5 rânduri
df_test = pd.read_csv('test_data.csv')
df_test.head()


df_test['Task1'] = df_test['GFR'].apply(
    lambda x : "Normal" if x >= 90 else ("Mildly Decreased" if x >= 60 else "Unknown")
)

def f1(Q1, Q2, Q3, x):
    if x <= Q1: 
        return "Very Low"
    elif x <= Q2:
        return "Low"
    elif x <= Q3:
        return "High"
    else:
        return "Very High"
    
Q1 = df_train['Serum Creatinine'].quantile(0.25)
Q2 = df_train['Serum Creatinine'].quantile(0.50)
Q3 = df_train['Serum Creatinine'].quantile(0.75)

df_test['Task2'] = df_test['Serum Creatinine'].apply(
    lambda x : f1(Q1, Q2, Q3, x)
)

median = df_train['BMI'].median()
df_test['Task3'] = df_test['BMI'].apply(
    lambda x : 1 if x > median else 0
)

def f2(x):
    cnt = 0
    for index in df_train['T Stage']:
        if x == index:
            cnt += 1
    return cnt

df_test['Task4'] = df_test['T Stage'].apply(lambda x : f2(x))


# t_stage_cnt = df_train['T Stage'].value_counts()
# df_test['Task4'] = df_test['T Stage'].map(t_stage_cnt).fillna(0)



df_test['Tumor Size'] = df_test['Tumor Size'].fillna(df_test['Tumor Size'].mean(numeric_only=True))
df_train['Tumor Size'] = df_train['Tumor Size'].fillna(df_train['Tumor Size'].mean(numeric_only=True))


categorical_columns = df_test.select_dtypes(include=['object']).columns.tolist()
categorical_columns = [col for col in categorical_columns if col not in ['Status', 'ID', 'Task1', 'Task2', 'Task3', 'Task4']]

le = LabelEncoder()

for column in categorical_columns:
    df_train[column] = le.fit_transform(df_train[column])
    df_test[column] = le.transform(df_test[column])


df_train['Status'] = df_train['Status'].map({'Alive': 0, 'Dead': 1})   


x = df_train.drop(columns=['ID', 'Status'])
y = df_train['Status']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)

model2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
# model2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model2.fit(x_train_scaled, y_train)

x_test = df_test[x_train.columns]
x_test_scaled = scaler.transform(x_test)
df_test['Task5'] = model2.predict(x_test_scaled)
df_test['Task5'] = df_test['Task5'].map({0: 'Alive', 1: 'Dead'})


from sklearn.metrics import recall_score, f1_score

y_pred_train = model2.predict(x_train)

precision_dead = precision_score(y_train, y_pred_train, pos_label=1)
accuracy = accuracy_score(y_train, y_pred_train)
recall_dead = recall_score(y_train, y_pred_train, pos_label=1)
f1_dead = f1_score(y_train, y_pred_train, pos_label=1)

print(f"Precision for 'Dead' class: {precision_dead:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall for 'Dead' class: {recall_dead:.4f}")
print(f"F1-score for 'Dead' class: {f1_dead:.4f}")






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
df_output.to_csv('submission.csv', index=False)


















"""
# Adăugăm coloane noi în setul de date de test pentru fiecare subtask
df_test['Task1'] = df_test['GFR'].apply(
    lambda gfr: 'Normal' if gfr >= 90 else ('Mildly Decreased' if 60 <= gfr < 90 else 'Unknown')
)



def function(x, Q1, Q2, Q3):
    if x <= Q1:
        return "Very low"
    elif x <= Q2:
        return "Low"
    elif x <= Q3:
        return "High"
    else:
        return "Very High"

Q1 = df_train['Serum Creatinine'].quantile(0.25)
Q2 = df_train['Serum Creatinine'].quantile(0.50)
Q3 = df_train['Serum Creatinine'].quantile(0.75)

df_test['Task2'] = df_test['Serum Creatinine'].apply(lambda x : function(x, Q1, Q2, Q3))





median = df_train['BMI'].median()
df_test['Task3'] = df_test['BMI'].apply(
    lambda bmi: 1 if bmi > median else 0
)


    

t_stage_cnt = df_train['T Stage'].value_counts()
df_test['Task4'] = df_test['T Stage'].map(t_stage_cnt).fillna(0)
"""

"""

df_train['Tumor Size'] = df_train['Tumor Size'].fillna(df_train['Tumor Size'].mean())
df_test['Tumor Size'] = df_test['Tumor Size'].fillna(df_train['Tumor Size'].mean())

cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()
cat_cols = [col for col in cat_cols if col not in ['Status', 'ID']]
encoder = LabelEncoder()

for col in cat_cols:
    df_train[col] = encoder.fit_transform(df_train[col])
    df_test[col] = encoder.transform(df_test[col])

df_train['Status'] = df_train['Status'].map({'Alive': 0, 'Dead': 1})

features = [col for col in df_train.columns if col not in ['ID', 'Status']]

X_train = df_train[features]
y_train = df_train['Status']

X_test = df_test[features]

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

df_test['Task5'] = model.predict(X_test)
df_test['Task5'] = df_test['Task5'].map({0: 'Alive', 1: 'Dead'})

y_pred_train = model.predict(X_train)
precision_dead = precision_score(y_train, y_pred_train, pos_label=1)

"""

