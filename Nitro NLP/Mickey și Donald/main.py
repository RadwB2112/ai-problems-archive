import pandas as pd, re

df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')
print(df_train.head())

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s\$]", "", text)
    return text


df_train['sample_clean'] = df_train['sample'].apply(preprocess)
df_test['sample_clean'] = df_test['sample'].apply(preprocess)


# ------------------------------------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = CountVectorizer(ngram_range=(1,2), min_df=2)
x_train = vectorizer.fit_transform(df_train['sample_clean'])
x_test = vectorizer.transform(df_test['sample_clean'])

# ------------------------------------------------------------------------



from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
"""
# ------------- dialectul
y_dialect = df_train['dialect']
clf_dialect = LogisticRegression(max_iter=1000)
clf_dialect.fit(x_train, y_dialect)

# ------------- categoria
y_category = df_train['category']
clf_category = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf_category.fit(x_train, y_category)

# ------------- prezicem
pred_dialect = clf_dialect.predict(x_test)
pred_category = clf_category.predict(x_test)
"""
# ---------------- dialectul
y_dialect = df_train['dialect']
clf_dialect = CatBoostClassifier(verbose=50)  
clf_dialect.fit(x_train, y_dialect)

# ---------------- categoria
y_category = df_train['category']
clf_category = CatBoostClassifier(verbose=0)
clf_category.fit(x_train, y_category)

# ---------------- predicții
pred_dialect = clf_dialect.predict(x_test)
pred_category = clf_category.predict(x_test)


# ------------- output 
df_out_dialect = pd.DataFrame({
    'subtaskID': '1',
    'datapointID': df_test['datapointID'],
    'answer': pred_dialect
})

df_out_category = pd.DataFrame({
    'subtaskID': 2,
    'datapointID': df_test['datapointID'],
    'answer': pred_category
})

df_output = pd.concat([df_out_dialect, df_out_category], ignore_index=True)
df_output.to_csv('output.csv', index=False)


# ------------- we see the f1_score for our model
from sklearn.metrics import f1_score

train_pred_dialect = clf_dialect.predict(x_train)
train_pred_category = clf_category.predict(x_train)

f1_dialect = f1_score(y_dialect, train_pred_dialect, pos_label=2)   
f1_category = f1_score(y_category, train_pred_category, average='weighted')  

print(f"Dialect F1 (train): {f1_dialect:.4f}")
print(f"Category F1 (train): {f1_category:.4f}")
