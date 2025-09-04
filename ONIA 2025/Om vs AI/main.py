import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# === citire date

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

submission = []

# === Subtask 1
X_train = train_df["text"]
y_train = train_df["label"]

train_pool = Pool(
    X_train, 
    y_train,
    text_features=[0]
)
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    eval_metric="F1",
    verbose=100
)
model.fit(train_pool)

test_sub1 = test_df[test_df["subtaskID"] == 1].copy()
test_pool = Pool(test_sub1["text"], text_features=[0])

preds_sub1 = model.predict(test_pool)
test_sub1["answer"] = preds_sub1

# Adauga in submisie submisie
submission.append(test_sub1[["subtaskID", "ID", "answer"]].rename(columns={"ID": "datapointID"}))

# === Subtask 2
test_sub2 = test_df[test_df["subtaskID"] == 2].copy()

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english'
)
X_tfidf = vectorizer.fit_transform(test_sub2["text"])

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X_tfidf)
test_sub2["cluster"] = cluster_labels

# We manual map each cluster
cluster_to_label = {}
for c in range(4):
    print(f"\nCluster {c}:")
    sample_texts = test_sub2[test_sub2["cluster"] == c]["text"].head(5).tolist()
    for t in sample_texts:
        print("-", t)
    
    label = input("Cluster: ")
    while label not in ["SCIENCE", "BUSINESS", "CRIME", "RELIGION"]:
        label = input("Pick from (SCIENCE/BUSINESS/CRIME/RELIGION): ").strip()
    cluster_to_label[c] = label

test_sub2["answer"] = test_sub2["cluster"].map(cluster_to_label)

# Adaugă în submisie
submission.append(test_sub2[["subtaskID", "ID", "answer"]].rename(columns={"ID": "datapointID"}))

# === Salveaza submisia
final_submission = pd.concat(submission, ignore_index=True)
final_submission.to_csv("submission.csv", index=False)