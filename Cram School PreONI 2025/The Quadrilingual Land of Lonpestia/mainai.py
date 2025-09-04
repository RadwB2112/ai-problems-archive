import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


df = pd.read_csv("test_data.csv")

sub1 = df.dropna(subset=['textB']).reset_index(drop=True)
sub2 = df[df['textB'].isna()].reset_index(drop=True)

# ------------------------------------------------ subtask1 ---------------------------------------

# vectorizare TF-IDF pe char‑ngrams
vec1 = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=2)
all_pairs = pd.concat([sub1['textA'], sub1['textB']])
X1 = vec1.fit_transform(all_pairs)

XA = X1[:len(sub1)]
XB = X1[len(sub1):]

sims = cosine_similarity(XA, XB).diagonal()
pred1 = sims > 0.25 # threshold

# dataframe pt task 1
out1 = sub1[['datapointID']].copy()
out1['subtaskID'] = 1
out1['answer'] = ['True' if v else 'False' for v in pred1]


# SUBTASK 2: KMEANS (k=4) ------------------------------------------------------------------------

# vectorizare TF-IDF pe char‑ngrams
vec2 = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), min_df=3)
X2 = vec2.fit_transform(sub2['textA'])

# ruleaza KMeans cu 4 clustere
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
clusters = kmeans.fit_predict(X2)
sub2['cluster'] = clusters


# 3.1. top‐urile n‑gramurilor din fiecare cluster |  ca sa decizi maparea cluster->limba
for cl in range(4):
    print(f"\nCluster {cl}:")
    print(sub2[sub2['cluster'] == cl]['textA'].head(3).to_string(index=False))


cluster_to_lang = {
    0: 'Hungeleabeen',
    1: 'Englcrevbeh',
    2: 'En Gli',
    3: 'Hure'
}
sub2['answer'] = sub2['cluster'].map(cluster_to_lang)

# subtask 2 - output
out2 = sub2[['datapointID']].copy()
out2['subtaskID'] = 2
out2['answer']   = sub2['answer']


# outfile
submission = pd.concat([out1, out2], ignore_index=True)
submission.to_csv("submission.csv", index=False)