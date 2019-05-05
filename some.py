import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.feature_extraction.text import TfidfVectorizer

with open('data/plots', 'r') as f:
  plots_array = f.read().split('<EOS>')

with open('data/titles', 'r') as f:
    titles_array = f.readlines()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(plots_array)
feature_names = vectorizer.get_feature_names()

with open('vocab', 'w') as f:
  f.write(', '.join(feature_names))

all_vecs = X.toarray()
one_vec = all_vecs[1]

with open('debug', 'w') as f:
  f.write(', '.join(str(v) for v in one_vec))

print('vector len is ', len(feature_names)) 