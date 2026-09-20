import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from models.logistic_model import train_logistic # <-- "models." added
from models.kmeans_model import run_kmeans
from models.dbscan_model import run_dbscan
from models.lda_model import run_lda

# 1. LOAD DATA
df = pd.read_csv('data/news_dataset.csv') # <-- "data/" added
df['label_num'] = df['label'].map({'Real': 1, 'Fake': 0})

# 2. TEXT TO NUMBERS
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X = vectorizer.fit_transform(df['text'])
y = df['label_num']

# 3. RUN ALL 4 ALGORITHMS
print("Starting all models...")
train_logistic(X, y)
run_kmeans(X)
run_dbscan(X)
run_lda(X, vectorizer)

print("\n✅ All 4 algorithms finished!")