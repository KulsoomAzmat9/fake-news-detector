import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 Fake News Detector")
st.write("Test 1 Supervised + 3 Unsupervised ML algorithms on your news dataset")

@st.cache_data
def load_data():
    df = pd.read_csv("data/news_dataset.csv")
    return df

try:
    df = load_data()
    st.success(f"✅ Dataset loaded: {len(df)} articles | Real: {sum(df['label']=='Real')} | Fake: {sum(df['label']=='Fake')}")
except FileNotFoundError:
    st.error("❌ Put news_dataset.csv inside a data/ folder")
    st.stop()

if st.button("🚀 Run All 4 Models", type="primary"):
    
    with st.spinner("Running all models... Please wait 10-20 seconds"):
        
        # Text to Numbers
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        X = vectorizer.fit_transform(df['text'])
        y = df['label']
        feature_names = vectorizer.get_feature_names_out()

        st.divider()

        # 1. LOGISTIC REGRESSION
        st.header("1. SUPERVISED: Logistic Regression")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.divider()

        # 2. K-MEANS
        st.header("2. UNSUPERVISED: K-Means Clustering")
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X.toarray())
        cluster_dist = pd.Series(clusters).value_counts().to_dict()
        st.write(f"*Cluster Distribution:* {cluster_dist}")
        st.info("Note: K-Means split data into 2 groups but doesn't know 'Real vs Fake' labels")

        st.divider()

        # 3. DBSCAN
        st.header("3. UNSUPERVISED: DBSCAN")
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        clusters_db = dbscan.fit_predict(X.toarray())
        cluster_dist_db = pd.Series(clusters_db).value_counts().to_dict()
        n_noise = list(clusters_db).count(-1)
        st.write(f"*Cluster Distribution:* {cluster_dist_db}")
        st.write(f"*Noise points found:* {n_noise}")
        st.warning("DBSCAN marked most points as noise because text data is high-dimensional")

        st.divider()

        # 4. LDA
        st.header("4. UNSUPERVISED: LDA Topic Modeling")
        lda = LatentDirichletAllocation(n_components=2, random_state=42)
        lda.fit(X)
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:]
            top_words = [feature_names[i] for i in top_words_idx]
            st.write(f"*Topic {topic_idx}:* {', '.join(top_words)}")
        
        st.divider()
        st.success("✅ All 4 algorithms finished!")

else:
    st.info("👆 Click the button above to start. Make sure your dataset is in data/news_dataset.csv")