📰 Fake News Detector - ML Comparison

This project compares 1 Supervised and 3 Unsupervised ML algorithms to detect Fake vs Real News using TF-IDF + Scikit-learn.

Project Structure

fake-news-detector/
│
├── data/
│   └── news_dataset.csv          40 sample news articles with labels
├── models/
│   ├── logistic_model.py         Supervised: Logistic Regression
│   ├── kmeans_model.py           Unsupervised: K-Means
│   ├── dbscan_model.py           Unsupervised: DBSCAN
│   └── lda_model.py              Unsupervised: LDA Topic Modeling
├── http://main.py                      # Runs all 4 models in terminal
├── streamlit_app.py              UI to run all models with 1 click
└── http://requirements.txt             # Dependencies

 Setup & Run

1.  Install dependencies
    bash
    pip install -r requirements.txt
2.  Run in Terminal
    py main.py
3.  Run Streamlit UI
    streamlit run streamlit_app.py
4. Then open: http://localhost:8501 in browser.


#Results & Comparison Report

All models were trained on a dataset of 40 news articles: 21 Real, 19 Fake. TF-IDF was used for text vectorization.
Algorithm	Type	Key Result	Analysis

Logistic Regression	Supervised	*Accuracy: 88%s
Precision: 0.88
Recall: 0.88
F1-Score: 0.87	*Best Performing Model.
Since it had labels "Real/Fake" during training, it learned the patterns very well and gave high accuracy.

K-Means Clustering	Unsupervised	Cluster 0: 31 articles
Cluster 1: 165 articles	*Poor Separation*.
It split data into 2 clusters but they were very imbalanced. It could not clearly separate Fake vs Real without labels.

DBSCAN	Unsupervised	All 196 points = Noise
Clusters found: 0	*Failed on Text Data*.
DBSCAN is density-based and struggles with high-dimensional sparse data like TF-IDF. With default eps=0.5, it marked everything as noise. Needs parameter tuning.

LDA Topic Modeling	Unsupervised	Topic 0: shows, discovered, scientists, government
Topic 1: scientists, hiding, month, government	*Topics are similar*.
LDA found 2 topics but the keywords overlap a lot. This is expected with a small dataset. It shows general news themes, not Fake vs Real.

Conclusion

1.  Supervised Learning wins: Logistic Regression with 81-88% accuracy is the most reliable for Fake News Detection because it learns from labeled data.
2.  Unsupervised Learning struggles: K-Means and DBSCAN could not meaningfully separate Fake vs Real news. Text data is too high-dimensional for them without feature reduction.
3.  LDA is for themes, not classification: LDA is good for finding what topics are in news, not for classifying if it's fake.
4.  Figure 1: Streamlit UI showing 81% accuracy 
5.  Figure 2: Terminal Output showing 88% accuracy.
6.  So Varies from 81% to 88%.

Feature Extraction:

TF-IDF Vectorizer converts news text into 500 numerical features for ML models. 

Tech Stack

| Component | Library | Purpose |

| TF-IDF Vectorizer | scikit-learn | Converts text to 500 numerical features |
| Logistic Regression | scikit-learn | Supervised Model (Real vs Fake) - 81.36% Acc |
| K-Means, DBSCAN, LDA | scikit-learn | Unsupervised Models |
| Data Handling | pandas, numpy | Reading & processing CSV data |
| Web App | streamlit | UI for http://localhost:8501 |


