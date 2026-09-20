from sklearn.decomposition import LatentDirichletAllocation

def run_lda(X, vectorizer):
    print("\n=== UNSUPERVISED: LDA TOPIC MODELING ===")
    lda = LatentDirichletAllocation(n_components=2, random_state=42)
    lda.fit(X)

    print("Top 5 keywords per Topic:")
    for idx, topic in enumerate(lda.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]
        print(f"Topic {idx}: {', '.join(top_words)}")