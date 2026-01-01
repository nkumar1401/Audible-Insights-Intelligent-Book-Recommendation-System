from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def build_models(df):
    # Dynamic K Heuristic
    heuristic_k = max(5, min(20, len(df) // 100)) 
    tfidf = TfidfVectorizer(stop_words='english', max_features=2700)
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    
    kmeans = KMeans(n_clusters=heuristic_k, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    return df, tfidf_matrix, tfidf
# src/ml_engine.py

def get_recommendations(df, tfidf_matrix, user_input, tfidf_vectorizer, num_recs=5):
    # Check if the input exists exactly as a Book Name
    if user_input in df['Book Name'].values:
        # PATH A: Precision Search (Existing Book)
        idx = df[df['Book Name'] == user_input].index[0]
        current_cluster = df.iloc[idx]['Cluster']
        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        related_indices = sim_scores.argsort()[-(num_recs+1):-1][::-1].tolist()
    else:
        # PATH B: Semantic Mood Search (Free Text)
        # We transform the mood into the same 2,500-feature space
        mood_vec = tfidf_vectorizer.transform([user_input])
        sim_scores = cosine_similarity(mood_vec, tfidf_matrix).flatten()
        related_indices = sim_scores.argsort()[-num_recs:][::-1].tolist()
        current_cluster = -1 # No specific cluster to avoid for mood search

    # HEURISTIC: Serendipity Injector
    if len(related_indices) >= 2 and current_cluster != -1:
        others = df[df['Cluster'] != current_cluster].sort_values(by='Confidence_Score', ascending=False)
        if not others.empty:
            related_indices[-1] = others.index[0] 
            
    return df.iloc[related_indices]