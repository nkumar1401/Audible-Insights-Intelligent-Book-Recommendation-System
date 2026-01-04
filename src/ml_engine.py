from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def build_models(df):
    """
    Intelligence Layer: Converts metadata to a 2,700-dimensional vector space
    and groups them into semantic neighborhoods.
    """
    # 1. Dynamic K Heuristic (Targeting ~100 books per cluster)
    heuristic_k = max(5, min(20, len(df) // 100)) 
    
    # 2. Vectorization
    # token_pattern=r"(?u)\b\w+\b" allows single-character words (default is 2+)
    # min_df=1 ensures we don't ignore rare words
    tfidf = TfidfVectorizer(
        stop_words='english', 
        max_features=2700,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1 
    )
    
    # Check for empty data before fitting
    if df['metadata'].isnull().all() or (df['metadata'] == "").all():
        raise ValueError("Data pipeline failure: Metadata column is entirely empty.")

    tfidf_matrix = tfidf.fit_transform(df['metadata'])
    
    # 3. Clustering
    kmeans = KMeans(n_clusters=heuristic_k, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    return df, tfidf_matrix, tfidf

def get_recommendations(df, tfidf_matrix, user_input, tfidf_vectorizer, num_recs=5):
    """
    Discovery Engine: Handles both Precision (Title) and Semantic (Mood) search paths.
    """
    # PATH A: Precision Search (Exact Match in Library)
    if user_input in df['Book Name'].values:
        idx = df[df['Book Name'] == user_input].index[0]
        current_cluster = df.iloc[idx]['Cluster']
        
        # Calculate Cosine Similarity
        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Get top matches (excluding the book itself)
        related_indices = sim_scores.argsort()[-(num_recs+1):-1][::-1].tolist()
    
    # PATH B: Semantic Mood Search (Free Text Inference)
    else:
        # Transform the user's mood/query using the fitted vectorizer
        mood_vec = tfidf_vectorizer.transform([user_input])
        sim_scores = cosine_similarity(mood_vec, tfidf_matrix).flatten()
        
        # Get top matches based on semantic proximity
        related_indices = sim_scores.argsort()[-num_recs:][::-1].tolist()
        current_cluster = -1 

    # 4. HEURISTIC: Serendipity Injector (The Anti-Filter Bubble)
    # Replaces the weakest match with a high-confidence outlier from a different cluster
    if len(related_indices) >= 2 and current_cluster != -1:
        others = df[df['Cluster'] != current_cluster].sort_values(by='Confidence_Score', ascending=False)
        if not others.empty:
            related_indices[-1] = others.index[0] 
            
    return df.iloc[related_indices]