import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# --- PROJECT CONFIGURATION ---
st.set_page_config(page_title="Audible Insights", layout="wide", page_icon="🎧")

# File paths based on your local directory structure
FILE1 = os.path.join("raw_data", "Audible_Catlog.csv")
FILE2 = os.path.join("raw_data", "Audible_Catlog_Advanced_Features.csv")

# --- DATA PROCESSING PIPELINE ---

@st.cache_data
def load_and_preprocess_data():
    # 1. Load Datasets
    df1 = pd.read_csv(FILE1)
    df2 = pd.read_csv(FILE2)
    
    # 2. Merge on Book Name and Author [cite: 165, 191]
    df = pd.merge(df1, df2, on=['Book Name', 'Author'], how='inner', suffixes=('', '_drop'))
    df = df.loc[:, ~df.columns.str.contains('_drop')] # Remove duplicate columns from merge
    
    # 3. Data Cleaning [cite: 167, 169]
    df.drop_duplicates(subset=['Book Name', 'Author'], inplace=True)
    
    # Handle ratings: replace -1 with median and convert to float
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').replace(-1, np.nan)
    df['Rating'] = df['Rating'].fillna(df['Rating'].median())
    
    # Handle reviews and prices
    df['Number of Reviews'] = pd.to_numeric(df['Number of Reviews'], errors='coerce').fillna(0)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    
    # 4. Feature Engineering (NLP) [cite: 174]
    df['Description'] = df['Description'].fillna("No description available")
    df['Ranks and Genre'] = df['Ranks and Genre'].fillna("General")
    # Combine text features for a rich vector space
    df['tags'] = df['Book Name'] + " " + df['Author'] + " " + df['Description'] + " " + df['Ranks and Genre']
    
    return df

# --- MODELING ---

def build_models(df):
    # Vectorization using TF-IDF [cite: 174, 195]
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    
    # Clustering using K-Means 
    kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    return df, tfidf_matrix

# --- UI COMPONENTS ---

def main():
    st.title("🚀 Audible Insights: Intelligent Book Recommendations")
    st.markdown("---")
    
    try:
        data = load_and_preprocess_data()
        df, tfidf_matrix = build_models(data)
    except Exception as e:
        st.error(f"Error loading data from 'raw_data/': {e}")
        return

    # Sidebar Navigation
    menu = ["Recommendation Engine", "Exploratory Data Analysis (EDA)", "Genre Search"]
    choice = st.sidebar.selectbox("Choose Action", menu)

    if choice == "Recommendation Engine":
        st.subheader("Personalized Book Suggestions [cite: 154]")
        book_list = df['Book Name'].unique()
        selected_book = st.selectbox("Select a book you enjoyed:", book_list)

        if st.button("Get Recommendations"):
            # Content-Based Filtering [cite: 178]
            idx = df[df['Book Name'] == selected_book].index[0]
            similarity = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            
            # Get top 5 indices (excluding the book itself)
            sim_indices = similarity.argsort()[-6:-1][::-1]
            recs = df.iloc[sim_indices]
            
            cols = st.columns(5)
            for i, (index, row) in enumerate(recs.iterrows()):
                with cols[i]:
                    st.info(f"**{row['Book Name']}**")
                    st.write(f"Author: {row['Author']}")
                    st.write(f"⭐ {row['Rating']}")
                    # Scenario: Hidden Gem check 
                    if row['Rating'] >= 4.5 and row['Number of Reviews'] < 100:
                        st.caption("💎 Hidden Gem!")

    elif choice == "Exploratory Data Analysis (EDA)":
        st.subheader("Data Insights & Trends [cite: 171, 221]")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Top 10 Highly Rated Authors")
            top_authors = df.groupby('Author')['Rating'].mean().sort_values(ascending=False).head(10)
            st.bar_chart(top_authors)
            
        with col2:
            st.write("Rating Distribution [cite: 223]")
            st.bar_chart(df['Rating'].value_counts())

    elif choice == "Genre Search":
        st.subheader("Discover by Genre [cite: 155]")
        # Simplified genre search 
        genre_query = st.text_input("Enter a genre (e.g., 'Science Fiction', 'Self-Help'):")
        if genre_query:
            genre_results = df[df['Ranks and Genre'].str.contains(genre_query, case=False, na=False)]
            st.write(f"Top {genre_query} Books:")
            st.table(genre_results[['Book Name', 'Author', 'Rating', 'Price']].sort_values(by='Rating', ascending=False).head(10))

if __name__ == "__main__":
    main()