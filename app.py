import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Page Config
st.set_page_config(page_title="Audible Insights", layout="wide")

# --- DATA LAYER ---
@st.cache_data
def get_processed_data():
    # Loading files from raw_data folder
    df1 = pd.read_csv(os.path.join("raw_data", "Audible_Catlog.csv"))
    df2 = pd.read_csv(os.path.join("raw_data", "Audible_Catlog_Advanced_Features.csv"))
    
    # Merging and Cleaning
    df = pd.merge(df1, df2, on=['Book Name', 'Author'], how='inner', suffixes=('', '_drop'))
    df = df.loc[:, ~df.columns.str.contains('_drop')]
    df.drop_duplicates(subset=['Book Name', 'Author'], inplace=True)
    
    # Cleaning columns for EDA
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').replace(-1, np.nan).fillna(df['Rating'].median())
    df['Number of Reviews'] = pd.to_numeric(df['Number of Reviews'], errors='coerce').fillna(0)
    
    # NLP Preprocessing
    df['Description'] = df['Description'].fillna("")
    df['Ranks and Genre'] = df['Ranks and Genre'].fillna("")
    df['metadata'] = df['Book Name'] + " " + df['Author'] + " " + df['Description'] + " " + df['Ranks and Genre']
    
    return df

# --- ML LAYER ---
def build_ml_assets(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=2500)
    tfidf_matrix = tfidf.fit_transform(df['metadata'])
    
    kmeans = KMeans(n_clusters=8, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    return df, tfidf_matrix

# --- UI LAYER ---
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Home & Recommendations", "Data Visualization & EDA"])
    
    try:
        df = get_processed_data()
        df, tfidf_matrix = build_ml_assets(df)
    except Exception as e:
        st.error(f"Please ensure CSV files are in the 'raw_data' folder. Error: {e}")
        return

    if app_mode == "Home & Recommendations":
        st.title("🎧 Audible Insights: Personalized Reading")
        st.write("Find your next favorite book using our NLP-powered clustering engine.")
        
        # Input Preferences
        selected_book = st.selectbox("Select a book you've read:", df['Book Name'].unique())
        num_recs = st.slider("Number of recommendations:", 3, 10, 5)

        if st.button("Generate Recommendations"):
            idx = df[df['Book Name'] == selected_book].index[0]
            sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            related_indices = sim_scores.argsort()[-(num_recs+1):-1][::-1]
            
            recs = df.iloc[related_indices]
            
            st.subheader(f"Top {num_recs} Recommendations for you:")
            for i, (idx, row) in enumerate(recs.iterrows()):
                with st.expander(f"{i+1}. {row['Book Name']} by {row['Author']}"):
                    st.write(f"**Rating:** ⭐ {row['Rating']}")
                    st.write(f"**Description:** {row['Description'][:300]}...")
                    if row['Rating'] > 4.5 and row['Number of Reviews'] < 200:
                        st.success("💎 This is a Hidden Gem!")

    elif app_mode == "Data Visualization & EDA":
        st.title("📊 Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution of Ratings")
            fig, ax = plt.subplots()
            sns.histplot(df['Rating'], bins=20, kde=True, ax=ax, color='skyblue')
            st.pyplot(fig)

        with col2:
            st.subheader("Top 10 Authors by Popularity")
            top_authors = df.groupby('Author')['Number of Reviews'].sum().sort_values(ascending=False).head(10)
            st.bar_chart(top_authors)

        st.subheader("Price vs. Rating Analysis")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x='Price', y='Rating', alpha=0.5, ax=ax2)
        st.pyplot(fig2)

if __name__ == "__main__":
    main()