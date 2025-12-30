import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from dotenv import load_dotenv
from groq import Groq

# 1. Load the .env file
load_dotenv()

st.set_page_config(page_title="Audible Insights", layout="wide", page_icon="🎧")

@st.cache_data
def get_processed_data():
    file_path = os.path.join("raw_data", "Audible_Catlog_Advanced_Features.csv")
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').replace(-1, np.nan).fillna(df['Rating'].median())
    df['Number of Reviews'] = pd.to_numeric(df['Number of Reviews'], errors='coerce').fillna(0)
    df['Description'] = df['Description'].fillna("No description available")
    df['Ranks and Genre'] = df['Ranks and Genre'].fillna("General")
    df['metadata'] = df['Book Name'] + " " + df['Author'] + " " + df['Description'] + " " + df['Ranks and Genre']
    return df

def build_ml_assets(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=2500)
    tfidf_matrix = tfidf.fit_transform(df['metadata'])
    kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(tfidf_matrix)
    return df, tfidf_matrix

def get_ai_reasoning(user_query, book_title, description):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "AI Reasoning requires an API key."
        client = Groq(api_key=api_key)
        prompt = f"User Interest: {user_query}\nBook: {book_title}\nExplain in 2 sentences why this matches."
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=100
        )
        return completion.choices[0].message.content
    except:
        return "Matched to your profile for growth."

def main():
    st.sidebar.title("🌍 Global Navigation")
    app_mode = st.sidebar.radio("Go to", ["Home & Recommendations", "Data Visualization & EDA"])
    
    df = get_processed_data()
    if df.empty: return
    df, tfidf_matrix = build_ml_assets(df)

    # Indented Sidebar Stats
    st.sidebar.markdown("---")
    st.sidebar.write(f"📊 Total Books: {len(df)}")
    path = 'raw_data/Audible_Catlog_Advanced_Features.csv'
    if os.path.exists(path):
        st.sidebar.write(f"🕒 Last Sync: {pd.to_datetime(os.path.getmtime(path), unit='s').strftime('%Y-%m-%d %H:%M')}")

    if app_mode == "Home & Recommendations":
        st.title("🚀 Audible Insights")
        audio = mic_recorder(start_prompt="Speak now", key='recorder')
        voice_query = ""
        if audio:
            r = sr.Recognizer()
            try:
                with sr.AudioFile(io.BytesIO(audio['bytes'])) as source:
                    voice_query = r.recognize_google(r.record(source))
            except: pass

        search_type = st.radio("Search Method:", ["Select from Library", "Describe Mood"], horizontal=True)
        if search_type == "Select from Library":
            user_input = st.selectbox("Pick a book:", [""] + list(df['Book Name'].unique()))
        else:
            user_input = st.text_input("Enter mood:", value=voice_query)
            
        num_recs = st.slider("Suggestions:", 3, 10, 5)

        if st.button("Generate My Reading Journey"):
            if user_input in df['Book Name'].values:
                idx = df[df['Book Name'] == user_input].index[0]
                current_cluster = df.iloc[idx]['Cluster']
                sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
                related_indices = sim_scores.argsort()[-(num_recs+1):-1][::-1].tolist()
            else:
                recs = df[df['metadata'].str.contains(user_input, case=False, na=False)].head(num_recs)
                related_indices = recs.index.tolist()
                current_cluster = -1

            if len(related_indices) >= 2:
                other = df[df['Cluster'] != current_cluster]
                if not other.empty: related_indices[-1] = other.sample(1).index[0]
            
            results = df.loc[related_indices]
            for i, (idx, row) in enumerate(results.iterrows()):
                with st.expander(f"{row['Book Name']}"):
                    with st.spinner("AI Reasoning..."):
                        st.markdown(f"**🧠 AI Insight:** {get_ai_reasoning(user_input, row['Book Name'], row['Description'])}")
                    st.write(f"**Author:** {row['Author']} | **Rating:** ⭐ {row['Rating']}")
                    st.write(f"**Description:** {row['Description'][:300]}...")

    elif app_mode == "Data Visualization & EDA":
        st.title("📊 Analytics")
        st.bar_chart(df['Rating'].value_counts())

if __name__ == "__main__":
    main()