import streamlit as st
import os
import io
import pandas as pd
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr

# Modular Imports
from src.data_preprocessor import get_processed_data 
from src.ml_engine import build_models, get_recommendations
from src.ai_agent import get_ai_reasoning
from src.utils import get_book_cover  # Ensure this exists in src/utils.py

# --- LUXURY CONFIG ---
st.set_page_config(page_title="Audible Insights | Elite Discovery", layout="wide", page_icon="🎧")

FILE1 = os.path.join("raw_data", "Audible_Catlog.csv")
FILE2 = os.path.join("raw_data", "Audible_Catlog_Advanced_Features.csv")

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .book-container {
        background-color: #1c2128;
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 25px;
    }
    .serendipity-tag { color: #ffd700; font-weight: bold; font-size: 0.8rem; }
    .amazon-link {
        background-color: #232f3e;
        color: #ff9900 !important;
        padding: 8px 15px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🏛️ Audible Insights: The Intelligent Discovery Suite")
    st.markdown("---")

    # 1. ORCHESTRATION: Load Data & ML Assets
    try:
        df = get_processed_data(FILE1, FILE2) 
        df, tfidf_matrix, tfidf_vectorizer = build_models(df)
    except Exception as e:
        st.error(f"System Offline: {e}")
        return

    # --- SIDEBAR ---
    st.sidebar.title("🌍 Global Navigation")
    app_mode = st.sidebar.radio("Navigation", ["Home & Discovery", "Intelligence Dashboard"])
    st.sidebar.divider()
    st.sidebar.metric("Curated Titles", len(df))

    if app_mode == "Home & Discovery":
        # Voice Recognition Section
        col_v1, col_v2 = st.columns([1, 4])
        voice_query = ""
        with col_v1:
            audio = mic_recorder(start_prompt="🎙️ Voice Search", key='recorder')
        
        if audio:
            r = sr.Recognizer()
            try:
                with sr.AudioFile(io.BytesIO(audio['bytes'])) as source:
                    voice_query = r.recognize_google(r.record(source))
                    st.success(f"Recognized: {voice_query}")
            except: st.error("Audio not clear")

        # Discovery Input
        search_type = st.radio("Search by:", ["Library History", "Describe Your Mood"], horizontal=True)
        if search_type == "Library History":
            user_input = st.selectbox("Which book shaped your thinking?", [""] + list(df['Book Name'].unique()))
        else:
            user_input = st.text_input("Describe the knowledge you seek:", value=voice_query)

        num_recs = st.slider("Depth of Journey", 3, 10, 5)

        if st.button("✨ Initialize Discovery Path"):
            if not user_input:
                st.warning("Please provide an input.")
                return

            # Fetch Recommendations
            results = get_recommendations(df, tfidf_matrix, user_input, tfidf_vectorizer, num_recs)
            
            # Find cluster for serendipity tagging
            current_cluster = df[df['Book Name'] == user_input]['Cluster'].values[0] if user_input in df['Book Name'].values else -1

            st.subheader("Your Personalized Path to Wisdom")
            
            # Display Results
            for i, (idx, row) in enumerate(results.iterrows()):
                is_wildcard = (i == len(results) - 1 and current_cluster != -1)
                
                with st.spinner(f"Fetching Cover for {row['Book Name']}..."):
                    cover_url = get_book_cover(row['Book Name'])

                # Luxury Book Card
                st.markdown(f"""
                <div class="book-container">
                    <div class="serendipity-tag">{'🌟 Serendipity Injector' if is_wildcard else '🎯 Precision Match'}</div>
                    <div style="display: flex; gap: 30px; margin-top: 15px;">
                        <div style="flex: 1;">
                            <img src="{cover_url}" style="border-radius: 10px; width: 100%; box-shadow: 0px 4px 15px rgba(0,0,0,0.5);">
                        </div>
                        <div style="flex: 4;">
                            <h2 style="margin:0;">{row['Book Name']}</h2>
                            <p style="color: #8b949e;">By {row['Author']} | Rating: ⭐ {row['Rating']}</p>
                            <p style="line-height: 1.6;">{row['Description'][:450]}...</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # AI Agent & Business Links
                with st.expander("🧠 View AI Reasoning & Acquisition"):
                    with st.spinner("Analyzing..."):
                        reason = get_ai_reasoning(user_input, row['Book Name'], row['Description'])
                        st.info(reason)
                    
                    q = row['Book Name'].replace(" ", "+")
                    st.markdown(f"""
                        <a href="https://www.amazon.in/s?k={q}" class="amazon-link">🛒 Buy on Amazon</a>
                        <a href="https://www.audible.in/search?keywords={q}" style="margin-left:20px; color:#58a6ff; text-decoration:none;">🎧 Listen on Audible</a>
                    """, unsafe_allow_html=True)

    elif app_mode == "Intelligence Dashboard":
        st.title("📊 Data Observability")
        st.bar_chart(df['Cluster'].value_counts())

if __name__ == "__main__":
    main()