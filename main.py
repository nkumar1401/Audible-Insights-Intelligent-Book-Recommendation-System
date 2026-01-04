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

FILE1 = os.path.join("raw_data", "Audible_Catlog.csv")
FILE2 = os.path.join("raw_data", "Audible_Catlog_Advanced_Features.csv")

# --- PREMIUM WINTER-THEMED CSS WITH ANIMATION ---
st.markdown("""
    <style>
    /* 1. Global Background & Snow Animation */
    .stApp {
        background: radial-gradient(circle at top, #1a1c2c 0%, #0d0e14 100%) !important;
        color: #f0f2f6 !important;
        font-family: 'Georgia', serif;
    }

    /* Falling Snow Effect */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: radial-gradient(circle, #ffffff 1px, transparent 1px);
        background-size: 50px 50px;
        animation: snow 10s linear infinite;
        opacity: 0.1;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes snow {
        0% { background-position: 0 0; }
        100% { background-position: 500px 1000px; }
    }

    /* 2. Glassmorphism Book Container */
    .book-container {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px);
        padding: 40px;
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.6);
        margin-bottom: 40px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        z-index: 1;
    }

    .book-container:hover {
        transform: translateY(-10px) scale(1.01);
        border: 1px solid rgba(212, 175, 55, 0.5); /* Nordic Gold Glow */
        background: rgba(255, 255, 255, 0.08) !important;
    }

    /* 3. Luxury Typography */
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        font-weight: 700;
    }

    /* 4. Nordic Gold Serendipity Tag */
    .serendipity-tag {
        color: #d4af37 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 800;
        font-size: 0.7rem;
        background: rgba(212, 175, 55, 0.15);
        padding: 6px 15px;
        border-radius: 50px;
        border: 1px solid rgba(212, 175, 55, 0.4);
        display: inline-block;
        margin-bottom: 10px;
    }

    /* 5. Premium CTA - Amazon Button */
    .amazon-link {
        background: linear-gradient(135deg, #d4af37 0%, #aa8919 100%) !important;
        color: #000000 !important;
        padding: 14px 30px;
        border-radius: 12px;
        text-decoration: none !important;
        font-weight: 800;
        display: inline-block;
        box-shadow: 0 10px 20px rgba(212, 175, 55, 0.2);
        transition: all 0.3s ease;
        text-transform: uppercase;
        font-size: 0.9rem;
    }

    .amazon-link:hover {
        box-shadow: 0 15px 30px rgba(212, 175, 55, 0.4);
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
# --- PRE-INITIALIZE THEME ENGINE (DO NOT MOVE) ---
    st.markdown("""
    <style>
    /* 1. FORCE THEME ON ALL LAYERS */
    .stApp, .main, .block-container {
        background: radial-gradient(circle at top, #1a1c2c 0%, #0d0e14 100%) !important;
    }

    /* 2. THE WINTER SNOW OVERLAY (Improved) */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(circle, #ffffff 1px, transparent 1px),
            radial-gradient(circle, #ffffff 1.5px, transparent 1px);
        background-size: 50px 50px, 100px 100px;
        animation: snow 15s linear infinite;
        opacity: 0.2;
        pointer-events: none;
        z-index: 1;
    }

    @keyframes snow {
        0% { background-position: 0 0, 0 0; }
        100% { background-position: 500px 1000px, 400px 400px; }
    }

    /* 3. ENSURE SIDEBAR MATCHES THE WINTER VIBE */
    [data-testid="stSidebar"] {
        background-color: #1a1c2c !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

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