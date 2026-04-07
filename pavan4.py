import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import time
from datetime import datetime
from io import BytesIO

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
# Use a proven multilingual XLM-RoBERTa model fine-tuned for sentiment analysis
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Premium Dark Theme Colors
COLORS = {
    "primary": "#3b82f6",
    "secondary": "#8b5cf6",
    "accent": "#06b6d4",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "background": "#0a0a0a",
    "surface": "#111827",
    "card": "#1f2937",
    "border": "#374151",
    "text": "#f3f4f6",
    "textSecondary": "#9ca3af",
    "sentiment": {
        "Positive": "#10b981",
        "Neutral": "#f59e0b",
        "Negative": "#ef4444"
    }
}

# ─────────────────────────────────────────────
# MODEL FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the XLM-RoBERTa model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

# ─────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_language(text: str) -> str:
    telugu_pattern = re.compile(r'[\u0C00-\u0C7F]')
    hindi_pattern = re.compile(r'[\u0900-\u097F]')
    tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')
    malayalam_pattern = re.compile(r'[\u0D00-\u0D7F]')
    
    if telugu_pattern.search(text):
        return "Telugu"
    elif hindi_pattern.search(text):
        return "Hindi"
    elif tamil_pattern.search(text):
        return "Tamil"
    elif malayalam_pattern.search(text):
        return "Malayalam"
    elif re.search(r'[a-zA-Z]', text):
        return "English"
    return "Code-Mixed"

def predict_single(text, tokenizer, model, max_length=128):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(DEVICE),
            attention_mask=encoding['attention_mask'].to(DEVICE)
        )
    probs = F.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()
    pred_label = ID2LABEL[int(np.argmax(probs))]
    return pred_label, probs

# ─────────────────────────────────────────────
# YOUTUBE UTILITIES
# ─────────────────────────────────────────────
def extract_video_id(url: str) -> str:
    patterns = [
        r'(?:v=)([a-zA-Z0-9_-]{11})',
        r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return url.strip()

def fetch_youtube_comments(api_key: str, video_id: str, max_comments: int = 200):
    try:
        from googleapiclient.discovery import build
        youtube = build('youtube', 'v3', developerKey=api_key)
        comments = []
        request = youtube.commentThreads().list(
            part='snippet', videoId=video_id,
            maxResults=min(100, max_comments), textFormat='plainText'
        )
        while request and len(comments) < max_comments:
            response = request.execute()
            for item in response.get('items', []):
                c = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'text': c['textDisplay'],
                    'likes': c.get('likeCount', 0),
                    'author': c.get('authorDisplayName', 'Anonymous')
                })
            request = youtube.commentThreads().list_next(request, response)
        return comments[:max_comments]
    except Exception as e:
        st.error(f"YouTube API Error: {e}")
        return []

# ─────────────────────────────────────────────
# PROFESSIONAL VISUALIZATIONS (unchanged)
# ─────────────────────────────────────────────
def create_modern_bar_chart(label_pct):
    labels = list(label_pct.keys())
    values = list(label_pct.values())
    colors = [COLORS['sentiment'][l] for l in labels]
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        textfont=dict(size=12, color=COLORS['text']),
        hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<extra></extra>',
        width=0.6
    )])
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        xaxis=dict(
            title="Sentiment",
            title_font=dict(color=COLORS['textSecondary']),
            tickfont=dict(color=COLORS['textSecondary']),
            showgrid=False
        ),
        yaxis=dict(
            title="Percentage (%)",
            title_font=dict(color=COLORS['textSecondary']),
            tickfont=dict(color=COLORS['textSecondary']),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        showlegend=False,
        margin=dict(t=30, b=30, l=30, r=30)
    )
    return fig

def create_modern_pie_chart(avg_probs):
    labels = ['Negative', 'Neutral', 'Positive']
    values = avg_probs * 100
    colors = [COLORS['sentiment'][l] for l in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        textinfo='label+percent',
        textfont=dict(size=12, color=COLORS['text']),
        hole=0.4,
        hoverinfo='label+percent+value'
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(t=30, b=30, l=30, r=30),
        showlegend=True,
        legend=dict(font=dict(color=COLORS['textSecondary']), orientation="h", yanchor="bottom", y=-0.1)
    )
    return fig

def create_modern_gauge(confidence, label):
    color = COLORS['sentiment'][label]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={'suffix': "%", 'font': {'size': 36, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': COLORS['textSecondary']},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': COLORS['surface'],
            'borderwidth': 1,
            'bordercolor': COLORS['border'],
            'steps': [
                {'range': [0, 33], 'color': 'rgba(239,68,68,0.2)'},
                {'range': [33, 66], 'color': 'rgba(245,158,11,0.2)'},
                {'range': [66, 100], 'color': 'rgba(16,185,129,0.2)'}
            ],
        },
        title={'text': "Confidence Score", 'font': {'size': 14, 'color': COLORS['textSecondary']}}
    ))
    
    fig.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_horizontal_confidence(probs):
    labels = ['Negative', 'Neutral', 'Positive']
    values = probs * 100
    colors = [COLORS['sentiment'][l] for l in labels]
    
    fig = go.Figure(data=[go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        textfont=dict(size=11, color=COLORS['text']),
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=250,
        xaxis=dict(
            title="Confidence (%)",
            title_font=dict(color=COLORS['textSecondary']),
            tickfont=dict(color=COLORS['textSecondary']),
            range=[0, 100],
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title="Sentiment",
            title_font=dict(color=COLORS['textSecondary']),
            tickfont=dict(color=COLORS['textSecondary']),
            showgrid=False
        ),
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    return fig

def create_language_sentiment_chart(language_stats):
    languages = [lang for lang in language_stats.keys() if sum(language_stats[lang].values()) > 0]
    
    fig = go.Figure()
    
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        values = [language_stats[lang][sentiment] for lang in languages]
        color = COLORS['sentiment'][sentiment]
        fig.add_trace(go.Bar(
            name=sentiment,
            x=languages,
            y=values,
            marker_color=color,
            text=values,
            textposition='auto',
            textfont=dict(size=10)
        ))
    
    fig.update_layout(
        title="Sentiment by Language",
        title_font=dict(size=14, color=COLORS['textSecondary']),
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        xaxis=dict(
            title="Language",
            title_font=dict(color=COLORS['textSecondary']),
            tickfont=dict(color=COLORS['textSecondary']),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title="Number of Comments",
            title_font=dict(color=COLORS['textSecondary']),
            tickfont=dict(color=COLORS['textSecondary']),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        legend=dict(font=dict(color=COLORS['textSecondary']), orientation="h", yanchor="bottom", y=1.02)
    )
    return fig

# ─────────────────────────────────────────────
# PRODUCTION-LEVEL CSS (unchanged)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cross-Lingual Sentiment Analysis using XLM-RoBERTa | Production-Grade Sentiment Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
<style>
    /* Global Styles */
    .stApp {{
        background: linear-gradient(135deg, #0a0a0a 0%, #0f172a 100%);
    }}
    
    /* Custom Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Header */
    .header {{
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #374151;
    }}
    
    .header h1 {{
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    
    .header p {{
        color: #9ca3af;
        margin-top: 0.5rem;
    }}
    
    /* Cards */
    .card {{
        background: #1f2937;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #374151;
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    
    .card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
    }}
    
    .card-title {{
        font-size: 1rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
    }}
    
    .card-value {{
        font-size: 2rem;
        font-weight: 700;
        color: #f3f4f6;
    }}
    
    /* Stats Grid */
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}
    
    /* Sentiment Result */
    .sentiment-result {{
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }}
    
    .sentiment-positive {{
        background: linear-gradient(135deg, #10b98120 0%, #05966920 100%);
        border: 1px solid #10b981;
    }}
    
    .sentiment-neutral {{
        background: linear-gradient(135deg, #f59e0b20 0%, #d9770620 100%);
        border: 1px solid #f59e0b;
    }}
    
    .sentiment-negative {{
        background: linear-gradient(135deg, #ef444420 0%, #dc262620 100%);
        border: 1px solid #ef4444;
    }}
    
    .sentiment-emoji {{
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }}
    
    .sentiment-label {{
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s;
        width: 100%;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    }}
    
    /* Input Fields */
    .stTextArea textarea, .stTextInput input {{
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        color: #f3f4f6;
    }}
    
    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: #3b82f6;
        box-shadow: 0 0 0 1px #3b82f6;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background: #1f2937;
        border-radius: 8px;
        padding: 0.25rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px;
        padding: 0.5rem 1rem;
        color: #9ca3af;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: #3b82f6;
        color: white;
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: #3b82f6 !important;
        font-size: 1.8rem !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: #9ca3af !important;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: #111827;
        border-right: 1px solid #1f2937;
    }}
    
    [data-testid="stSidebar"] * {{
        color: #e5e7eb !important;
    }}
    
    /* Divider */
    hr {{
        border-color: #374151;
        margin: 1.5rem 0;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        padding: 2rem;
        color: #6b7280;
        font-size: 0.8rem;
        border-top: 1px solid #1f2937;
        margin-top: 2rem;
    }}
</style>
""", unsafe_allow_html=True)

# Load the XLM-RoBERTa model
with st.spinner("Loading XLM-RoBERTa model from Hugging Face..."):
    try:
        tokenizer, model = load_model()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

# Initialize session state
if 'total_analyses' not in st.session_state:
    st.session_state.total_analyses = 0

# Header
st.markdown("""
<div class="header">
    <h1>🎭 SentimentAI</h1>
    <p>Production-Grade Multilingual Sentiment Analysis | XLM-RoBERTa | 6+ Languages | Real-time Processing</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    max_length = st.slider("Context Window", 64, 256, 128, help="Number of tokens for analysis")
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, help="Minimum confidence for reliable results")
    st.markdown("---")
    st.metric("📊 Total Analyses", st.session_state.total_analyses)
    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.markdown(f"- **Model:** `{MODEL_NAME}`")
    st.markdown("- **Architecture:** XLM-RoBERTa")
    st.markdown("- **Languages:** English + Indic (Hindi, Telugu, Tamil, Malayalam, etc.)")
    st.markdown("- **Inference:** ~50ms")
    st.markdown("---")
    st.caption("© 2024 | Final Year Project | VIT University")

# Tabs
tab1, tab2 = st.tabs(["📝 Text Analysis", "🎬 YouTube Analytics"])

# ─────────────────────────────────────────────
# TAB 1 — TEXT ANALYSIS (unchanged)
# ─────────────────────────────────────────────
with tab1:
    st.markdown("### Real-time Text Analysis")
    st.markdown("Enter text in any supported language for instant sentiment analysis")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_text = st.text_area(
            "",
            placeholder="Example: చాలా బాగుంది! (Telugu) | यह बहुत अच्छा है! (Hindi) | This is amazing! (English)",
            height=120,
            label_visibility="collapsed"
        )
        analyze_btn = st.button("Analyze Sentiment", use_container_width=True)
    
    if analyze_btn and user_text.strip():
        with st.spinner("Processing..."):
            cleaned = clean_text(user_text)
            lang = detect_language(user_text)
            label, probs = predict_single(cleaned, tokenizer, model, max_length)
            st.session_state.total_analyses += 1
            
            if max(probs) < threshold:
                st.warning(f"⚠️ Low confidence ({max(probs)*100:.1f}%) - Results may be unreliable")
            
            # Sentiment Result Card
            sentiment_class = f"sentiment-{label.lower()}"
            emoji = "🌟" if label == "Positive" else "💫" if label == "Neutral" else "⚡"
            color = COLORS['sentiment'][label]
            
            st.markdown(f"""
            <div class="sentiment-result {sentiment_class}">
                <div class="sentiment-emoji">{emoji}</div>
                <div class="sentiment-label" style="color: {color};">{label.upper()}</div>
                <div style="font-size: 1rem;">Confidence: {max(probs)*100:.1f}%</div>
                <div style="color: #9ca3af; margin-top: 0.5rem;">Detected Language: {lang}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔴 Negative", f"{probs[0]*100:.1f}%")
            with col2:
                st.metric("🟡 Neutral", f"{probs[1]*100:.1f}%")
            with col3:
                st.metric("🟢 Positive", f"{probs[2]*100:.1f}%")
            with col4:
                st.metric("🎯 Max Confidence", f"{max(probs)*100:.1f}%")
            
            # Visualizations
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_modern_gauge(max(probs)*100, label), use_container_width=True)
            with col2:
                st.plotly_chart(create_modern_pie_chart(probs), use_container_width=True)
            
            st.plotly_chart(create_horizontal_confidence(probs), use_container_width=True)
            
            with st.expander("🔍 Detailed Analysis"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Text:**")
                    st.code(user_text)
                    st.markdown("**Cleaned Text:**")
                    st.code(cleaned)
                with col2:
                    st.json({
                        "sentiment": label,
                        "confidence": f"{max(probs)*100:.2f}%",
                        "negative": f"{probs[0]*100:.2f}%",
                        "neutral": f"{probs[1]*100:.2f}%",
                        "positive": f"{probs[2]*100:.2f}%",
                        "language": lang,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
    elif analyze_btn:
        st.warning("⚠️ Please enter text to analyze")

# ─────────────────────────────────────────────
# TAB 2 — YOUTUBE ANALYSIS (unchanged)
# ─────────────────────────────────────────────
with tab2:
    st.markdown("### YouTube Comments Analytics")
    st.markdown("Fetch and analyze comments from any YouTube video")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        yt_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
        api_key_inp = st.text_input("YouTube API Key", type="password", placeholder="AIzaSy...")
        max_comments = st.number_input("Max Comments", 100, 5000, 500)
        show_samples = st.slider("Sample Comments", 5, 30, 15)
        fetch_btn = st.button("Fetch & Analyze", use_container_width=True)
    
    if fetch_btn:
        if not yt_url.strip():
            st.warning("⚠️ Please enter a YouTube URL")
        elif not api_key_inp.strip():
            st.warning("⚠️ Please enter your YouTube API key")
        else:
            video_id = extract_video_id(yt_url)
            st.info(f"📹 Video ID: `{video_id}`")
            
            # Progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Fetching comments..."):
                comments = fetch_youtube_comments(api_key_inp, video_id, max_comments)
            
            if not comments:
                st.error("❌ No comments fetched. Check your API key and video ID.")
                st.stop()
            
            st.success(f"✅ Fetched {len(comments)} comments")
            
            # Analyze
            results = []
            all_probs = []
            language_stats = {
                "Telugu": {"Positive": 0, "Neutral": 0, "Negative": 0},
                "Hindi": {"Positive": 0, "Neutral": 0, "Negative": 0},
                "Tamil": {"Positive": 0, "Neutral": 0, "Negative": 0},
                "Malayalam": {"Positive": 0, "Neutral": 0, "Negative": 0},
                "English": {"Positive": 0, "Neutral": 0, "Negative": 0},
                "Code-Mixed": {"Positive": 0, "Neutral": 0, "Negative": 0}
            }
            
            for i, comment in enumerate(comments):
                text_c = clean_text(comment['text'])
                if len(text_c) < 3:
                    continue
                lang = detect_language(text_c)
                lbl, probs = predict_single(text_c, tokenizer, model, max_length)
                results.append({
                    'text': text_c,
                    'label': lbl,
                    'probs': probs,
                    'author': comment['author'],
                    'likes': comment['likes'],
                    'language': lang
                })
                all_probs.append(probs)
                language_stats[lang][lbl] += 1
                
                progress_bar.progress((i + 1) / len(comments))
                status_text.text(f"Analyzing... {i+1}/{len(comments)}")
            
            progress_bar.empty()
            status_text.empty()
            
            if not results:
                st.error("No valid comments to analyze")
                st.stop()
            
            st.session_state.total_analyses += len(results)
            
            # Statistics
            total = len(results)
            label_counts = Counter([r['label'] for r in results])
            label_pct = {k: v/total*100 for k, v in label_counts.items()}
            avg_probs = np.array(all_probs).mean(axis=0)
            dominant = max(label_pct, key=label_pct.get)
            
            # Sentiment Result
            sentiment_class = f"sentiment-{dominant.lower()}"
            emoji = "🌟" if dominant == "Positive" else "💫" if dominant == "Neutral" else "⚡"
            color = COLORS['sentiment'][dominant]
            
            st.markdown(f"""
            <div class="sentiment-result {sentiment_class}">
                <div class="sentiment-emoji">{emoji}</div>
                <div class="sentiment-label" style="color: {color};">Overall: {dominant.upper()}</div>
                <div style="font-size: 1rem;">{label_pct[dominant]:.1f}% of {total} comments</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Stats Grid
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Comments", total)
            with col2:
                st.metric("Negative", f"{label_pct.get('Negative', 0):.1f}%", f"{label_counts.get('Negative', 0)}")
            with col3:
                st.metric("Neutral", f"{label_pct.get('Neutral', 0):.1f}%", f"{label_counts.get('Neutral', 0)}")
            with col4:
                st.metric("Positive", f"{label_pct.get('Positive', 0):.1f}%", f"{label_counts.get('Positive', 0)}")
            
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_modern_bar_chart(label_pct), use_container_width=True)
            with col2:
                st.plotly_chart(create_modern_pie_chart(avg_probs), use_container_width=True)
            
            # Language Analysis
            st.plotly_chart(create_language_sentiment_chart(language_stats), use_container_width=True)
            
            # Language Summary
            st.markdown("### 📊 Language Distribution")
            lang_summary = []
            for lang, stats in language_stats.items():
                total_lang = sum(stats.values())
                if total_lang > 0:
                    lang_summary.append({
                        "Language": lang,
                        "Comments": total_lang,
                        "Percentage": f"{total_lang/total*100:.1f}%",
                        "Positive": stats['Positive'],
                        "Neutral": stats['Neutral'],
                        "Negative": stats['Negative']
                    })
            
            lang_df = pd.DataFrame(lang_summary)
            st.dataframe(lang_df, use_container_width=True)
            
            # Sample Comments
            st.markdown("### 📝 Sample Comments")
            tabs = st.tabs(["🟢 Positive", "🟡 Neutral", "🔴 Negative"])
            
            for idx, lbl in enumerate(["Positive", "Neutral", "Negative"]):
                with tabs[idx]:
                    samples = [r for r in results if r['label'] == lbl][:show_samples]
                    if not samples:
                        st.info(f"No {lbl.lower()} comments found")
                    else:
                        for sample in samples:
                            conf = sample['probs'][np.argmax(sample['probs'])] * 100
                            author = sample['author'][:20] + "..." if len(sample['author']) > 20 else sample['author']
                            
                            with st.expander(f"👤 {author} · 👍 {sample['likes']} likes · Confidence: {conf:.1f}% · {sample['language']}"):
                                st.write(sample['text'])
            
            # Export
            st.markdown("### 📥 Export Results")
            
            df_out = pd.DataFrame([{
                'Author': r['author'],
                'Comment': r['text'],
                'Sentiment': r['label'],
                'Confidence': f"{r['probs'][np.argmax(r['probs'])]*100:.2f}%",
                'Likes': r['likes'],
                'Language': r['language']
            } for r in results])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("📊 CSV", df_out.to_csv(index=False), f"sentiment_{video_id}.csv", "text/csv", use_container_width=True)
            with col2:
                st.download_button("📄 JSON", df_out.to_json(indent=2), f"sentiment_{video_id}.json", "application/json", use_container_width=True)
            with col3:
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df_out.to_excel(writer, index=False, sheet_name='Sentiment Analysis')
                st.download_button("📑 Excel", excel_buffer.getvalue(), f"sentiment_{video_id}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
    <p>🎭 SentimentAI | Production-Grade Multilingual Sentiment Analysis System</p>
    <p>Powered by XLM-RoBERTa | 6+ Languages | Telugu · Hindi · Tamil · Malayalam · English · Code-Mixed</p>
    <p>© 2026 Final Year Project | Department of Computer Science & Engineering | Raghu College</p>
</div>
""", unsafe_allow_html=True)