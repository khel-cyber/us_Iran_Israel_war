"""
Sentiment Analysis with Topic Modeling — US / Israel–Iran War


"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title=" US / Israel–Iran War ",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── Exact CSS theme from app2.py ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
  --bg-dark:#0d1117; --bg-card:#161b22; --bg-panel:#1c2230;
  --accent1:#e8a838; --accent2:#3b82f6; --accent3:#ef4444; --accent4:#22c55e;
  --text-main:#e6edf3; --text-muted:#8b949e; --border:#30363d;
}
.stApp { background:var(--bg-dark); color:var(--text-main); }
.stApp * { font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:var(--bg-card) !important; border-right:1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color:var(--text-main) !important; }
[data-testid="metric-container"] { background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:16px !important; }
[data-testid="stMetricLabel"] { color:var(--text-muted) !important; font-size:0.8rem !important; }
[data-testid="stMetricValue"] { color:var(--text-main) !important; font-size:1.6rem !important; font-weight:600 !important; }
.section-header { font-family:'Playfair Display',serif; font-size:2rem; font-weight:700; color:var(--text-main); border-left:4px solid var(--accent1); padding-left:16px; margin-bottom:8px; }
.section-sub { color:var(--text-muted); font-size:0.95rem; margin-bottom:24px; padding-left:20px; }
.insight-card { background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:20px 24px; margin-bottom:16px; }
.insight-card h4 { color:var(--accent1); font-size:0.85rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px; }
.insight-card p { color:var(--text-main); font-size:0.95rem; line-height:1.6; margin:0; }
.callout { background:rgba(59,130,246,0.08); border-left:3px solid var(--accent2); border-radius:0 8px 8px 0; padding:14px 18px; margin:16px 0; color:var(--text-main); font-size:0.9rem; line-height:1.6; }
.callout-warn { background:rgba(232,168,56,0.08); border-left-color:var(--accent1); }
.callout-success { background:rgba(34,197,94,0.08); border-left-color:var(--accent4); }
.hero-banner { background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#1a1f2e 100%); border:1px solid var(--border); border-radius:16px; padding:40px 48px; margin-bottom:32px; position:relative; overflow:hidden; }
.hero-banner::before { content:''; position:absolute; top:0; right:0; width:300px; height:300px; background:radial-gradient(circle,rgba(232,168,56,0.08) 0%,transparent 70%); border-radius:50%; }
.hero-title { font-family:'Playfair Display',serif; font-size:2.8rem; font-weight:900; color:var(--text-main); line-height:1.15; margin-bottom:12px; }
.hero-title span { color:var(--accent1); }
.hero-meta { color:var(--text-muted); font-size:0.9rem; margin-top:16px; }
.hero-meta strong { color:var(--accent2); }
.phase-card { background:var(--bg-panel); border:1px solid var(--border); border-radius:10px; padding:14px 18px; margin-bottom:10px; display:flex; align-items:flex-start; gap:14px; }
.phase-num { background:var(--accent1); color:#000; font-weight:700; font-size:0.85rem; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.phase-title { color:var(--text-main); font-weight:600; font-size:0.95rem; }
.phase-desc { color:var(--text-muted); font-size:0.85rem; margin-top:2px; }
hr { border-color:var(--border) !important; margin:24px 0 !important; }
.stTabs [data-baseweb="tab-list"] { gap:6px; background:transparent; }
.stTabs [data-baseweb="tab"] { background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:8px !important; color:var(--text-muted) !important; padding:8px 16px !important; }
.stTabs [aria-selected="true"] { background:var(--accent1) !important; color:#000 !important; border-color:var(--accent1) !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)


# ── NLTK setup ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Setting up NLP libraries…")
def load_nltk():
    import nltk
    for pkg in ['stopwords','punkt','punkt_tab','wordnet','omw-1.4',
                'averaged_perceptron_tagger_eng','vader_lexicon']:
        nltk.download(pkg, quiet=True)
    return True

load_nltk()


# ── Session state ─────────────────────────────────────────────────────────────
for k in ['df','lda_model','count_matrix','vocab_lda','document_topic_matrix',
          'word_freq','angle_results','avg_by_source','avg_by_platform',
          'dist_pct','heatmap_df','angle_score_df','topic_keywords_dict',
          'TOPIC_LABELS','ANGLE_KEYWORDS']:
    if k not in st.session_state:
        st.session_state[k] = None


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE  — verbatim notebook code, only read source changed
# ═════════════════════════════════════════════════════════════════════════════
def run_pipeline(uploaded_file):

    # ── Cell 6 — Imports ─────────────────────────────────────────────────
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mtick
    import seaborn as sns
    import re, string, os
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from textblob import TextBlob
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from wordcloud import WordCloud
    from collections import Counter
    os.makedirs("outputs", exist_ok=True)

    bar = st.progress(0, text="Loading dataset…")

    # ── Cell 8 — Load dataset (only change: uploader instead of file path) ──
    df = pd.read_csv(uploaded_file)
    bar.progress(5, text=f"✅ Loaded {len(df):,} records")

    # ── Cell 15 — Text length ─────────────────────────────────────────────
    df['text_length'] = df['text'].str.len()

    # ── Cell 18 — Standardize source names ───────────────────────────────
    df['source'] = df['source'].replace({'RT': 'RT News'})

    # ── Cell 20 — Remove very short texts ────────────────────────────────
    bar.progress(8, text="Filtering short/empty records…")
    before = len(df)
    df = df[df['text'].str.len() >= 30].copy()
    df = df[df['text'].str.strip() != ''].copy()
    df = df.reset_index(drop=True)

    # ── Cell 22 — Stop words ─────────────────────────────────────────────
    bar.progress(12, text="Configuring stop words…")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    CUSTOM_STOPWORDS = {
        'said', 'say', 'says', 'told', 'according', 'also', 'would',
        'could', 'one', 'two', 'three', 'year', 'week', 'day', 'time',
        'us', 'u', 's', 'amp', 'rt', 'via', 'http', 'https',
        'bbc', 'reuters', 'aljazeera', 'rtnews', 'news'
    }
    stop_words.update(CUSTOM_STOPWORDS)

    # ── Cell 24 — Clean text ──────────────────────────────────────────────
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return 'a'
        elif treebank_tag.startswith('V'):
            return 'v'
        elif treebank_tag.startswith('N'):
            return 'n'
        elif treebank_tag.startswith('R'):
            return 'r'
        else:
            return 'n'

    def clean_text(text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ''
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        pos_tagged = pos_tag(tokens)
        lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                      for word, tag in pos_tagged]
        return ' '.join(lemmatized)

    # ── Cell 26 — Apply cleaning ──────────────────────────────────────────
    bar.progress(20, text="Cleaning & lemmatizing text… (takes 1–2 min for large datasets)")
    df['clean_text'] = df['text'].apply(clean_text)
    before = len(df)
    df = df[df['clean_text'].str.len() > 20].copy()
    df = df.reset_index(drop=True)

    # ── Cell 28 — Word frequency ──────────────────────────────────────────
    bar.progress(52, text="Computing word frequencies…")
    all_tokens = ' '.join(df['clean_text']).split()
    word_freq = Counter(all_tokens)

    # ── Cell 32 — CountVectorizer ─────────────────────────────────────────
    bar.progress(58, text="Vectorizing corpus…")
    corpus = df['clean_text'].dropna().tolist()
    count_vectorizer = CountVectorizer(min_df=5, max_df=0.9)
    count_matrix = count_vectorizer.fit_transform(corpus)
    vocab_lda = count_vectorizer.get_feature_names_out()

    # ── Cell 34 — Train LDA ───────────────────────────────────────────────
    bar.progress(65, text="Training LDA model (7 topics)…")
    NUM_TOPICS = 7
    lda_model = LatentDirichletAllocation(
        n_components=NUM_TOPICS,
        random_state=122,
        max_iter=100
    )
    lda_model.fit(count_matrix)

    # ── Cell 38 — Topic labels ────────────────────────────────────────────
    TOPIC_LABELS = {
        0: 'Military Operations',
        1: 'Civilian & Humanitarian',
        2: 'Diplomatic Negotiations',
        3: 'Geopolitical Tensions',
        4: 'Economic Impact',
        5: 'Regional Responses',
        6: 'Media Narratives',
    }
    NUM_TOP_WORDS = 10
    topic_keywords_dict = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[:-NUM_TOP_WORDS - 1:-1]
        top_words = [vocab_lda[i] for i in top_indices]
        topic_keywords_dict[TOPIC_LABELS[topic_idx]] = top_words

    # ── Cell 40 — Dominant topic ──────────────────────────────────────────
    bar.progress(75, text="Assigning dominant topics…")
    document_topic_matrix = lda_model.transform(count_matrix)
    dominant_topic_indices = document_topic_matrix.argmax(axis=1)
    df['dominant_topic'] = [TOPIC_LABELS.get(i, f'Topic {i+1}') for i in dominant_topic_indices]

    # ── Cell 50 — VADER (sentiment_score / sentiment_label) ───────────────
    bar.progress(78, text="Running VADER sentiment analysis…")
    sia = SentimentIntensityAnalyzer()

    def get_vader_sentiment(text):
        if not isinstance(text, str):
            return 0.0, 'neutral'
        score = sia.polarity_scores(text)['compound']
        if score >= 0.05:
            label = 'positive'
        elif score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        return round(score, 4), label

    df[['sentiment_score', 'sentiment_label']] = df['text'].apply(
        lambda x: pd.Series(get_vader_sentiment(x))
    )

    # ── Cell 51 — vader_score / vader_class (Capitalised) ─────────────────
    bar.progress(82, text="Computing VADER classes…")
    sia2 = SentimentIntensityAnalyzer()

    def get_vader_score(text):
        return sia2.polarity_scores(str(text))['compound']

    def classify_vader_sentiment(score):
        if score >= 0.05:  return 'Positive'
        elif score <= -0.05: return 'Negative'
        else: return 'Neutral'

    df['vader_score'] = df['text'].apply(get_vader_score)
    df['vader_class'] = df['vader_score'].apply(classify_vader_sentiment)

    # ── Cell 52 — TextBlob ────────────────────────────────────────────────
    bar.progress(86, text="Running TextBlob analysis…")

    def get_sentiment_metrics(text):
        blob = TextBlob(str(text))
        polarity     = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        objectivity  = 1 - subjectivity
        sentiment = 'Positive' if polarity > 0.05 else ('Negative' if polarity < -0.05 else 'Neutral')
        return pd.Series([polarity, subjectivity, objectivity, sentiment])

    df[['tb_polarity', 'tb_subjectivity', 'tb_objectivity', 'tb_class']] = \
        df['text'].apply(get_sentiment_metrics)

    # ── Cell 54 — Angle keywords ──────────────────────────────────────────
    ANGLE_KEYWORDS = {
        'i. Military Operations/Strategy': [
            'missile', 'drone', 'airstrike', 'strike', 'attack', 'military',
            'bomb', 'weapon', 'soldier', 'troops', 'navy', 'rocket', 'fighter',
            'warplane', 'artillery', 'radar', 'nuclear', 'defence', 'defense'
        ],
        'ii. Geopolitical Tensions': [
            'tension', 'alliance', 'sanction', 'diplomatic', 'nato', 'russia',
            'china', 'africa', 'asia', 'hezbollah', 'houthi', 'proxy', 'region',
            'gulf', 'geopolit', 'superpower', 'influence', 'coalition'
        ],
        'iii. Economic Impact': [
            'oil', 'gas', 'price', 'market', 'energy', 'hormuz', 'strait',
            'supply', 'inflation', 'trade', 'sanction', 'economic', 'recession',
            'fuel', 'barrel', 'opec', 'economy', 'financial', 'cost'
        ],
        'iv. Media Narratives & Propaganda': [
            'propaganda', 'fake', 'bias', 'misinformation', 'disinformation',
            'media', 'narrative', 'misleading', 'claim', 'censorship',
            'manipulation', 'distort', 'report', 'coverage', 'framing'
        ],
        'v. Support for the War': [
            'support', 'oppose', 'protest', 'ally', 'condemn', 'backing',
            'anti-war', 'pro-war', 'public opinion', 'demonstration',
            'rally', 'solidarity', 'resistance', 'stand with', 'against war'
        ]
    }

    def filter_by_angle(dataframe, keywords):
        pattern = '|'.join(keywords)
        return dataframe[dataframe['text'].str.lower().str.contains(pattern, na=False)]

    # ── Cell 56 — Sentiment per angle ────────────────────────────────────
    bar.progress(90, text="Computing sentiment per conflict angle…")
    angle_results = {}
    for angle, keywords in ANGLE_KEYWORDS.items():
        subset = filter_by_angle(df, keywords)
        if len(subset) == 0:
            continue
        summary = subset.groupby('source').agg(
            avg_score=('sentiment_score', 'mean'),
            count=('sentiment_score', 'count')
        ).round(4).sort_values('avg_score', ascending=False)
        angle_results[angle] = summary

    # ── Cell 59 — Aggregates ──────────────────────────────────────────────
    bar.progress(94, text="Aggregating results…")
    avg_by_source = df.groupby('source').agg(
        avg_score=('sentiment_score', 'mean'),
        total_articles=('sentiment_score', 'count')
    ).round(4).sort_values('avg_score', ascending=False)

    avg_by_platform = df.groupby('platform').agg(
        avg_score=('sentiment_score', 'mean'),
        total_articles=('sentiment_score', 'count')
    ).round(4).sort_values('avg_score', ascending=False)

    # ── Cell 61 — Sentiment distribution ─────────────────────────────────
    dist = df.groupby(['source', 'sentiment_label']).size().unstack(fill_value=0)
    dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100

    # ── Cell 65 — Heatmap data ────────────────────────────────────────────
    heatmap_data = {}
    for angle, keywords in ANGLE_KEYWORDS.items():
        subset = filter_by_angle(df, keywords)
        if len(subset) > 0:
            heatmap_data[angle] = subset.groupby('source')['sentiment_score'].mean()
    heatmap_df = pd.DataFrame(heatmap_data).T

    # ── Cell 69 — Angle score df ──────────────────────────────────────────
    angle_avg_scores = {}
    for angle, keywords in ANGLE_KEYWORDS.items():
        subset = filter_by_angle(df, keywords)
        if len(subset) > 0:
            angle_avg_scores[angle] = subset.groupby('source')['sentiment_score'].mean()
    angle_score_df = pd.DataFrame(angle_avg_scores).T

    # ── Save to session state ─────────────────────────────────────────────
    bar.progress(100, text="✅ Pipeline complete!")
    st.session_state.update({
        'df': df, 'lda_model': lda_model,
        'count_matrix': count_matrix, 'vocab_lda': vocab_lda,
        'document_topic_matrix': document_topic_matrix,
        'word_freq': word_freq, 'angle_results': angle_results,
        'avg_by_source': avg_by_source, 'avg_by_platform': avg_by_platform,
        'dist_pct': dist_pct, 'heatmap_df': heatmap_df,
        'angle_score_df': angle_score_df,
        'topic_keywords_dict': topic_keywords_dict,
        'TOPIC_LABELS': TOPIC_LABELS,
        'ANGLE_KEYWORDS': ANGLE_KEYWORDS,
    })
    import time; time.sleep(0.4)
    bar.empty()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def upload_widget(compact=False):
    if not compact:
        st.markdown("""<div class="callout callout-warn">
        <strong>📂 Upload Required:</strong> Upload your
        <code>dataset.csv</code> file.
        The full notebook pipeline runs automatically after upload.</div>""",
        unsafe_allow_html=True)
    f = st.file_uploader(
        "Upload dataset CSV", type=["csv"],
        label_visibility="collapsed" if compact else "visible",
    )
    if f is not None:
        with st.spinner("Running full pipeline… 1–2 min for large datasets."):
            try:
                run_pipeline(f)
                st.success("✅ Pipeline complete! Navigate using the sidebar.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)

def need_data():
    if st.session_state.get('df') is None:
        st.markdown("""<div class="callout callout-warn">
        <strong>📂 No dataset loaded.</strong> Go to
        <em>🏠 Project Overview</em> and upload your CSV first.</div>""",
        unsafe_allow_html=True)
        upload_widget(compact=True)
        return False
    return True

def pc():
    return dict(plot_bgcolor="#161b22", paper_bgcolor="#161b22", font_color="#e6edf3")

def mpl_dark(fig):
    """Apply dark background to matplotlib figures for the dark UI."""
    fig.patch.set_facecolor('#161b22')
    for ax in fig.get_axes():
        ax.set_facecolor('#1c2230')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 10px 0;'>
      <div style='font-family:"Playfair Display",serif;font-size:1.3rem;font-weight:700;color:#e6edf3;'>🌐US / Israel–Iran War Sentiment</div>
      <div style='font-size:0.75rem;color:#8b949e;margin-top:4px;'>BA Project · Q2 Report</div>
    </div>
    <hr style='border-color:#30363d;margin:0 0 16px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("nav", [
        "🏠   Overview",
        "📦  Data Collection",
        "🔧  Pre-processing",
        "🧠  Topic Modeling (LDA)",
        "💬  Sentiment Analysis",
        "📊  Required Visualizations",
        "📋  Methodology & Summary",
    ], label_visibility="collapsed")

    st.markdown('# Group 5')
    st.markdown('Group members:')
    st.markdown('Sheila Semenyo Ayertey -11334501')
    st.markdown('Zanu Christopher-11179138')
    st.markdown('Emmanuel Kofi Atta Aboagye- 11259266')
    st.markdown('Kelvin Larbi Yeboah- 11169462')
    st.markdown('Nyantakyi Bright Kofi- 11296261')



    st.markdown("<hr>", unsafe_allow_html=True)
    df_ss = st.session_state.get('df')
    if df_ss is not None:
        st.markdown(f"""
        <div style='font-size:0.75rem;color:#8b949e;padding:0 4px;'>
          <div style='margin-bottom:6px;'><strong style='color:#22c55e;'>✅ Dataset Loaded</strong></div>
          <div>📁 {len(df_ss):,} records</div>
          <div>📰 {df_ss['source'].nunique()} sources</div>
          <div>🌍 {df_ss['platform'].nunique() if 'platform' in df_ss.columns else '—'} platforms</div>
          <div style='margin-top:8px;'><strong style='color:#e6edf3;'>Tools</strong></div>
          <div>🐍 NLTK · VADER · TextBlob</div><div>📊 sklearn LDA</div><div>📈 Plotly + Matplotlib</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='font-size:0.75rem;color:#e8a838;padding:0 4px;'>
          ⚠ No dataset loaded.<br>Upload CSV on Overview page.
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
def page_overview():
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">Sentiment Analysis<br>with <span>Topic Modeling</span></div>
      <div style='color:#8b949e;font-size:1rem;margin-top:8px;'>US / Israel–Iran War · Media Coverage Analysis</div>
      <div class="hero-meta"><strong>Method:</strong> LDA (sklearn) + VADER + TextBlob (NLTK) · </div>
    </div>""", unsafe_allow_html=True)

    df = st.session_state.get('df')
    if df is None:
        st.markdown("#### 📂 Upload Your Dataset to Begin")
        upload_widget()
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("📰 Records", f"{len(df):,}")
        c2.metric("🗞️ Sources", df['source'].nunique())
        c3.metric("🌍 Platforms", df['platform'].nunique() if 'platform' in df.columns else "—")
        c4.metric("🧠 Topics", "7")
        st.markdown("""<div class="callout callout-success">
        <strong>✅ Pipeline complete.</strong> All notebook code has been executed on your dataset.
        Use the sidebar to navigate each phase.</div>""", unsafe_allow_html=True)
        upload_widget(compact=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Project Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Five phases — all executed automatically from your uploaded CSV</div>', unsafe_allow_html=True)
    for num, title, desc in [
        ("1","Load & Explore the Dataset","Read CSV · shape, columns, null values, source & platform distributions"),
        ("2","Pre-processing","Standardize names → filter junk → lowercase → remove URLs/symbols → tokenize → stop words → POS lemmatize"),
        ("3","Topic Modeling (LDA)","CountVectorizer(min_df=5, max_df=0.9) → LDA(n=7, seed=122) → topic labels → document-topic matrix"),
        ("4","Sentiment Analysis (5 Angles)","VADER + TextBlob on original text → compound score → pos/neu/neg → per-source, per-angle averages"),
        ("5","Required Outputs","Avg sentiment by outlet · Stacked distribution · Diverging bar · Heatmap"),
    ]:
        st.markdown(f"""<div class="phase-card">
        <div class="phase-num">{num}</div>
        <div><div class="phase-title">{title}</div><div class="phase-desc">{desc}</div></div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DATA COLLECTION  — Visual: Cell 10 (bar + pie)
# ═════════════════════════════════════════════════════════════════════════════
def page_data_collection():
    import plotly.graph_objects as go
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    if not need_data(): return
    df = st.session_state['df']

    st.markdown('<div class="section-header">📦 Data Collection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Phase 1 · Source overview, record counts, collection methods</div>', unsafe_allow_html=True)

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total Records", f"{len(df):,}")
    news_n   = int(len(df[df['platform']=='News']))        if 'platform' in df.columns else "—"
    social_n = int(len(df[df['platform']=='Social Media'])) if 'platform' in df.columns else "—"
    m2.metric("News Records",   f"{news_n:,}"   if isinstance(news_n,int)   else news_n)
    m3.metric("Social Records", f"{social_n:,}" if isinstance(social_n,int) else social_n)
    m4.metric("Sources", df['source'].nunique())
    m5.metric("Columns", df.shape[1])
    st.markdown("---")

    # ── DATA COLLECTION SUMMARY TABLE ────────────────────────────────────
    st.markdown("#### 📋 Data Collection Summary Table")
    COLLECTION_METADATA = {
        'BBC':          {'method':'RSS feed + newspaper4k',           'filter':'War keywords: US Iran, Israel Iran, Middle East war'},
        'BBC news':     {'method':'RSS feed + newspaper4k',           'filter':'War keywords: US Iran, Israel Iran, Middle East war'},
        'BBC (NewsAPI)':{'method':'NewsAPI (news-please)',            'filter':'War keywords, English language only'},
        'Al Jazeera':   {'method':'RSS feed + newspaper4k',           'filter':'War keywords: Iran conflict, Middle East, Gaza'},
        'Reuters':      {'method':'RSS feed + newspaper4k',           'filter':'War keywords: Iran, Israel, US strikes'},
        'RT':           {'method':'RSS feed + newspaper4k',           'filter':'War keywords: Iran war, US military, Middle East'},
        'RT News':      {'method':'RSS feed + newspaper4k',           'filter':'War keywords: Iran war, US military, Middle East'},
        'Google News':  {'method':'Google News RSS API (feedparser)', 'filter':'War keywords, English, deduplicated by title'},
        'YouTube':      {'method':'youtube-comment-downloader',       'filter':'Comments from BBC/AJ/Reuters war video uploads'},
        'Facebook':     {'method':'Manual collection (public posts)', 'filter':'Public pages, war-related hashtags and keywords'},
        'Twitter / X':  {'method':'Manual collection (snscrape)',     'filter':'English tweets, war-related hashtags, keyword search'},
    }
    rows = []
    for source in df['source'].unique():
        sub = df[df['source']==source].copy()
        if 'date' in sub.columns:
            dates = pd.to_datetime(sub['date'], errors='coerce').dropna()
            dr = f"{dates.min().strftime('%Y-%m-%d')} → {dates.max().strftime('%Y-%m-%d')}" if len(dates)>0 else 'Unknown'
        else:
            dr = 'Unknown'
        platform = sub['platform'].mode()[0] if 'platform' in sub.columns else 'Unknown'
        meta = COLLECTION_METADATA.get(source, {'method':'RSS / web scraping','filter':'War-related keywords'})
        rows.append({'Outlet / Platform':source,'Type':platform,'Articles / Posts':len(sub),
                     'Date Range':dr,'Collection Method':meta['method'],'Filtering Criteria':meta['filter']})
    df_sum = pd.DataFrame(rows).sort_values('Articles / Posts',ascending=False).reset_index(drop=True)
    def hl(row):
        return ["background-color:rgba(139,92,246,0.12)"]*len(row) if row["Type"]=="Social Media" else ["background-color:rgba(59,130,246,0.06)"]*len(row)
    st.dataframe(df_sum.style.apply(hl,axis=1), use_container_width=True, hide_index=True)
    st.markdown("---")

    # ── CELL 10 — VISUALIZE DATA COLLECTION (reproduced verbatim) ────────
    st.markdown("#### 📊  Articles / Posts Collected by Source")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    source_counts = df['source'].value_counts()
    if 'platform' in df.columns:
        platform_map = df.drop_duplicates('source').set_index('source')['platform'].to_dict()
        bar_colors = ['#2196F3' if platform_map.get(s, '') == 'News' else '#FF5722'
                      for s in source_counts.index]
    else:
        bar_colors = ['#2196F3'] * len(source_counts)
    bars = axes[0].bar(source_counts.index, source_counts.values,
                       color=bar_colors, edgecolor='white', linewidth=0.8)
    axes[0].set_title('Articles / Posts Collected by Source', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Source')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, source_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 5, str(val),
                     ha='center', fontsize=9, fontweight='bold')
    legend_elements = [
        mpatches.Patch(color='#2196F3', label='News'),
        mpatches.Patch(color='#FF5722', label='Social Media'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')
    if 'platform' in df.columns:
        platform_counts = df['platform'].value_counts()
        axes[1].pie(
            platform_counts.values, labels=platform_counts.index, autopct='%1.1f%%',
            colors=['#2196F3', '#FF5722'], startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        axes[1].set_title('News vs Social Media Split', fontweight='bold', fontsize=12)
    else:
        axes[1].text(0.5, 0.5, 'Platform column not found', ha='center', va='center')
    plt.tight_layout()
    st.pyplot(mpl_dark(fig))
    plt.close()
    st.markdown("---")

    st.markdown("#### 🗂️ Dataset Schema")
    st.dataframe(pd.DataFrame({
        "Column": df.columns.tolist(),
        "Dtype":  [str(d) for d in df.dtypes.tolist()],
        "Non-Null": df.notnull().sum().tolist(),
        "Null":   df.isnull().sum().tolist(),
        "Sample": [str(df[c].dropna().iloc[0])[:80] if len(df[c].dropna())>0 else "—" for c in df.columns]
    }), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PRE-PROCESSING  — Visuals: Cell 27 (word freq bar), Cell 28 (word cloud)
# ═════════════════════════════════════════════════════════════════════════════
def page_preprocessing():
    import plotly.graph_objects as go
    import plotly.express as px
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from collections import Counter
    if not need_data(): return
    df        = st.session_state['df']
    word_freq = st.session_state['word_freq']

    st.markdown('<div class="section-header">🔧 Pre-processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Phase 2 · Pipeline applied to your dataset</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Records After Cleaning", f"{len(df):,}")
    c2.metric("Avg Text Length", f"{df['text_length'].mean():.0f} chars" if 'text_length' in df.columns else "—")
    c3.metric("Unique Tokens", f"{len(word_freq):,}" if word_freq else "—")
    c4.metric("Top Word", word_freq.most_common(1)[0][0] if word_freq else "—")
    st.markdown("---")

    st.markdown("#### 🔄 8-Step Cleaning Pipeline")
    steps = [
        ("1","🔡","Lowercase","text = text.lower()"),
        ("2","🔗","Remove URLs","text = re.sub(r'http\\S+|www\\S+', '', text)"),
        ("3","🏷️","Remove @mentions & #hashtags","text = re.sub(r'@\\w+|#\\w+', '', text)"),
        ("4","🔠","Remove non-letter chars","text = re.sub(r'[^a-z\\s]', '', text)"),
        ("5","↔️","Normalize whitespace","text = re.sub(r'\\s+', ' ', text).strip()"),
        ("6","✂️","Tokenize","tokens = word_tokenize(text)"),
        ("7","🚫","Remove stop words","tokens = [t for t in tokens if t not in stop_words and len(t) > 2]"),
        ("8","🌿","POS-aware lemmatize","lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w,t in pos_tag(tokens)]"),
    ]
    tabs = st.tabs([f"Step {s[0]}" for s in steps])
    for tab,(num,icon,title,code) in zip(tabs,steps):
        with tab:
            ca,cb = st.columns([1,1])
            with ca:
                st.markdown(f"""<div class="insight-card"><h4>{icon} Step {num} — {title}</h4></div>""",unsafe_allow_html=True)
            with cb:
                st.code(code, language="python")
    st.markdown("---")

    st.markdown("#### 🔬 Before / After — Samples from Your Data")
    for _,row in df[['text','clean_text']].dropna().head(3).iterrows():
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("*Original:*")
            st.markdown(f"""<div style='background:#1c2230;border:1px solid #30363d;border-radius:8px;padding:12px;font-size:0.82rem;color:#e6edf3;line-height:1.6;max-height:90px;overflow:auto;'>{str(row['text'])[:300]}</div>""",unsafe_allow_html=True)
        with c2:
            st.markdown("*After cleaning:*")
            st.markdown(f"""<div style='background:#1c2230;border:1px solid #22c55e;border-radius:8px;padding:12px;font-size:0.82rem;color:#86efac;line-height:1.6;font-family:monospace;max-height:90px;overflow:auto;'>{str(row['clean_text'])[:300]}</div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("---")

    # ── CELL 27 — Top 20 Most Frequent Words (verbatim) ──────────────────
    st.markdown("#### 📊 Top 20 Most Frequent Words — Cleaned Corpus")
    top_20 = word_freq.most_common(20)
    words_top, counts_top = zip(*top_20)
    plt.figure(figsize=(12, 5))
    bars = plt.bar(words_top, counts_top, color='steelblue', edgecolor='navy')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 20 Most Frequent Words — Cleaned Corpus', fontweight='bold')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    for bar, count in zip(bars, counts_top):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha='center', fontsize=8)
    plt.tight_layout()
    fig27 = plt.gcf()
    st.pyplot(mpl_dark(fig27))
    plt.close()

    # ── CELL 28 — Word Cloud (verbatim) ──────────────────────────────────
    st.markdown("#### ☁️ Top Word Cloud — US/Israel–Iran War Corpus")
    wordcloud = WordCloud(
        width=1000, height=500,
        background_color='black',
        colormap='cool',
        max_words=100
    ).generate_from_frequencies(word_freq)
    plt.figure(figsize=(14, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud — US/Israel–Iran War Corpus', fontsize=14, fontweight='bold',
              color='white', bbox=dict(facecolor='black', edgecolor='none'))
    plt.tight_layout()
    fig28 = plt.gcf()
    fig28.patch.set_facecolor('black')
    st.pyplot(fig28)
    plt.close()

    if 'text_length' in df.columns:
        st.markdown("---")
        st.markdown("#### 📏 Text Length Distribution by Source")
        fig_b = px.box(df, x='source', y='text_length', color='source',
                       title='Text Length (chars) per Source',
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig_b.update_layout(**pc(), xaxis=dict(tickangle=-30, gridcolor="#30363d"),
                            yaxis=dict(gridcolor="#30363d"), showlegend=False, height=400)
        st.plotly_chart(fig_b, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TOPIC MODELING
# Visuals: Cell 41 (doc-topic heatmap), Cell 42 (topic bar), Cell 44 (source×topic heatmap),
#          Cell 46 (7 word clouds per topic), Cell 47 (stacked bar topic by source)
# ═════════════════════════════════════════════════════════════════════════════
def page_topic_modeling():
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from wordcloud import WordCloud
    if not need_data(): return

    df            = st.session_state['df']
    lda_model     = st.session_state['lda_model']
    doc_topic_mat = st.session_state['document_topic_matrix']
    kw_dict       = st.session_state['topic_keywords_dict']
    TOPIC_LABELS  = st.session_state['TOPIC_LABELS']
    vocab_lda     = st.session_state['vocab_lda']

    ICONS  = {0:"💣",1:"🩺",2:"🤝",3:"🌍",4:"💰",5:"🗺️",6:"📺"}
    COLORS = {0:"#ef4444",1:"#f97316",2:"#3b82f6",3:"#8b5cf6",4:"#22c55e",5:"#e8a838",6:"#ec4899"}
    NUM_TOPICS = 7

    st.markdown('<div class="section-header">🧠 Topic Modeling (LDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Phase 3 · 7 topics discovered from your dataset</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Topics (k)","7"); c2.metric("min_df","5"); c3.metric("max_df","0.90"); c4.metric("random_state","122")
    st.markdown("---")

    # Topic keyword pills
    st.markdown("#### 🏷️ Topics & Keywords")
    ct,cb2 = st.columns(4), st.columns(3)
    for i in range(7):
        kws   = kw_dict.get(TOPIC_LABELS[i], [])
        color = COLORS[i]
        pills = " ".join([f"<span style='background:#1c2230;border:1px solid {color}40;color:{color};padding:2px 8px;border-radius:99px;font-size:0.72rem;display:inline-block;margin:2px;'>{k}</span>" for k in kws[:8]])
        with (ct+cb2)[i]:
            st.markdown(f"""<div style='background:#161b22;border:1px solid {color}40;border-top:3px solid {color};
            border-radius:10px;padding:14px;margin-bottom:8px;'>
            <div style='font-size:1.3rem;margin-bottom:4px;'>{ICONS[i]}</div>
            <div style='color:#e6edf3;font-weight:600;font-size:0.85rem;margin-bottom:8px;'>Topic {i+1}: {TOPIC_LABELS[i]}</div>
            <div style='line-height:2;'>{pills}</div></div>""",unsafe_allow_html=True)
    st.markdown("---")

    # ── CELL 41 — Document–Topic Distribution Heatmap (50 docs) ──────────
    st.markdown("#### 🔥  Document–Topic Distribution Heatmap (Sample of 50 Documents)")
    np.random.seed(0)
    sample_idx = np.random.choice(len(doc_topic_mat), size=min(50, len(doc_topic_mat)), replace=False)
    sample_matrix = doc_topic_mat[sample_idx]
    short_labels = [f'T{i+1}' for i in range(NUM_TOPICS)]
    doc_labels   = [f'Doc {i+1}' for i in range(len(sample_idx))]
    df_doc_topic = pd.DataFrame(sample_matrix, index=doc_labels, columns=short_labels)
    fig41, ax41 = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_doc_topic, cmap='YlOrRd', annot=False, linewidths=0.3, ax=ax41)
    ax41.set_title('Document–Topic Distribution Heatmap (Sample of 50 Documents)', fontweight='bold')
    ax41.set_xlabel('Topics')
    ax41.set_ylabel('Documents')
    plt.tight_layout()
    st.pyplot(mpl_dark(fig41))
    plt.close()
    st.markdown("---")

    # ── CELL 42 — Topic Distribution Horizontal Bar ───────────────────────
    st.markdown("#### 📊 Topic Distribution Across All Sources")
    topic_counts = df['dominant_topic'].value_counts()
    fig42, ax42 = plt.subplots(figsize=(10, 5))
    bars42 = ax42.barh(topic_counts.index, topic_counts.values, color='steelblue', edgecolor='white')
    ax42.set_xlabel('Number of Documents')
    ax42.set_title('Topic Distribution Across All Sources', fontsize=14, fontweight='bold')
    for bar, val in zip(bars42, topic_counts.values):
        ax42.text(val + 2, bar.get_y() + bar.get_height() / 2, str(val), va='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(mpl_dark(fig42))
    plt.close()
    st.markdown("---")

    # ── CELL 44 — Topic Focus by Source Heatmap ───────────────────────────
    st.markdown("#### 🗺️ Topic Focus by Source (% of Articles)")
    topic_by_source = df.groupby(['source', 'dominant_topic']).size().unstack(fill_value=0)
    topic_by_source_pct = topic_by_source.div(topic_by_source.sum(axis=1), axis=0) * 100
    fig44, ax44 = plt.subplots(figsize=(13, 6))
    sns.heatmap(
        topic_by_source_pct.round(1), annot=True, fmt='.1f', cmap='Blues',
        linewidths=0.5, linecolor='white', cbar_kws={'label': '% of source articles'}, ax=ax44
    )
    ax44.set_title('Topic Focus by Source (% of Articles)', fontsize=13, fontweight='bold')
    ax44.set_xlabel(''); ax44.set_ylabel('')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(mpl_dark(fig44))
    plt.close()
    st.markdown("---")

    # ── CELL 46 — Word Cloud per Topic (7 clouds) ─────────────────────────
    st.markdown("#### ☁️  Word Clouds per Topic")
    NUM_TOP_WORDS = 15
    cols46 = st.columns(2)
    for i, topic in enumerate(lda_model.components_):
        top_word_indices = topic.argsort()[:-NUM_TOP_WORDS - 1:-1]
        top_words = [vocab_lda[idx] for idx in top_word_indices]
        word_string = ' '.join(top_words)
        wc = WordCloud(width=800, height=400, background_color='black',
                       relative_scaling=0).generate(word_string)
        fig_wc, ax_wc = plt.subplots(figsize=(7, 3.5))
        ax_wc.imshow(wc)
        ax_wc.set_title(f"Topic {i + 1}: {TOPIC_LABELS.get(i, '')}", fontsize=13, fontweight='bold')
        ax_wc.axis('off')
        fig_wc.patch.set_facecolor('black')
        ax_wc.set_facecolor('black')
        ax_wc.title.set_color('#e6edf3')
        plt.tight_layout()
        with cols46[i % 2]:
            st.pyplot(fig_wc)
        plt.close()
    st.markdown("---")

    # ── CELL 47 — Topic Distribution by Source Stacked Bar ───────────────
    st.markdown("#### 📊  Topic Distribution by Source (%)")
    topic_by_source2 = df.groupby(['source', 'dominant_topic']).size().unstack(fill_value=0)
    topic_by_source_pct2 = topic_by_source2.div(topic_by_source2.sum(axis=1), axis=0) * 100
    fig47, ax47 = plt.subplots(figsize=(13, 6))
    topic_by_source_pct2.plot(kind='bar', ax=ax47, colormap='Set2', edgecolor='white', width=0.8)
    ax47.set_title("Topic Distribution by Source (% of each source's content)", fontweight='bold')
    ax47.set_xlabel('Source')
    ax47.set_ylabel('Percentage (%)')
    plt.xticks(rotation=30, ha='right')
    ax47.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.35, 1))
    plt.tight_layout()
    st.pyplot(mpl_dark(fig47))
    plt.close()

    st.markdown("---")
    with st.expander("💻 LDA Code (from notebook — unchanged)"):
        st.code("""corpus = df['clean_text'].dropna().tolist()
count_vectorizer = CountVectorizer(min_df=5, max_df=0.9)
count_matrix = count_vectorizer.fit_transform(corpus)
vocab_lda = count_vectorizer.get_feature_names_out()

NUM_TOPICS = 7
lda_model = LatentDirichletAllocation(
    n_components=NUM_TOPICS, random_state=122, max_iter=100
)
lda_model.fit(count_matrix)

document_topic_matrix = lda_model.transform(count_matrix)
dominant_topic_indices = document_topic_matrix.argmax(axis=1)
df['dominant_topic'] = [TOPIC_LABELS.get(i, f'Topic {i+1}') for i in dominant_topic_indices]""",
        language="python")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT ANALYSIS
# Visuals: Cell 53 (VADER pie + TextBlob pie + histogram),
#          Cell 55 (VADER by source + TextBlob by source grouped bars),
#          Cell 70 (heatmap source×angle), Cell 72 (platform stacked bar + boxplot),
#          Cell 74 (grouped bar per angle)
# ═════════════════════════════════════════════════════════════════════════════
def page_sentiment():
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mtick
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    if not need_data(): return

    df             = st.session_state['df']
    avg_by_source  = st.session_state['avg_by_source']
    avg_by_platform= st.session_state['avg_by_platform']
    angle_results  = st.session_state['angle_results']
    heatmap_df     = st.session_state['heatmap_df']
    angle_score_df = st.session_state['angle_score_df']
    ANGLE_KEYWORDS = st.session_state['ANGLE_KEYWORDS']

    st.markdown('<div class="section-header">💬 Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Phase 4 · VADER + TextBlob results from your dataset</div>', unsafe_allow_html=True)
    st.markdown("""<div class="callout"><strong>🔍 VADER applied to original text</strong> (not clean_text).
    Compound: <strong>≥ 0.05 = Positive · ≤ -0.05 = Negative · otherwise = Neutral</strong></div>""",
    unsafe_allow_html=True)
    st.markdown("---")

    # Per-source score badges
    sources = avg_by_source.index.tolist()
    scores  = avg_by_source['avg_score'].tolist()
    cols_b = st.columns(min(4, len(sources)))
    for i,(src,score) in enumerate(zip(sources,scores)):
        color = "#22c55e" if score>=0.05 else "#ef4444" if score<=-0.05 else "#9ca3af"
        badge = "🟢 Positive" if score>=0.05 else "🔴 Negative" if score<=-0.05 else "⚪ Neutral"
        with cols_b[i%4]:
            st.markdown(f"""<div style='background:#161b22;border:1px solid #30363d;border-left:4px solid {color};
            border-radius:10px;padding:12px 14px;margin-bottom:10px;'>
            <div style='color:#8b949e;font-size:0.75rem;text-transform:uppercase;font-weight:600;'>{src}</div>
            <div style='color:{color};font-size:1.6rem;font-weight:700;margin:4px 0;'>{score:+.4f}</div>
            <div style='color:#8b949e;font-size:0.78rem;'>{badge}</div></div>""",unsafe_allow_html=True)
    st.markdown("---")

    # ── CELL 53 — VADER pie + TextBlob pie + VADER score histogram ────────
    st.markdown("#### 📊 Overall Sentiment Analysis Results (VADER + TextBlob)")
    colors53 = {'Positive': '#4CAF50', 'Neutral': '#FF9800', 'Negative': '#F44336'}
    fig53, axes53 = plt.subplots(1, 3, figsize=(16, 5))

    vader_dist = df['vader_class'].value_counts()
    axes53[0].pie(vader_dist.values, labels=vader_dist.index, autopct='%1.1f%%',
                  colors=[colors53.get(l,'grey') for l in vader_dist.index], startangle=90)
    axes53[0].set_title('VADER Sentiment Distribution', fontweight='bold')

    tb_dist = df['tb_class'].value_counts()
    axes53[1].pie(tb_dist.values, labels=tb_dist.index, autopct='%1.1f%%',
                  colors=[colors53.get(l,'grey') for l in tb_dist.index], startangle=90)
    axes53[1].set_title('TextBlob Sentiment Distribution', fontweight='bold')

    axes53[2].hist(df['vader_score'], bins=30, color='steelblue', edgecolor='white')
    axes53[2].axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Neutral boundary')
    axes53[2].axvline(x=df['vader_score'].mean(), color='green', linestyle='--',
                      linewidth=1.5, label=f"Mean={df['vader_score'].mean():.2f}")
    axes53[2].set_title('VADER Score Distribution', fontweight='bold')
    axes53[2].set_xlabel('Compound Score')
    axes53[2].set_ylabel('Frequency')
    axes53[2].legend(fontsize=8)

    plt.suptitle('Overall Sentiment Analysis Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(mpl_dark(fig53))
    plt.close()
    st.markdown("---")

    # ── CELL 55 — VADER by Source + TextBlob by Source grouped bars ───────
    st.markdown("#### Sentiment Distribution Across Sources (VADER & TextBlob)")
    colors55 = {'Positive': '#4CAF50', 'Neutral': '#FF9800', 'Negative': '#F44336'}
    fig55, axes55 = plt.subplots(1, 2, figsize=(14, 5))

    vader_src = df.groupby(['source', 'vader_class']).size().unstack(fill_value=0)
    vader_src_pct = vader_src.div(vader_src.sum(axis=1), axis=0) * 100
    vader_src_pct.plot(kind='bar', ax=axes55[0],
                       color=[colors55.get(c,'grey') for c in vader_src_pct.columns],
                       edgecolor='white', width=0.7)
    axes55[0].set_title('VADER Sentiment by Source (%)', fontweight='bold')
    axes55[0].set_xlabel('Source')
    axes55[0].set_ylabel('Percentage (%)')
    axes55[0].tick_params(axis='x', rotation=30)
    axes55[0].legend(title='Sentiment', fontsize=9)

    tb_src = df.groupby(['source', 'tb_class']).size().unstack(fill_value=0)
    tb_src_pct = tb_src.div(tb_src.sum(axis=1), axis=0) * 100
    tb_src_pct.plot(kind='bar', ax=axes55[1],
                    color=[colors55.get(c,'grey') for c in tb_src_pct.columns],
                    edgecolor='white', width=0.7)
    axes55[1].set_title('TextBlob Sentiment by Source (%)', fontweight='bold')
    axes55[1].set_xlabel('Source')
    axes55[1].set_ylabel('Percentage (%)')
    axes55[1].tick_params(axis='x', rotation=30)
    axes55[1].legend(title='Sentiment', fontsize=9)

    plt.suptitle('Sentiment Distribution Across Sources', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(mpl_dark(fig55))
    plt.close()
    st.markdown("---")

    # ── CELL 70 — Sentiment Heatmap: Source × Conflict Angle ─────────────
    st.markdown("#### 🗺️ Sentiment Heatmap: Source × Conflict Angle")
    def filter_by_angle(dataframe, keywords):
        pattern = '|'.join(keywords)
        return dataframe[dataframe['text'].str.lower().str.contains(pattern, na=False)]

    heatmap_data70 = {}
    for angle, keywords in ANGLE_KEYWORDS.items():
        subset = filter_by_angle(df, keywords)
        if len(subset) > 0:
            heatmap_data70[angle] = subset.groupby('source')['sentiment_score'].mean()
    heatmap_df70 = heatmap_df  # already computed in pipeline

    if heatmap_df70 is not None and not heatmap_df70.empty:
        fig70, ax70 = plt.subplots(figsize=(13, 5))
        sns.heatmap(
            heatmap_df70.round(3), annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Avg VADER Score'}, ax=ax70
        )
        ax70.set_title('Sentiment Heatmap: Source × Conflict Angle', fontsize=13, fontweight='bold')
        ax70.set_xlabel(''); ax70.set_ylabel('')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(mpl_dark(fig70))
        plt.close()
    st.markdown("---")

    # ── CELL 72 — News vs Social Media (stacked bar + box plot) ──────────
    st.markdown("#### 📊  News vs Social Media Sentiment Comparison")
    colors72 = {'positive': '#4CAF50', 'neutral': '#9E9E9E', 'negative': '#F44336'}
    platform_sentiment = df.groupby(['platform', 'sentiment_label']).size().unstack(fill_value=0)
    platform_pct = platform_sentiment.div(platform_sentiment.sum(axis=1), axis=0) * 100
    fig72, axes72 = plt.subplots(1, 2, figsize=(13, 5))
    cols72 = [c for c in ['positive', 'neutral', 'negative'] if c in platform_pct.columns]
    platform_pct[cols72].plot(
        kind='bar', stacked=True,
        color=[colors72.get(c, 'gray') for c in cols72],
        ax=axes72[0], edgecolor='white'
    )
    axes72[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes72[0].set_title('Sentiment: News vs Social Media', fontsize=12, fontweight='bold')
    axes72[0].set_xlabel('')
    axes72[0].set_ylabel('Percentage')
    axes72[0].tick_params(axis='x', rotation=0)
    axes72[0].legend(title='Sentiment')
    df.boxplot(column='sentiment_score', by='platform', ax=axes72[1], grid=False)
    axes72[1].set_title('VADER Score Distribution by Platform', fontsize=12, fontweight='bold')
    axes72[1].set_xlabel('')
    axes72[1].set_ylabel('VADER Compound Score')
    plt.suptitle('')
    plt.tight_layout()
    st.pyplot(mpl_dark(fig72))
    plt.close()
    st.markdown("---")

    # ── CELL 74 — Avg Sentiment per Source by Conflict Angle ─────────────
    st.markdown("#### 📊 Average Sentiment per Source — by Conflict Angle")
    if angle_score_df is not None and not angle_score_df.empty:
        fig74, ax74 = plt.subplots(figsize=(14, 6))
        angle_score_df.plot(kind='bar', ax=ax74, edgecolor='white', width=0.75)
        ax74.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax74.set_title('Average Sentiment per Source — by Conflict Angle',
                       fontsize=13, fontweight='bold')
        ax74.set_ylabel('Avg VADER Compound Score')
        ax74.set_xlabel('')
        ax74.set_ylim(-1, 1)
        plt.xticks(rotation=25, ha='right', fontsize=9)
        ax74.legend(title='Source', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        st.pyplot(mpl_dark(fig74))
        plt.close()

    st.markdown("---")
    # Angle tables
    st.markdown("#### 📋 Sentiment per Angle & Source (Tables)")
    for angle, summary in angle_results.items():
        with st.expander(f"📌 {angle}  (n = {int(summary['count'].sum()):,})"):
            st.dataframe(summary.reset_index(), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: REQUIRED VISUALIZATIONS
# Visuals: Cell 66 (stacked bar pos/neu/neg), Cell 68 (avg VADER bar by source),
#          Cell 70 heatmap (interactive Plotly version), diverging bar, treemap
# ═════════════════════════════════════════════════════════════════════════════
def page_visualizations():
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mtick
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    if not need_data(): return

    df            = st.session_state['df']
    avg_by_source = st.session_state['avg_by_source']
    dist_pct      = st.session_state['dist_pct']
    heatmap_df    = st.session_state['heatmap_df']
    word_freq     = st.session_state['word_freq']

    st.markdown('<div class="section-header">📊 Required Outputs & Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Phase 5 · All mandatory deliverables from the notebook</div>', unsafe_allow_html=True)
    st.markdown("""<div class="callout callout-success"><strong>✅ Mandatory Deliverables:</strong>
    (1) Avg sentiment by outlet/platform · (2) Positive/Neutral/Negative stacked bar ·
    (3) Source comparison bar · (4) Sentiment heatmap (bonus)</div>""", unsafe_allow_html=True)
    st.markdown("---")

    sources = avg_by_source.index.tolist()
    scores  = avg_by_source['avg_score'].tolist()
    bcolors = ["#4CAF50" if s>=0.05 else "#F44336" if s<=-0.05 else "#9E9E9E" for s in scores]
    colors_sent = {'positive':'#4CAF50','neutral':'#9E9E9E','negative':'#F44336'}

    # ── OUTPUT 1 — Average Sentiment Score by Outlet (Plotly) ────────────
    st.markdown("#### 📌  Average Sentiment Score by Outlet / Platform")
    fig1 = go.Figure(go.Bar(x=sources, y=scores, marker_color=bcolors,
                            text=[f"{s:+.4f}" for s in scores], textposition="outside", textfont_color="#e6edf3"))
    fig1.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
    fig1.add_hline(y=0.05, line_dash="dot", line_color="#4CAF50", line_width=0.8,
                   annotation_text="Positive (≥0.05)", annotation_font_color="#4CAF50")
    fig1.add_hline(y=-0.05, line_dash="dot", line_color="#F44336", line_width=0.8,
                   annotation_text="Negative (≤-0.05)", annotation_font_color="#F44336")
    fig1.update_layout(**pc(), title="Sentiment Score by Source — US/Israel–Iran War Coverage",
                       yaxis=dict(range=[-1,1], gridcolor="#30363d", title="Average VADER Compound Score"),
                       xaxis=dict(tickangle=-30, gridcolor="#30363d"), height=480, margin=dict(t=50))
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("---")

    # ── CELL 66 — Stacked Bar Pos/Neu/Neg (verbatim matplotlib) ──────────
    st.markdown("#### 📌  Positive / Neutral / Negative Distribution (Stacked Bar)")
    colors66 = {'positive': '#4CAF50', 'neutral': '#9E9E9E', 'negative': '#F44336'}
    cols_to_plot = [c for c in ['positive', 'neutral', 'negative'] if c in dist_pct.columns]
    fig66, ax66 = plt.subplots(figsize=(13, 6))
    dist_pct[cols_to_plot].plot(
        kind='bar', stacked=True,
        color=[colors66.get(c, 'gray') for c in cols_to_plot],
        ax=ax66, edgecolor='white', linewidth=0.5
    )
    ax66.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax66.set_title('Sentiment Distribution by Source', fontsize=14, fontweight='bold')
    ax66.set_xlabel('')
    ax66.set_ylabel('Percentage of Articles')
    plt.xticks(rotation=30, ha='right')
    ax66.legend(title='Sentiment', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(mpl_dark(fig66))
    plt.close()
    st.markdown("---")

    # ── CELL 68 — Comparing Sentiment Across Sources (verbatim matplotlib)
    st.markdown("#### 📌 Comparing Sentiment Across All Sources")
    outlets = avg_by_source.index.tolist()
    scores68 = avg_by_source['avg_score'].tolist()
    bar_colors68 = [
        '#4CAF50' if s >= 0.05 else '#F44336' if s <= -0.05 else '#9E9E9E'
        for s in scores68
    ]
    fig68, ax68 = plt.subplots(figsize=(13, 6))
    bars68 = ax68.bar(outlets, scores68, color=bar_colors68, edgecolor='white', width=0.6)
    ax68.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax68.set_ylim(-1, 1)
    ax68.set_ylabel('Average VADER Compound Score', fontsize=11)
    ax68.set_title('Sentiment Score by Source — US/Israel–Iran War Coverage',
                   fontsize=13, fontweight='bold')
    ax68.set_xlabel('')
    plt.xticks(rotation=30, ha='right', fontsize=10)
    for bar, score in zip(bars68, scores68):
        ypos = bar.get_height() + 0.02 if score >= 0 else bar.get_height() - 0.06
        ax68.text(bar.get_x() + bar.get_width() / 2, ypos,
                  f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    legend_patches68 = [
        mpatches.Patch(color='#4CAF50', label='Positive (≥ 0.05)'),
        mpatches.Patch(color='#9E9E9E', label='Neutral'),
        mpatches.Patch(color='#F44336', label='Negative (≤ -0.05)'),
    ]
    ax68.legend(handles=legend_patches68, loc='upper right')
    plt.tight_layout()
    st.pyplot(mpl_dark(fig68))
    plt.close()
    st.markdown("---")

    # ── OUTPUT 4 BONUS — Diverging Bar (Plotly) ───────────────────────────
    st.markdown("#### 📌 Diverging Sentiment Comparison")
    fig_div = go.Figure()
    nv  = [dist_pct.loc[s,'negative'] if s in dist_pct.index and 'negative' in dist_pct.columns else 0 for s in sources]
    nuv = [dist_pct.loc[s,'neutral']  if s in dist_pct.index and 'neutral'  in dist_pct.columns else 0 for s in sources]
    pv  = [dist_pct.loc[s,'positive'] if s in dist_pct.index and 'positive' in dist_pct.columns else 0 for s in sources]
    fig_div.add_trace(go.Bar(name="Negative", x=[-v for v in nv], y=sources, orientation="h", marker_color="#F44336",
                             text=[f"-{v:.1f}%" for v in nv], textposition="inside", textfont_color="#fff"))
    fig_div.add_trace(go.Bar(name="Neutral",  x=nuv, y=sources, orientation="h", marker_color="#9E9E9E",
                             text=[f"{v:.1f}%" for v in nuv], textposition="inside", textfont_color="#fff"))
    fig_div.add_trace(go.Bar(name="Positive", x=pv,  y=sources, orientation="h", marker_color="#4CAF50",
                             text=[f"{v:.1f}%" for v in pv], textposition="inside", textfont_color="#fff"))
    fig_div.add_vline(x=0, line_color="#8b949e", line_width=1.5)
    fig_div.update_layout(**pc(), barmode="relative", title="Diverging Sentiment Comparison Across All Sources",
                          xaxis=dict(gridcolor="#30363d", title="← Negative   |   Positive →", ticksuffix="%"),
                          yaxis=dict(gridcolor="#30363d"), legend=dict(font_color="#e6edf3"),
                          height=480, margin=dict(t=50))
    st.plotly_chart(fig_div, use_container_width=True)
    st.markdown("---")

    # ── OUTPUT 4 BONUS — Heatmap Source × Angle (Plotly) ─────────────────
    st.markdown("#### 📌  Sentiment Heatmap: Source × Conflict Angle")
    if heatmap_df is not None and not heatmap_df.empty:
        fig4 = go.Figure(go.Heatmap(
            z=heatmap_df.values, x=heatmap_df.columns.tolist(), y=heatmap_df.index.tolist(),
            colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
            text=heatmap_df.values.round(3), texttemplate="%{text}", textfont_size=11,
            colorbar=dict(title="Avg VADER Score", tickfont_color="#e6edf3", title_font_color="#e6edf3")
        ))
        fig4.update_layout(**pc(), title="Sentiment Heatmap: Source × Conflict Angle (Red=Negative · Green=Positive)",
                           xaxis=dict(tickangle=-25), height=440, margin=dict(t=50, b=80))
        st.plotly_chart(fig4, use_container_width=True)
    st.markdown("---")

    # ── Additional — Topic bar + Word treemap ─────────────────────────────
    st.markdown("#### 📊 Topic Distribution & Word Frequencies")
    ca, cb = st.columns(2)
    with ca:
        tc = df['dominant_topic'].value_counts()
        fig_t = go.Figure(go.Bar(x=tc.values.tolist(), y=tc.index.tolist(), orientation='h',
                                 marker_color="#3b82f6", text=tc.values.tolist(),
                                 textposition="outside", textfont_color="#8b949e"))
        fig_t.update_layout(**pc(), title="Document Count per LDA Topic",
                            xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"),
                            height=380, margin=dict(t=50, r=80))
        st.plotly_chart(fig_t, use_container_width=True)
    with cb:
        tw = dict(word_freq.most_common(25))
        fig_w = go.Figure(go.Treemap(
            labels=list(tw.keys()), values=list(tw.values()),
            parents=[""]*len(tw), marker_colorscale="Blues",
            textinfo="label+value", textfont_size=12
        ))
        fig_w.update_layout(**pc(), title="Top 25 Words — Post-Cleaning", height=380, margin=dict(t=50))
        st.plotly_chart(fig_w, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: METHODOLOGY
# ═════════════════════════════════════════════════════════════════════════════
def page_methodology():
    import pandas as pd
    if not need_data(): return

    df = st.session_state['df']
    avg_by_source = st.session_state['avg_by_source']

    st.markdown('<div class="section-header">📋 Methodology & Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Technical decisions, limitations, and conclusions from your data</div>', unsafe_allow_html=True)
    st.markdown("---")

    tabs = st.tabs(["📐 Parameters","⚠️ Limitations","🎯 Conclusions from Your Data","📋 Reference Table"])

    with tabs[0]:
        for title,body in [
            ("📦 Dataset","Source names standardized (RT → RT News) · Short/empty records filtered (len < 30) · text_length column added"),
            ("🔧 Pre-processing","Lowercase → URL removal → @/# removal → non-letter strip → whitespace normalize → word_tokenize → stop words (NLTK + 19 custom) → POS-aware lemmatization → drop if clean ≤ 20 chars"),
            ("🧠 LDA","CountVectorizer(min_df=5, max_df=0.9) · LDA(n_components=7, random_state=122, max_iter=100) · dominant_topic = argmax of document-topic row"),
            ("💬 VADER","Applied to original text · compound ≥ 0.05 = positive · compound ≤ -0.05 = negative · otherwise = neutral"),
            ("📝 TextBlob","Applied to original text · polarity (-1→+1) · subjectivity (0=objective, 1=subjective) · objectivity = 1 - subjectivity"),
            ("🎯 Angles","5 angles via keyword matching on original text · filter_by_angle() · angles are not mutually exclusive"),
        ]:
            st.markdown(f"""<div class="insight-card"><h4>{title}</h4><p>{body}</p></div>""",unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("""<div class="insight-card"><h4>⚠️ Limitations & Caveats</h4><p>
        <strong>Social media sample size:</strong> Social media records are a small fraction. Findings are exploratory.<br><br>
        <strong>Source imbalance:</strong> Google News dominates many datasets. Short headlines bias sentiment vs. full articles.<br><br>
        <strong>VADER scope:</strong> Designed for English social media. Sarcasm and complex geopolitical language may be misclassified.<br><br>
        <strong>Keyword angles:</strong> Documents can match multiple angles. Each angle is treated independently.
        </p></div>""",unsafe_allow_html=True)

    with tabs[2]:
        most_neg = avg_by_source['avg_score'].idxmin()
        most_pos = avg_by_source['avg_score'].idxmax()
        top_topic = df['dominant_topic'].value_counts().idxmax()
        top_topic_n = int(df['dominant_topic'].value_counts().iloc[0])
        top_topic_pct = top_topic_n/len(df)*100
        st.markdown(f"""
        <div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px;margin-bottom:12px;display:flex;gap:14px;'>
          <div style='font-size:1.6rem;'>🔴</div>
          <div><div style='color:#e8a838;font-weight:600;'>Most Negative Source</div>
          <div style='color:#8b949e;font-size:0.87rem;'><strong style='color:#e6edf3;'>{most_neg}</strong> — avg score {avg_by_source.loc[most_neg,'avg_score']:+.4f}</div></div></div>
        <div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px;margin-bottom:12px;display:flex;gap:14px;'>
          <div style='font-size:1.6rem;'>🟢</div>
          <div><div style='color:#e8a838;font-weight:600;'>Most Positive Source</div>
          <div style='color:#8b949e;font-size:0.87rem;'><strong style='color:#e6edf3;'>{most_pos}</strong> — avg score {avg_by_source.loc[most_pos,'avg_score']:+.4f}</div></div></div>
        <div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px;margin-bottom:12px;display:flex;gap:14px;'>
          <div style='font-size:1.6rem;'>💣</div>
          <div><div style='color:#e8a838;font-weight:600;'>Dominant LDA Topic</div>
          <div style='color:#8b949e;font-size:0.87rem;'><strong style='color:#e6edf3;'>{top_topic}</strong> — {top_topic_n:,} documents ({top_topic_pct:.1f}% of corpus)</div></div></div>
        """,unsafe_allow_html=True)

    with tabs[3]:
        st.dataframe(pd.DataFrame({
            "Parameter":["Source standardization","Short text filter","Stop words","Tokenizer","Lemmatizer",
                         "Vectorizer","LDA topics (k)","LDA min_df","LDA max_df","LDA random_state","LDA max_iter",
                         "Sentiment tool","VADER input","Positive threshold","Negative threshold"],
            "Value":["RT → RT News","len(text) < 30 or empty","NLTK English + 19 custom","NLTK word_tokenize",
                     "WordNetLemmatizer + POS tags","CountVectorizer","7","5","0.9","122","100",
                     "VADER SentimentIntensityAnalyzer","Original raw text","compound ≥ 0.05","compound ≤ -0.05"],
            "Notebook Cell":["Cell 18","Cell 20","Cell 22","Cell 24","Cell 24","Cell 32","Cell 34",
                              "Cell 32","Cell 32","Cell 34","Cell 34","Cell 50","Cell 50","Cell 50","Cell 50"]
        }), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
if   "Overview"    in page: page_overview()
elif "Collection"  in page: page_data_collection()
elif "processing"  in page: page_preprocessing()
elif "Topic"       in page: page_topic_modeling()
elif "Sentiment"   in page: page_sentiment()
elif "Visualiz"    in page: page_visualizations()
elif "Methodology" in page: page_methodology()