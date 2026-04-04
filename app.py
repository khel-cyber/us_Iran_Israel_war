"""
Sentiment Analysis with Topic Modeling — US / Israel–Iran War
Business Analyst Project | Q2 Full Report
Run: streamlit run app.py
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="US / Israel–Iran War Sentiment Analysis",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
[data-testid="stFileUploaderDropzone"] { padding:8px !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { display:none !important; }
[data-testid="stFileUploader"] label { display:none !important; }
[data-testid="stFileUploader"] > div > label { display:none !important; }
[data-testid="stFileUploaderDropzone"] > div > button { width:100% !important; }
div[data-testid="stFileUploader"] section { min-height:unset !important; padding:4px !important; }
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


@st.cache_resource(show_spinner="Setting up NLP libraries…")
def load_nltk():
    import nltk
    for pkg in ['stopwords','punkt','punkt_tab','wordnet','omw-1.4',
                'averaged_perceptron_tagger_eng','vader_lexicon']:
        nltk.download(pkg, quiet=True)
    return True

load_nltk()

for k in ['df','lda_model','count_matrix','vocab_lda','document_topic_matrix',
          'word_freq','angle_results','avg_by_source','avg_by_platform',
          'dist_pct','heatmap_df','angle_score_df','topic_keywords_dict',
          'TOPIC_LABELS','ANGLE_KEYWORDS']:
    if k not in st.session_state:
        st.session_state[k] = None


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE — notebook code verbatim, only upload source changed
# ═════════════════════════════════════════════════════════════════════════════
def run_pipeline(uploaded_file):
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

    # Cell 8 — only change: uploader instead of file path
    df = pd.read_csv(uploaded_file)
    bar.progress(5, text=f"✅ Loaded {len(df):,} records")

    # Cell 15
    df['text_length'] = df['text'].str.len()

    # Cell 18
    df['source'] = df['source'].replace({'RT': 'RT News'})

    # Cell 20
    bar.progress(8, text="Filtering short/empty records…")
    before = len(df)
    df = df[df['text'].str.len() >= 30].copy()
    df = df[df['text'].str.strip() != ''].copy()
    df = df.reset_index(drop=True)

    # Cell 22
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

    # Cell 24
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'): return 'a'
        elif treebank_tag.startswith('V'): return 'v'
        elif treebank_tag.startswith('N'): return 'n'
        elif treebank_tag.startswith('R'): return 'r'
        else: return 'n'

    def clean_text(text):
        if not isinstance(text, str) or len(text.strip()) == 0: return ''
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        pos_tagged = pos_tag(tokens)
        lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tagged]
        return ' '.join(lemmatized)

    # Cell 26
    bar.progress(20, text="Cleaning & lemmatizing text… (takes 1–2 min)")
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 20].copy()
    df = df.reset_index(drop=True)

    # Cell 28
    bar.progress(52, text="Computing word frequencies…")
    all_tokens = ' '.join(df['clean_text']).split()
    word_freq = Counter(all_tokens)

    # Cell 32
    bar.progress(58, text="Vectorizing corpus…")
    corpus = df['clean_text'].dropna().tolist()
    count_vectorizer = CountVectorizer(min_df=5, max_df=0.9)
    count_matrix = count_vectorizer.fit_transform(corpus)
    vocab_lda = count_vectorizer.get_feature_names_out()

    # Cell 34
    bar.progress(65, text="Training LDA model (7 topics)…")
    NUM_TOPICS = 7
    lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=122, max_iter=100)
    lda_model.fit(count_matrix)

    # Cell 38
    TOPIC_LABELS = {
        0: 'Military Operations', 1: 'Civilian & Humanitarian',
        2: 'Diplomatic Negotiations', 3: 'Geopolitical Tensions',
        4: 'Economic Impact', 5: 'Regional Responses', 6: 'Media Narratives',
    }
    NUM_TOP_WORDS = 10
    topic_keywords_dict = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[:-NUM_TOP_WORDS - 1:-1]
        topic_keywords_dict[TOPIC_LABELS[topic_idx]] = [vocab_lda[i] for i in top_indices]

    # Cell 40
    bar.progress(75, text="Assigning dominant topics…")
    document_topic_matrix = lda_model.transform(count_matrix)
    dominant_topic_indices = document_topic_matrix.argmax(axis=1)
    df['dominant_topic'] = [TOPIC_LABELS.get(i, f'Topic {i+1}') for i in dominant_topic_indices]

    # Cell 50
    bar.progress(78, text="Running VADER sentiment analysis…")
    sia = SentimentIntensityAnalyzer()

    def get_vader_sentiment(text):
        if not isinstance(text, str): return 0.0, 'neutral'
        score = sia.polarity_scores(text)['compound']
        if score >= 0.05: label = 'positive'
        elif score <= -0.05: label = 'negative'
        else: label = 'neutral'
        return round(score, 4), label

    df[['sentiment_score', 'sentiment_label']] = df['text'].apply(lambda x: pd.Series(get_vader_sentiment(x)))

    # Cell 51
    bar.progress(82, text="Computing VADER classes…")
    sia2 = SentimentIntensityAnalyzer()
    def get_vader_score(text): return sia2.polarity_scores(str(text))['compound']
    def classify_vader_sentiment(score):
        if score >= 0.05: return 'Positive'
        elif score <= -0.05: return 'Negative'
        else: return 'Neutral'
    df['vader_score'] = df['text'].apply(get_vader_score)
    df['vader_class'] = df['vader_score'].apply(classify_vader_sentiment)

    # Cell 52
    bar.progress(85, text="Running TextBlob analysis…")
    def get_sentiment_metrics(text):
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        objectivity = 1 - subjectivity
        sentiment = 'Positive' if polarity > 0.05 else ('Negative' if polarity < -0.05 else 'Neutral')
        return pd.Series([polarity, subjectivity, objectivity, sentiment])
    df[['tb_polarity', 'tb_subjectivity', 'tb_objectivity', 'tb_class']] = df['text'].apply(get_sentiment_metrics)

    # Cell 54
    ANGLE_KEYWORDS = {
        'i. Military Operations/Strategy': ['missile','drone','airstrike','strike','attack','military','bomb','weapon','soldier','troops','navy','rocket','fighter','warplane','artillery','radar','nuclear','defence','defense'],
        'ii. Geopolitical Tensions': ['tension','alliance','sanction','diplomatic','nato','russia','china','africa','asia','hezbollah','houthi','proxy','region','gulf','geopolit','superpower','influence','coalition'],
        'iii. Economic Impact': ['oil','gas','price','market','energy','hormuz','strait','supply','inflation','trade','sanction','economic','recession','fuel','barrel','opec','economy','financial','cost'],
        'iv. Media Narratives & Propaganda': ['propaganda','fake','bias','misinformation','disinformation','media','narrative','misleading','claim','censorship','manipulation','distort','report','coverage','framing'],
        'v. Support for the War': ['support','oppose','protest','ally','condemn','backing','anti-war','pro-war','public opinion','demonstration','rally','solidarity','resistance','stand with','against war']
    }

    def filter_by_angle(dataframe, keywords):
        pattern = '|'.join(keywords)
        return dataframe[dataframe['text'].str.lower().str.contains(pattern, na=False)]

    # Cell 56
    bar.progress(88, text="Computing sentiment per conflict angle…")
    angle_results = {}
    for angle, keywords in ANGLE_KEYWORDS.items():
        subset = filter_by_angle(df, keywords)
        if len(subset) == 0: continue
        summary = subset.groupby('source').agg(avg_score=('sentiment_score','mean'), count=('sentiment_score','count')).round(4).sort_values('avg_score', ascending=False)
        angle_results[angle] = summary

    # Cell 59
    bar.progress(92, text="Aggregating results…")
    avg_by_source = df.groupby('source').agg(avg_score=('sentiment_score','mean'), total_articles=('sentiment_score','count')).round(4).sort_values('avg_score', ascending=False)
    avg_by_platform = df.groupby('platform').agg(avg_score=('sentiment_score','mean'), total_articles=('sentiment_score','count')).round(4).sort_values('avg_score', ascending=False)

    # Cell 61
    dist = df.groupby(['source','sentiment_label']).size().unstack(fill_value=0)
    dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100

    # Cell 65
    heatmap_data = {}
    for angle, keywords in ANGLE_KEYWORDS.items():
        subset = filter_by_angle(df, keywords)
        if len(subset) > 0:
            heatmap_data[angle] = subset.groupby('source')['sentiment_score'].mean()
    heatmap_df = pd.DataFrame(heatmap_data).T

    # Cell 69
    angle_avg_scores = {}
    for angle, keywords in ANGLE_KEYWORDS.items():
        subset = filter_by_angle(df, keywords)
        if len(subset) > 0:
            angle_avg_scores[angle] = subset.groupby('source')['sentiment_score'].mean()
    angle_score_df = pd.DataFrame(angle_avg_scores).T

    bar.progress(100, text="✅ Pipeline complete!")
    st.session_state.update({
        'df': df, 'lda_model': lda_model, 'count_matrix': count_matrix,
        'vocab_lda': vocab_lda, 'document_topic_matrix': document_topic_matrix,
        'word_freq': word_freq, 'angle_results': angle_results,
        'avg_by_source': avg_by_source, 'avg_by_platform': avg_by_platform,
        'dist_pct': dist_pct, 'heatmap_df': heatmap_df,
        'angle_score_df': angle_score_df, 'topic_keywords_dict': topic_keywords_dict,
        'TOPIC_LABELS': TOPIC_LABELS, 'ANGLE_KEYWORDS': ANGLE_KEYWORDS,
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
        <code>combined_dataset.csv</code> file.
        .</div>""", unsafe_allow_html=True)
        
    f = st.file_uploader(
        "Select your CSV file:",
        type=["csv"],
        label_visibility="visible"
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
        <em>🏠 Project Overview</em> and upload your CSV first.</div>""", unsafe_allow_html=True)
        upload_widget(compact=True)
        return False
    return True

def pc():
    return dict(plot_bgcolor="#161b22", paper_bgcolor="#161b22", font_color="#e6edf3")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-size:0.8rem;font-weight:700;color:#e6edf3;margin:12px 0 2px 0;"
        "line-height:1.5;'>🌐 US / Israel–Iran War<br>Sentiment Analysis</p>"
        "<hr style='border-color:#30363d;margin:8px 0 12px 0;'>",
        unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Project Overview",
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
          <div>🐍 NLTK · VADER · TextBlob</div><div>📊 sklearn LDA</div><div>📈 Plotly</div>
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
      <div class="hero-meta"><strong>Method:</strong> LDA (sklearn) + VADER + TextBlob · <strong>Pipeline:</strong> Runs automatically from your uploaded CSV</div>
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
# PAGE: DATA COLLECTION  — Cell 10: bar + donut
# ═════════════════════════════════════════════════════════════════════════════
def page_data_collection():
    import plotly.graph_objects as go
    import pandas as pd
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
        rows.append({'Outlet / Platform':source,'Type':platform,'Articles / Posts':len(sub),'Date Range':dr,'Collection Method':meta['method'],'Filtering Criteria':meta['filter']})
    df_sum = pd.DataFrame(rows).sort_values('Articles / Posts',ascending=False).reset_index(drop=True)
    def hl(row):
        return ["background-color:rgba(139,92,246,0.12)"]*len(row) if row["Type"]=="Social Media" else ["background-color:rgba(59,130,246,0.06)"]*len(row)
    st.dataframe(df_sum.style.apply(hl,axis=1), use_container_width=True, hide_index=True)
    st.markdown("---")

    # ── CELL 10 — Articles by source (bar) + News vs Social (donut) ──────
    st.markdown("#### 📊 Cell 10 — Articles / Posts Collected by Source")
    col1, col2 = st.columns(2)
    sc = df['source'].value_counts()

    with col1:
        if 'platform' in df.columns:
            pm = df.drop_duplicates('source').set_index('source')['platform'].to_dict()
            bcolors = ['#2196F3' if pm.get(s,'')=='News' else '#FF5722' for s in sc.index]
        else:
            bcolors = ['#2196F3'] * len(sc)
        fig_bar = go.Figure(go.Bar(
            x=sc.index.tolist(), y=sc.values.tolist(),
            marker_color=bcolors, marker_line_color='white', marker_line_width=0.8,
            text=sc.values.tolist(), textposition='outside', textfont_color='#8b949e'
        ))
        fig_bar.update_layout(**pc(), title='Articles / Posts Collected by Source',
            xaxis=dict(tickangle=-30, gridcolor='#30363d'),
            yaxis=dict(gridcolor='#30363d', title='Count'),
            height=420, margin=dict(t=50,b=80),
            legend=dict(font_color='#e6edf3'))
        # add legend manually
        fig_bar.add_trace(go.Bar(x=[None],y=[None],marker_color='#2196F3',name='News',showlegend=True))
        fig_bar.add_trace(go.Bar(x=[None],y=[None],marker_color='#FF5722',name='Social Media',showlegend=True))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        if 'platform' in df.columns:
            pcounts = df['platform'].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=pcounts.index.tolist(), values=pcounts.values.tolist(),
                hole=0.5, marker_colors=['#2196F3','#FF5722'],
                textinfo='label+percent+value', textfont_color='#e6edf3'
            ))
            fig_pie.update_layout(**pc(), title='News vs Social Media Split',
                legend=dict(font_color='#e6edf3'), height=420)
            st.plotly_chart(fig_pie, use_container_width=True)

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
# PAGE: PRE-PROCESSING — Cell 27 (word freq bar), Cell 28 (word cloud scatter)
# ═════════════════════════════════════════════════════════════════════════════
def page_preprocessing():
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
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

    # ── CELL 27 — Top 20 Most Frequent Words (Plotly bar) ────────────────
    st.markdown("#### 📊 Cell 27 — Top 20 Most Frequent Words — Cleaned Corpus")
    top_20 = word_freq.most_common(20)
    words_top, counts_top = zip(*top_20)
    fig27 = go.Figure(go.Bar(
        x=list(words_top), y=list(counts_top),
        marker_color='steelblue', marker_line_color='navy', marker_line_width=1,
        text=list(counts_top), textposition='outside', textfont_color='#8b949e'
    ))
    fig27.update_layout(**pc(), title='Top 20 Most Frequent Words — Cleaned Corpus',
        xaxis=dict(tickangle=-45, gridcolor='#30363d', title='Word'),
        yaxis=dict(gridcolor='#30363d', title='Frequency'),
        height=440, margin=dict(t=50, b=80))
    st.plotly_chart(fig27, use_container_width=True)

    # ── CELL 28 — Word Cloud rendered as Plotly bubble/scatter ───────────
    st.markdown("#### ☁️ Cell 28 — Word Cloud — US/Israel–Iran War Corpus")
    top_100 = word_freq.most_common(100)
    wc_words = [w for w,_ in top_100]
    wc_counts = [c for _,c in top_100]
    max_c, min_c = max(wc_counts), min(wc_counts)
    np.random.seed(42)
    x_pos = np.random.uniform(10, 90, len(wc_words))
    y_pos = np.random.uniform(10, 90, len(wc_words))
    sizes = [10 + 70 * ((c - min_c) / (max_c - min_c + 1)) ** 0.5 for c in wc_counts]
    palette = px.colors.sequential.ice
    color_idx = [int((c - min_c)/(max_c - min_c + 1) * (len(palette)-1)) for c in wc_counts]
    marker_colors = [palette[i] for i in color_idx]
    fig28 = go.Figure(go.Scatter(
        x=x_pos.tolist(), y=y_pos.tolist(), mode='text',
        text=wc_words,
        textfont=dict(size=sizes, color=marker_colors),
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))
    fig28.update_layout(**pc(),
        title='Word Cloud — US/Israel–Iran War Corpus',
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False, range=[0, 100]),
        height=550, margin=dict(t=50, l=20, r=20, b=20))
    st.plotly_chart(fig28, use_container_width=True)

    if 'text_length' in df.columns:
        st.markdown("---")
        st.markdown("#### 📏 Text Length Distribution by Source")
        fig_b = px.box(df, x='source', y='text_length', color='source',
                       title='Text Length (chars) per Source',
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig_b.update_layout(**pc(), xaxis=dict(tickangle=-30, gridcolor='#30363d'),
                            yaxis=dict(gridcolor='#30363d'), showlegend=False, height=400)
        st.plotly_chart(fig_b, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TOPIC MODELING
# Cell 41 (doc-topic heatmap), 42 (topic bar), 44 (source×topic heatmap),
# Cell 46 (7 word clouds as Plotly text scatter), 47 (stacked bar)
# ═════════════════════════════════════════════════════════════════════════════
def page_topic_modeling():
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    import pandas as pd
    if not need_data(): return

    df            = st.session_state['df']
    lda_model     = st.session_state['lda_model']
    doc_topic_mat = st.session_state['document_topic_matrix']
    kw_dict       = st.session_state['topic_keywords_dict']
    TOPIC_LABELS  = st.session_state['TOPIC_LABELS']
    vocab_lda     = st.session_state['vocab_lda']
    NUM_TOPICS = 7

    ICONS  = {0:"💣",1:"🩺",2:"🤝",3:"🌍",4:"💰",5:"🗺️",6:"📺"}
    COLORS = {0:"#ef4444",1:"#f97316",2:"#3b82f6",3:"#8b5cf6",4:"#22c55e",5:"#e8a838",6:"#ec4899"}
    COLOR_LIST = [COLORS[i] for i in range(NUM_TOPICS)]

    st.markdown('<div class="section-header">🧠 Topic Modeling (LDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Phase 3 · 7 topics discovered from your dataset</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Topics (k)","7"); c2.metric("min_df","5"); c3.metric("max_df","0.90"); c4.metric("random_state","122")
    st.markdown("---")

    # Keyword pills
    st.markdown("#### 🏷️ Topics & Keywords")
    ct, cb2 = st.columns(4), st.columns(3)
    for i in range(7):
        kws   = kw_dict.get(TOPIC_LABELS[i], [])
        color = COLORS[i]
        pills = " ".join([f"<span style='background:#1c2230;border:1px solid {color}40;color:{color};padding:2px 8px;border-radius:99px;font-size:0.72rem;display:inline-block;margin:2px;'>{k}</span>" for k in kws[:8]])
        with (ct+cb2)[i]:
            st.markdown(f"""<div style='background:#161b22;border:1px solid {color}40;border-top:3px solid {color};border-radius:10px;padding:14px;margin-bottom:8px;'>
            <div style='font-size:1.3rem;margin-bottom:4px;'>{ICONS[i]}</div>
            <div style='color:#e6edf3;font-weight:600;font-size:0.85rem;margin-bottom:8px;'>Topic {i+1}: {TOPIC_LABELS[i]}</div>
            <div style='line-height:2;'>{pills}</div></div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ── CELL 41 — Document–Topic Heatmap (50 docs) ───────────────────────
    st.markdown("#### 🔥 Cell 41 — Document–Topic Distribution Heatmap (Sample of 50 Documents)")
    np.random.seed(0)
    sample_idx = np.random.choice(len(doc_topic_mat), size=min(50, len(doc_topic_mat)), replace=False)
    sample_matrix = doc_topic_mat[sample_idx]
    short_labels = [f'T{i+1}: {TOPIC_LABELS[i][:10]}' for i in range(NUM_TOPICS)]
    doc_labels   = [f'Doc {i+1}' for i in range(len(sample_idx))]
    df_dtm = pd.DataFrame(sample_matrix.round(3), index=doc_labels, columns=short_labels)
    fig41 = go.Figure(go.Heatmap(
        z=df_dtm.values, x=df_dtm.columns.tolist(), y=df_dtm.index.tolist(),
        colorscale='YlOrRd', text=df_dtm.values.round(2), texttemplate='%{text}',
        textfont_size=8,
        colorbar=dict(title='Topic Prob.', tickfont_color='#e6edf3', title_font_color='#e6edf3')
    ))
    fig41.update_layout(**pc(), title='Document–Topic Distribution Heatmap (Sample of 50 Documents)',
        xaxis=dict(tickangle=-20, tickfont_size=10), height=700, margin=dict(t=50,b=80))
    st.plotly_chart(fig41, use_container_width=True)
    st.markdown("""<div class="callout">Each row sums to 1.0. Dominant topic = argmax per row.</div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ── CELL 42 — Topic Distribution Horizontal Bar ───────────────────────
    st.markdown("#### 📊 Cell 42 — Topic Distribution Across All Sources")
    tc = df['dominant_topic'].value_counts()
    col_list = [COLORS.get(list(TOPIC_LABELS.values()).index(t), '#8b949e') if t in TOPIC_LABELS.values() else '#8b949e' for t in tc.index]
    fig42 = go.Figure(go.Bar(
        x=tc.values.tolist(), y=tc.index.tolist(), orientation='h',
        marker_color=col_list, text=tc.values.tolist(),
        textposition='outside', textfont_color='#8b949e'
    ))
    fig42.update_layout(**pc(), title='Topic Distribution Across All Sources',
        xaxis=dict(gridcolor='#30363d', title='Number of Documents'),
        yaxis=dict(gridcolor='#30363d'), height=420, margin=dict(t=50,r=80))
    st.plotly_chart(fig42, use_container_width=True)
    st.markdown("---")

    # ── CELL 44 — Topic Focus by Source Heatmap ───────────────────────────
    st.markdown("#### 🗺️ Cell 44 — Topic Focus by Source (% of Articles)")
    tbs = df.groupby(['source','dominant_topic']).size().unstack(fill_value=0)
    tbs_pct = tbs.div(tbs.sum(axis=1), axis=0) * 100
    fig44 = go.Figure(go.Heatmap(
        z=tbs_pct.values, x=tbs_pct.columns.tolist(), y=tbs_pct.index.tolist(),
        colorscale='Blues', text=tbs_pct.values.round(1), texttemplate='%{text}%',
        textfont_size=10,
        colorbar=dict(title='% of Source', tickfont_color='#e6edf3', title_font_color='#e6edf3')
    ))
    fig44.update_layout(**pc(), title='Topic Focus by Source (% of Articles)',
        xaxis=dict(tickangle=-30), height=450, margin=dict(t=50,b=100))
    st.plotly_chart(fig44, use_container_width=True)
    st.markdown("---")
    
    st.markdown("#### ☁️ Cell 46 — Word Clouds per Topic")
    NUM_TOP_WORDS = 15
    cols46 = st.columns(2)
    for i, topic in enumerate(lda_model.components_):
        top_word_indices = topic.argsort()[:-NUM_TOP_WORDS - 1:-1]
        top_words  = [vocab_lda[idx] for idx in top_word_indices]
        top_weights = [topic[idx] for idx in top_word_indices]
        w_min, w_max = min(top_weights), max(top_weights)
        sizes = [12 + 38 * ((w - w_min) / (w_max - w_min + 1e-9)) ** 0.6 for w in top_weights]
        np.random.seed(i * 13 + 5)
        x_pos = np.random.uniform(15, 85, len(top_words))
        y_pos = np.random.uniform(15, 85, len(top_words))
        color = COLORS[i]
        alphas = [0.6 + 0.4 * (w - w_min)/(w_max - w_min + 1e-9) for w in top_weights]
        fig_wc = go.Figure(go.Scatter(
            x=x_pos.tolist(), y=y_pos.tolist(), mode='text',
            text=top_words,
            textfont=dict(size=sizes, color=[f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},{a:.2f})' for a in alphas]),
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        fig_wc.update_layout(
            plot_bgcolor='#0d1117', paper_bgcolor='#161b22', font_color='#e6edf3',
            title=dict(text=f'Topic {i+1}: {TOPIC_LABELS.get(i,"")}', font_color='#e6edf3', font_size=13),
            xaxis=dict(visible=False, range=[0, 100]),
            yaxis=dict(visible=False, range=[0, 100]),
            height=380, margin=dict(t=45, l=20, r=20, b=20)
        )
        with cols46[i % 2]:
            st.plotly_chart(fig_wc, use_container_width=True)
            
    st.markdown("---")

    # ── CELL 47 — Topic Distribution by Source Stacked Bar ───────────────
    st.markdown("#### 📊 Cell 47 — Topic Distribution by Source (%)")
    tbs2 = df.groupby(['source','dominant_topic']).size().unstack(fill_value=0)
    tbs_pct2 = tbs2.div(tbs2.sum(axis=1), axis=0) * 100
    fig47 = go.Figure()
    for j, col in enumerate(tbs_pct2.columns):
        color_j = COLORS.get(list(TOPIC_LABELS.values()).index(col), '#8b949e') if col in TOPIC_LABELS.values() else '#8b949e'
        fig47.add_trace(go.Bar(
            name=col, x=tbs_pct2.index.tolist(), y=tbs_pct2[col].tolist(),
            marker_color=color_j, text=[f'{v:.1f}%' for v in tbs_pct2[col].tolist()],
            textposition='inside', textfont_color='#fff', textfont_size=9
        ))
    fig47.update_layout(**pc(), barmode='stack',
        title="Topic Distribution by Source (% of each source's content)",
        xaxis=dict(tickangle=-30, gridcolor='#30363d'),
        yaxis=dict(gridcolor='#30363d', title='Percentage (%)', ticksuffix='%'),
        legend=dict(font_color='#e6edf3', orientation='h', y=-0.3),
        height=520, margin=dict(t=50, b=140))
    st.plotly_chart(fig47, use_container_width=True)

    st.markdown("---")
    with st.expander(" LDA Code "):
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
df['dominant_topic'] = [TOPIC_LABELS.get(i, f'Topic {i+1}') for i in dominant_topic_indices]""", language="python")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT ANALYSIS
# Cell 53 (VADER pie + TextBlob pie + histogram), Cell 55 (grouped bars),
# Cell 70 (heatmap), Cell 72 (platform stacked + box), Cell 74 (angle grouped bar)
# ═════════════════════════════════════════════════════════════════════════════
def page_sentiment():
    import plotly.graph_objects as go
    import plotly.express as px
    if not need_data(): return

    df             = st.session_state['df']
    avg_by_source  = st.session_state['avg_by_source']
    avg_by_platform= st.session_state['avg_by_platform']
    angle_results  = st.session_state['angle_results']
    heatmap_df     = st.session_state['heatmap_df']
    angle_score_df = st.session_state['angle_score_df']

    st.markdown('<div class="section-header">💬 Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Phase 4 · VADER + TextBlob results from your dataset</div>', unsafe_allow_html=True)
    st.markdown("""<div class="callout"><strong>🔍 VADER applied to original text</strong> (not clean_text).
    Compound: <strong>≥ 0.05 = Positive · ≤ -0.05 = Negative · otherwise = Neutral</strong></div>""", unsafe_allow_html=True)
    st.markdown("---")

    # Per-source score badges
    sources = avg_by_source.index.tolist()
    scores  = avg_by_source['avg_score'].tolist()
    cols_b = st.columns(min(4, len(sources)))
    for i,(src,score) in enumerate(zip(sources,scores)):
        color = "#22c55e" if score>=0.05 else "#ef4444" if score<=-0.05 else "#9ca3af"
        badge = "🟢 Positive" if score>=0.05 else "🔴 Negative" if score<=-0.05 else "⚪ Neutral"
        with cols_b[i%4]:
            st.markdown(f"""<div style='background:#161b22;border:1px solid #30363d;border-left:4px solid {color};border-radius:10px;padding:12px 14px;margin-bottom:10px;'>
            <div style='color:#8b949e;font-size:0.75rem;text-transform:uppercase;font-weight:600;'>{src}</div>
            <div style='color:{color};font-size:1.6rem;font-weight:700;margin:4px 0;'>{score:+.4f}</div>
            <div style='color:#8b949e;font-size:0.78rem;'>{badge}</div></div>""", unsafe_allow_html=True)
    st.markdown("---")

    CM = {"Positive":"#4CAF50","Neutral":"#FF9800","Negative":"#F44336"}

    # ── CELL 53 — VADER pie + TextBlob pie + histogram ────────────────────
    st.markdown("#### 📊 Cell 53 — Overall Sentiment Analysis Results (VADER + TextBlob)")
    c53a, c53b, c53c = st.columns(3)

    with c53a:
        vader_dist = df['vader_class'].value_counts()
        fig_v = go.Figure(go.Pie(
            labels=vader_dist.index.tolist(), values=vader_dist.values.tolist(),
            marker_colors=[CM.get(l,'grey') for l in vader_dist.index],
            textinfo='label+percent', textfont_color='#e6edf3', hole=0
        ))
        fig_v.update_layout(**pc(), title='VADER Sentiment Distribution',
            legend=dict(font_color='#e6edf3'), height=340, margin=dict(t=50,b=10))
        st.plotly_chart(fig_v, use_container_width=True)

    with c53b:
        tb_dist = df['tb_class'].value_counts()
        fig_tb = go.Figure(go.Pie(
            labels=tb_dist.index.tolist(), values=tb_dist.values.tolist(),
            marker_colors=[CM.get(l,'grey') for l in tb_dist.index],
            textinfo='label+percent', textfont_color='#e6edf3', hole=0
        ))
        fig_tb.update_layout(**pc(), title='TextBlob Sentiment Distribution',
            legend=dict(font_color='#e6edf3'), height=340, margin=dict(t=50,b=10))
        st.plotly_chart(fig_tb, use_container_width=True)

    with c53c:
        mean_score = df['vader_score'].mean()
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=df['vader_score'].tolist(), nbinsx=30,
            marker_color='steelblue', marker_line_color='white', marker_line_width=0.5, name='Score'))
        fig_hist.add_vline(x=0, line_dash='dash', line_color='red', line_width=1.5,
            annotation_text='Neutral', annotation_font_color='red')
        fig_hist.add_vline(x=mean_score, line_dash='dash', line_color='#22c55e', line_width=1.5,
            annotation_text=f'Mean={mean_score:.2f}', annotation_font_color='#22c55e')
        fig_hist.update_layout(**pc(), title='VADER Score Distribution',
            xaxis=dict(title='Compound Score', gridcolor='#30363d'),
            yaxis=dict(title='Frequency', gridcolor='#30363d'),
            height=340, margin=dict(t=50,b=10), showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("---")

    # ── CELL 55 — VADER by Source + TextBlob by Source ────────────────────
    st.markdown("#### 📊 Cell 55 — Sentiment Distribution Across Sources (VADER & TextBlob)")
    CM2 = {"Positive":"#4CAF50","Neutral":"#FF9800","Negative":"#F44336"}
    c55a, c55b = st.columns(2)

    with c55a:
        vader_src = df.groupby(['source','vader_class']).size().unstack(fill_value=0)
        vader_src_pct = vader_src.div(vader_src.sum(axis=1), axis=0) * 100
        fig55a = go.Figure()
        for s in vader_src_pct.columns:
            fig55a.add_trace(go.Bar(
                name=s, x=vader_src_pct.index.tolist(), y=vader_src_pct[s].tolist(),
                marker_color=CM2.get(s,'grey'), text=[f'{v:.1f}%' for v in vader_src_pct[s].tolist()],
                textposition='inside', textfont_color='#fff'
            ))
        fig55a.update_layout(**pc(), barmode='group', title='VADER Sentiment by Source (%)',
            xaxis=dict(tickangle=-30, gridcolor='#30363d'),
            yaxis=dict(gridcolor='#30363d', title='Percentage (%)', ticksuffix='%'),
            legend=dict(font_color='#e6edf3', title_text='Sentiment'), height=420, margin=dict(t=50,b=80))
        st.plotly_chart(fig55a, use_container_width=True)

    with c55b:
        tb_src = df.groupby(['source','tb_class']).size().unstack(fill_value=0)
        tb_src_pct = tb_src.div(tb_src.sum(axis=1), axis=0) * 100
        fig55b = go.Figure()
        for s in tb_src_pct.columns:
            fig55b.add_trace(go.Bar(
                name=s, x=tb_src_pct.index.tolist(), y=tb_src_pct[s].tolist(),
                marker_color=CM2.get(s,'grey'), text=[f'{v:.1f}%' for v in tb_src_pct[s].tolist()],
                textposition='inside', textfont_color='#fff'
            ))
        fig55b.update_layout(**pc(), barmode='group', title='TextBlob Sentiment by Source (%)',
            xaxis=dict(tickangle=-30, gridcolor='#30363d'),
            yaxis=dict(gridcolor='#30363d', title='Percentage (%)', ticksuffix='%'),
            legend=dict(font_color='#e6edf3', title_text='Sentiment'), height=420, margin=dict(t=50,b=80))
        st.plotly_chart(fig55b, use_container_width=True)
    st.markdown("---")

    # ── CELL 70 — Sentiment Heatmap: Source × Angle ───────────────────────
    st.markdown("#### 🗺️ Cell 70 — Sentiment Heatmap: Source × Conflict Angle")
    if heatmap_df is not None and not heatmap_df.empty:
        fig70 = go.Figure(go.Heatmap(
            z=heatmap_df.values, x=heatmap_df.columns.tolist(), y=heatmap_df.index.tolist(),
            colorscale='RdYlGn', zmid=0, zmin=-1, zmax=1,
            text=heatmap_df.values.round(3), texttemplate='%{text}', textfont_size=10,
            colorbar=dict(title='Avg VADER Score', tickfont_color='#e6edf3', title_font_color='#e6edf3')
        ))
        fig70.update_layout(**pc(), title='Sentiment Heatmap: Source × Conflict Angle',
            xaxis=dict(tickangle=-30), height=420, margin=dict(t=50,b=80))
        st.plotly_chart(fig70, use_container_width=True)
    st.markdown("---")

    # ── CELL 72 — News vs Social (stacked bar + box plot) ─────────────────
    st.markdown("#### 📊 Cell 72 — News vs Social Media Sentiment Comparison")
    CM3 = {"positive":"#4CAF50","neutral":"#9E9E9E","negative":"#F44336"}
    c72a, c72b = st.columns(2)

    with c72a:
        plat_sent = df.groupby(['platform','sentiment_label']).size().unstack(fill_value=0)
        plat_pct = plat_sent.div(plat_sent.sum(axis=1), axis=0) * 100
        fig72a = go.Figure()
        for s in [c for c in ['positive','neutral','negative'] if c in plat_pct.columns]:
            fig72a.add_trace(go.Bar(
                name=s.capitalize(), x=plat_pct.index.tolist(), y=plat_pct[s].tolist(),
                marker_color=CM3[s], text=[f'{v:.1f}%' for v in plat_pct[s].tolist()],
                textposition='inside', textfont_color='#fff'
            ))
        fig72a.update_layout(**pc(), barmode='stack', title='Sentiment: News vs Social Media',
            xaxis=dict(gridcolor='#30363d'),
            yaxis=dict(gridcolor='#30363d', ticksuffix='%', title='Percentage'),
            legend=dict(font_color='#e6edf3'), height=380, margin=dict(t=50))
        st.plotly_chart(fig72a, use_container_width=True)

    with c72b:
        platforms = df['platform'].unique().tolist() if 'platform' in df.columns else []
        fig72b = go.Figure()
        for plat in platforms:
            sub = df[df['platform']==plat]['sentiment_score'].tolist()
            fig72b.add_trace(go.Box(y=sub, name=plat, marker_color='#3b82f6', line_color='#e8a838'))
        fig72b.update_layout(**pc(), title='VADER Score Distribution by Platform',
            yaxis=dict(gridcolor='#30363d', title='VADER Compound Score'),
            xaxis=dict(gridcolor='#30363d'), height=380, margin=dict(t=50))
        st.plotly_chart(fig72b, use_container_width=True)
    st.markdown("---")

    # ── CELL 74 — Avg Sentiment per Source by Angle ───────────────────────
    st.markdown("#### 📊 Cell 74 — Average Sentiment per Source — by Conflict Angle")
    if angle_score_df is not None and not angle_score_df.empty:
        fig74 = go.Figure()
        for j, src in enumerate(angle_score_df.columns):
            fig74.add_trace(go.Bar(
                name=src, x=angle_score_df.index.tolist(), y=angle_score_df[src].tolist(),
                marker_color=px.colors.qualitative.Set2[j % 8]
            ))
        fig74.add_hline(y=0, line_dash='dash', line_color='#8b949e')
        fig74.update_layout(**pc(), barmode='group',
            title='Average Sentiment per Source — by Conflict Angle',
            yaxis=dict(gridcolor='#30363d', title='Avg VADER Score', range=[-1,1]),
            xaxis=dict(gridcolor='#30363d', tickangle=-20),
            legend=dict(font_color='#e6edf3', orientation='h', y=-0.3),
            height=520, margin=dict(t=50,b=140))
        st.plotly_chart(fig74, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 Sentiment per Angle & Source (Tables)")
    for angle, summary in angle_results.items():
        with st.expander(f"  {angle}  (n = {int(summary['count'].sum()):,})"):
            st.dataframe(summary.reset_index(), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: REQUIRED VISUALIZATIONS
# Cell 66 (stacked bar), Cell 68 (avg VADER bar), diverging bar, heatmap, extras
# ═════════════════════════════════════════════════════════════════════════════
def page_visualizations():
    import plotly.graph_objects as go
    import plotly.express as px
    if not need_data(): return

    df            = st.session_state['df']
    avg_by_source = st.session_state['avg_by_source']
    avg_by_platform = st.session_state['avg_by_platform']
    dist_pct      = st.session_state['dist_pct']
    heatmap_df    = st.session_state['heatmap_df']
    word_freq     = st.session_state['word_freq']

    st.markdown('<div class="section-header">📊 Required Outputs & Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Phase 5 · All mandatory deliverables from the notebook</div>', unsafe_allow_html=True)
    st.markdown("""<div class="callout callout-success"><strong>✅ Mandatory Deliverables:</strong>
    (1) Avg sentiment by outlet/platform · (2) Pos/Neu/Neg stacked bar ·
    (3) Source comparison bar · (4) Sentiment heatmap (bonus)</div>""", unsafe_allow_html=True)
    st.markdown("---")

    sources = avg_by_source.index.tolist()
    scores  = avg_by_source['avg_score'].tolist()
    bcolors = ["#4CAF50" if s>=0.05 else "#F44336" if s<=-0.05 else "#9E9E9E" for s in scores]
    CM = {"positive":"#4CAF50","neutral":"#9E9E9E","negative":"#F44336"}

    # ── OUTPUT 1 — Avg Sentiment by Outlet ───────────────────────────────
    st.markdown("#### 📌 Output 1 — Average Sentiment Score by Outlet / Platform")
    fig1 = go.Figure(go.Bar(
        x=sources, y=scores, marker_color=bcolors,
        text=[f"{s:+.4f}" for s in scores], textposition='outside', textfont_color='#e6edf3'
    ))
    fig1.add_hline(y=0, line_dash='dash', line_color='#8b949e', line_width=0.8)
    fig1.add_hline(y=0.05, line_dash='dot', line_color='#4CAF50', line_width=0.8,
        annotation_text='Positive (≥0.05)', annotation_font_color='#4CAF50')
    fig1.add_hline(y=-0.05, line_dash='dot', line_color='#F44336', line_width=0.8,
        annotation_text='Negative (≤-0.05)', annotation_font_color='#F44336')
    fig1.update_layout(**pc(), title='Sentiment Score by Source — US/Israel–Iran War Coverage',
        yaxis=dict(range=[-1,1], gridcolor='#30363d', title='Average VADER Compound Score'),
        xaxis=dict(tickangle=-30, gridcolor='#30363d'), height=480, margin=dict(t=50))
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("##### By Platform")
    ps = avg_by_platform.index.tolist(); pv = avg_by_platform['avg_score'].tolist()
    fig1b = go.Figure(go.Bar(
        x=ps, y=pv, marker_color=["#4CAF50" if v>=0.05 else "#F44336" if v<=-0.05 else "#9E9E9E" for v in pv],
        text=[f"{v:+.4f}" for v in pv], textposition='outside', textfont_color='#e6edf3'
    ))
    fig1b.add_hline(y=0, line_dash='dash', line_color='#8b949e')
    fig1b.update_layout(**pc(), title='Avg VADER Score by Platform',
        yaxis=dict(gridcolor='#30363d'), height=340, margin=dict(t=50))
    st.plotly_chart(fig1b, use_container_width=True)
    st.markdown("---")

    # ── CELL 66 — Stacked Bar Pos/Neu/Neg ────────────────────────────────
    st.markdown("#### 📌 Output 2 / Cell 66 — Positive / Neutral / Negative Distribution")
    fig66 = go.Figure()
    for s in [c for c in ['positive','neutral','negative'] if c in dist_pct.columns]:
        fig66.add_trace(go.Bar(
            name=s.capitalize(), x=dist_pct.index.tolist(), y=dist_pct[s].tolist(),
            marker_color=CM[s], text=[f'{v:.1f}%' for v in dist_pct[s].tolist()],
            textposition='inside', textfont_color='#fff', textfont_size=10
        ))
    fig66.update_layout(**pc(), barmode='stack', title='Sentiment Distribution by Source (%)',
        yaxis=dict(gridcolor='#30363d', title='Percentage of Articles', ticksuffix='%'),
        xaxis=dict(tickangle=-30, gridcolor='#30363d'),
        legend=dict(font_color='#e6edf3', orientation='h', y=1.08),
        height=480, margin=dict(t=70))
    st.plotly_chart(fig66, use_container_width=True)
    st.markdown("---")

    # ── CELL 68 — Comparing Sentiment Across Sources ──────────────────────
    st.markdown("#### 📌 Output 3 / Cell 68 — Comparing Sentiment Across All Sources")
    fig68 = go.Figure(go.Bar(
        x=sources, y=scores, marker_color=bcolors,
        text=[f"{s:+.3f}" for s in scores], textposition='outside', textfont_color='#e6edf3',
        marker_line_color='white', marker_line_width=0.6
    ))
    fig68.add_hline(y=0, line_dash='dash', line_color='#8b949e', line_width=0.8)
    fig68.add_hline(y=0.05, line_dash='dot', line_color='#4CAF50', line_width=0.8)
    fig68.add_hline(y=-0.05, line_dash='dot', line_color='#F44336', line_width=0.8)
    fig68.update_layout(**pc(), title='Sentiment Score by Source — US/Israel–Iran War Coverage',
        yaxis=dict(range=[-1,1], gridcolor='#30363d', title='Avg VADER Compound Score'),
        xaxis=dict(tickangle=-30, gridcolor='#30363d'), height=480, margin=dict(t=50))
    st.plotly_chart(fig68, use_container_width=True)
    st.markdown("---")

    # ── OUTPUT 4 — Diverging Bar ──────────────────────────────────────────
    st.markdown("#### 📌 Output 4 (Bonus) — Diverging Sentiment Comparison Across All Sources")
    nv  = [dist_pct.loc[s,'negative'] if s in dist_pct.index and 'negative' in dist_pct.columns else 0 for s in sources]
    nuv = [dist_pct.loc[s,'neutral']  if s in dist_pct.index and 'neutral'  in dist_pct.columns else 0 for s in sources]
    pv  = [dist_pct.loc[s,'positive'] if s in dist_pct.index and 'positive' in dist_pct.columns else 0 for s in sources]
    fig_div = go.Figure()
    fig_div.add_trace(go.Bar(name='Negative', x=[-v for v in nv], y=sources, orientation='h',
        marker_color='#F44336', text=[f'-{v:.1f}%' for v in nv], textposition='inside', textfont_color='#fff'))
    fig_div.add_trace(go.Bar(name='Neutral',  x=nuv, y=sources, orientation='h',
        marker_color='#9E9E9E', text=[f'{v:.1f}%' for v in nuv], textposition='inside', textfont_color='#fff'))
    fig_div.add_trace(go.Bar(name='Positive', x=pv,  y=sources, orientation='h',
        marker_color='#4CAF50', text=[f'{v:.1f}%' for v in pv], textposition='inside', textfont_color='#fff'))
    fig_div.add_vline(x=0, line_color='#8b949e', line_width=1.5)
    fig_div.update_layout(**pc(), barmode='relative', title='Diverging Sentiment Comparison Across All Sources',
        xaxis=dict(gridcolor='#30363d', title='← Negative   |   Positive →', ticksuffix='%'),
        yaxis=dict(gridcolor='#30363d'), legend=dict(font_color='#e6edf3'),
        height=480, margin=dict(t=50))
    st.plotly_chart(fig_div, use_container_width=True)
    st.markdown("---")

    # ── Heatmap Source × Angle ────────────────────────────────────────────
    st.markdown("#### 📌 Output 4 (Bonus) — Sentiment Heatmap: Source × Conflict Angle")
    if heatmap_df is not None and not heatmap_df.empty:
        fig_hm = go.Figure(go.Heatmap(
            z=heatmap_df.values, x=heatmap_df.columns.tolist(), y=heatmap_df.index.tolist(),
            colorscale='RdYlGn', zmid=0, zmin=-1, zmax=1,
            text=heatmap_df.values.round(3), texttemplate='%{text}', textfont_size=11,
            colorbar=dict(title='Avg VADER Score', tickfont_color='#e6edf3', title_font_color='#e6edf3')
        ))
        fig_hm.update_layout(**pc(), title='Sentiment Heatmap: Source × Conflict Angle (Red=Negative · Green=Positive)',
            xaxis=dict(tickangle=-25), height=440, margin=dict(t=50,b=80))
        st.plotly_chart(fig_hm, use_container_width=True)
    st.markdown("---")

    # ── Additional ────────────────────────────────────────────────────────
    st.markdown("#### 📊 Additional — Topic Distribution & Word Frequencies")
    ca, cb = st.columns(2)
    with ca:
        tc = df['dominant_topic'].value_counts()
        fig_t = go.Figure(go.Bar(
            x=tc.values.tolist(), y=tc.index.tolist(), orientation='h',
            marker_color='#3b82f6', text=tc.values.tolist(),
            textposition='outside', textfont_color='#8b949e'
        ))
        fig_t.update_layout(**pc(), title='Document Count per LDA Topic',
            xaxis=dict(gridcolor='#30363d'), yaxis=dict(gridcolor='#30363d'),
            height=380, margin=dict(t=50,r=80))
        st.plotly_chart(fig_t, use_container_width=True)
    with cb:
        tw = dict(word_freq.most_common(25))
        fig_w = go.Figure(go.Treemap(
            labels=list(tw.keys()), values=list(tw.values()),
            parents=[""]*len(tw), marker_colorscale='Blues',
            textinfo='label+value', textfont_size=12
        ))
        fig_w.update_layout(**pc(), title='Top 25 Words — Post-Cleaning', height=380, margin=dict(t=50))
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
        for title, body in [
            ("📦 Dataset","Source names standardized (RT → RT News) · Short/empty records filtered (len < 30) · text_length column added"),
            ("🔧 Pre-processing","Lowercase → URL removal → @/# removal → non-letter strip → whitespace normalize → word_tokenize → stop words (NLTK + 19 custom) → POS-aware lemmatization → drop if clean ≤ 20 chars"),
            ("🧠 LDA","CountVectorizer(min_df=5, max_df=0.9) · LDA(n_components=7, random_state=122, max_iter=100) · dominant_topic = argmax of document-topic row"),
            ("💬 VADER","Applied to original text · compound ≥ 0.05 = positive · compound ≤ -0.05 = negative · otherwise = neutral"),
            ("📝 TextBlob","Applied to original text · polarity (-1→+1) · subjectivity (0=objective, 1=subjective)"),
            ("🎯 Angles","5 angles via keyword matching on original text · filter_by_angle() · angles are not mutually exclusive"),
        ]:
            st.markdown(f"""<div class="insight-card"><h4>{title}</h4><p>{body}</p></div>""", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("""<div class="insight-card"><h4>⚠️ Limitations & Caveats</h4><p>
        <strong>Social media sample size:</strong> Social media records are a small fraction. Findings are exploratory.<br><br>
        <strong>Source imbalance:</strong> Google News dominates many datasets. Short headlines bias sentiment vs. full articles.<br><br>
        <strong>VADER scope:</strong> Designed for English social media. Sarcasm and complex geopolitical language may be misclassified.<br><br>
        <strong>Keyword angles:</strong> Documents can match multiple angles. Each angle is treated independently.
        </p></div>""", unsafe_allow_html=True)

    with tabs[2]:
        most_neg = avg_by_source['avg_score'].idxmin()
        most_pos = avg_by_source['avg_score'].idxmax()
        top_topic = df['dominant_topic'].value_counts().idxmax()
        top_topic_n = int(df['dominant_topic'].value_counts().iloc[0])
        top_topic_pct = top_topic_n / len(df) * 100
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
        """, unsafe_allow_html=True)

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
