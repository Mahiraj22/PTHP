"""
Complete Sentiment Analysis Pipeline - Streamlit Application
FIXED VERSION with proper text cleaning, analysis, and comprehensive comparisons
"""

import auth
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime
import time
import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# TextBlob for Milestone 2
try:
    from textblob import TextBlob
except ImportError:
    st.error("Please install textblob: pip install textblob")

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    .winner-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #28a745;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
# ==================== AUTHENTICATION CHECK ====================
# Initialize session state
auth.init_session_state()

# Check authentication
if not auth.check_authentication():
    # User not authenticated, show login page
    auth.show_auth_page()
    st.stop()  # Stop execution here if not authenticated

# If we reach here, user is authenticated
# Add logout button in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown(f"**👤 Logged in as:** {st.session_state.username}")
    if st.button("🚪 Logout", type="secondary"):
        auth.logout()
        st.rerun()
    st.markdown("---")
    
# Initialize session state
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'm2_results' not in st.session_state:
    st.session_state.m2_results = None
if 'm3_results' not in st.session_state:
    st.session_state.m3_results = None
if 'm2_time' not in st.session_state:
    st.session_state.m2_time = None
if 'm3_time' not in st.session_state:
    st.session_state.m3_time = None

# ==================== HELPER FUNCTIONS ====================

def clean_text_with_regex(text):
    """Clean text using regex patterns - LESS AGGRESSIVE to preserve sentiment words"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove mentions (but keep the rest)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags symbol but KEEP the text (important for sentiment!)
    text = re.sub(r'#', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob - FIXED VERSION"""
    if not text or pd.isna(text) or text.strip() == "":
        return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment': 'neutral'}
    
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Use stricter thresholds for better classification
        if polarity > 0.05:
            sentiment = 'positive'
        elif polarity < -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': round(polarity, 4),
            'subjectivity': round(subjectivity, 4),
            'sentiment': sentiment
        }
    except Exception as e:
        print(f"TextBlob error: {e}")
        return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment': 'neutral'}

class TextDataset(Dataset):
    """Dataset for LLM processing"""
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] else ""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def normalize_ground_truth(label):
    """Normalize ground truth labels - ENHANCED VERSION"""
    if pd.isna(label):
        return None
    
    label_str = str(label).lower().strip()
    
    # Handle numeric labels (Twitter format: 0=negative, 2=neutral, 4=positive)
    if label_str in ['0', '0.0']:
        return 'negative'
    elif label_str in ['2', '2.0']:
        return 'neutral'
    elif label_str in ['4', '4.0']:
        return 'positive'
    
    # Handle 0-1 binary labels
    if label_str in ['1', '1.0']:
        return 'positive'
    
    # Handle text labels
    if 'neg' in label_str:
        return 'negative'
    elif 'neu' in label_str:
        return 'neutral'
    elif 'pos' in label_str:
        return 'positive'
    
    return None

def create_database_schema(db_file='analysis_results.db'):
    """Create database tables"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            cleaned_text TEXT,
            polarity REAL,
            subjectivity REAL,
            sentiment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def store_results_in_db(df, db_file='analysis_results.db'):
    """Store M2 results in database"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM analysis_results")
    conn.commit()
    
    for idx, row in df.iterrows():
        cursor.execute("""
            INSERT INTO analysis_results 
            (text, cleaned_text, polarity, subjectivity, sentiment)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(row.get('original_text', '')),
            str(row['cleaned_text']),
            row['m2_polarity'],
            row['m2_subjectivity'],
            row['m2_sentiment']
        ))
    
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM analysis_results")
    count = cursor.fetchone()[0]
    conn.close()
    
    return count

def send_email_report(recipient_email, report_content, attachment_data=None, attachment_name="report.txt"):
    """Send email with report"""
    sender_email = "mahisingh16945@gmail.com"
    sender_password = "sjbt tusb fjnv wqeo"
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"Sentiment Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    body = f"""
Sentiment Analysis Pipeline - Complete Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{report_content}

---
This is an automated report from the Sentiment Analysis Pipeline.
"""
    
    msg.attach(MIMEText(body, 'plain'))
    
    if attachment_data:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment_data)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={attachment_name}')
        msg.attach(part)
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email error: {e}")
        return False

# ==================== MAIN APP ====================

st.markdown('<div class="main-header">📊 Sentiment Analysis Pipeline Dashboard</div>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎯 Navigation")
    page = st.radio(
        "Select Stage",
        ["📁 Data Upload & Cleaning", 
         "🎯 Milestone 2: TextBlob", 
         "🤖 Milestone 3: LLM Analysis", 
         "📊 Compare Results",
         "📧 Generate & Email Report"]
    )
    
    st.markdown("---")
    st.subheader("📌 Pipeline Status")
    
    status_items = {
        "Raw Data": st.session_state.raw_data is not None,
        "Cleaned Data": st.session_state.cleaned_data is not None,
        "M2 Analysis": st.session_state.m2_results is not None,
        "M3 Analysis": st.session_state.m3_results is not None
    }
    
    for name, status in status_items.items():
        icon = "✅" if status else "⏳"
        st.text(f"{icon} {name}")

# ==================== DATA UPLOAD & CLEANING ====================
if page == "📁 Data Upload & Cleaning":
    st.header("📁 Step 1: Data Upload & Cleaning")
    
    tab1, tab2 = st.tabs(["Upload New Data", "View Loaded Data"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, encoding='latin1')
                st.success(f"✅ Loaded {len(df)} rows")
                
                st.subheader("Data Preview")
                st.dataframe(df.head(10))
                
                # Column selection
                st.subheader("Select Columns")
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    text_col = st.selectbox("Text column", text_columns)
                with col2:
                    has_labels = st.checkbox("Dataset has ground truth labels")
                    label_col = None
                    if has_labels:
                        label_col = st.selectbox("Label column", df.columns.tolist())
                
                # Sample size
                sample_size = st.number_input(
                    "Number of rows to process (0 = all)", 
                    min_value=0, 
                    max_value=len(df), 
                    value=min(1000, len(df))
                )
                
                if st.button("🧹 Clean Data", type="primary"):
                    with st.spinner("Cleaning data..."):
                        progress_bar = st.progress(0)
                        
                        # Sample data if needed
                        if sample_size > 0 and sample_size < len(df):
                            df_sample = df.head(sample_size).copy()
                        else:
                            df_sample = df.copy()
                        
                        progress_bar.progress(20)
                        
                        # Store original text FIRST
                        df_sample['original_text'] = df_sample[text_col].astype(str)
                        
                        # Clean text - less aggressive
                        df_sample['cleaned_text'] = df_sample[text_col].apply(clean_text_with_regex)
                        
                        # Remove completely empty texts
                        df_sample = df_sample[df_sample['cleaned_text'].str.len() > 0].reset_index(drop=True)
                        
                        progress_bar.progress(60)
                        
                        # Handle labels if exists
                        if label_col:
                            df_sample['ground_truth'] = df_sample[label_col].apply(normalize_ground_truth)
                            # Remove rows with invalid labels
                            df_sample = df_sample[df_sample['ground_truth'].notna()].reset_index(drop=True)
                        
                        progress_bar.progress(100)
                        
                        # Save to session state
                        st.session_state.raw_data = df_sample
                        st.session_state.cleaned_data = df_sample
                        
                        st.success(f"✅ Cleaned {len(df_sample)} rows!")
                        st.balloons()
                        
                        # Show preview
                        st.subheader("Cleaned Data Preview")
                        preview_cols = ['original_text', 'cleaned_text']
                        if label_col and 'ground_truth' in df_sample.columns:
                            preview_cols.append('ground_truth')
                        st.dataframe(df_sample[preview_cols].head(10))
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Valid Rows", len(df_sample))
                        with col2:
                            avg_len = df_sample['cleaned_text'].str.len().mean()
                            st.metric("Avg Text Length", f"{avg_len:.0f}")
                        with col3:
                            if 'ground_truth' in df_sample.columns:
                                label_dist = df_sample['ground_truth'].value_counts()
                                st.write("**Label Distribution:**")
                                st.write(label_dist)
                        
            except Exception as e:
                st.error(f"Error loading file: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    with tab2:
        if st.session_state.cleaned_data is not None:
            df = st.session_state.cleaned_data
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                avg_length = df['cleaned_text'].str.len().mean()
                st.metric("Avg Text Length", f"{avg_length:.0f}")
            
            st.dataframe(df.head(20))
        else:
            st.info("No data loaded yet. Please upload and clean data first.")

# ==================== MILESTONE 2: TEXTBLOB ====================
elif page == "🎯 Milestone 2: TextBlob":
    st.header("🎯 Milestone 2: Traditional NLP with TextBlob")
    
    if st.session_state.cleaned_data is None:
        st.warning("⚠️ Please upload and clean data first!")
    else:
        df = st.session_state.cleaned_data.copy()
        
        st.info("""
        **TextBlob Analysis:**
        - Polarity: -1 (negative) to +1 (positive)
        - Subjectivity: 0 (objective) to 1 (subjective)
        - Fast dictionary-based approach
        """)
        
        # Show sample of texts to be analyzed
        with st.expander("Preview texts to be analyzed"):
            st.dataframe(df[['cleaned_text']].head(10))
        
        if st.button("🚀 Run TextBlob Analysis", type="primary"):
            with st.spinner("Analyzing sentiment with TextBlob..."):
                start_time = time.time()
                progress_bar = st.progress(0)
                
                results = []
                total = len(df)
                
                for idx, row in df.iterrows():
                    result = analyze_sentiment_textblob(row['cleaned_text'])
                    results.append(result)
                    
                    if idx % max(1, total // 100) == 0:
                        progress = int((idx + 1) / total * 100)
                        progress_bar.progress(progress)
                
                progress_bar.progress(100)
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Add results to dataframe
                df['m2_polarity'] = [r['polarity'] for r in results]
                df['m2_subjectivity'] = [r['subjectivity'] for r in results]
                df['m2_sentiment'] = [r['sentiment'] for r in results]
                
                # Store timing
                st.session_state.m2_time = processing_time
                
                # Store in database
                create_database_schema()
                db_count = store_results_in_db(df)
                
                # Store results
                st.session_state.m2_results = df
                
                st.success(f"✅ TextBlob analysis complete in {processing_time:.2f}s! Stored {db_count} results.")
                st.balloons()
                
                # Show results
                st.subheader("Results Preview")
                preview_cols = ['cleaned_text', 'm2_sentiment', 'm2_polarity', 'm2_subjectivity']
                st.dataframe(df[preview_cols].head(10))
                
                # Statistics
                st.subheader("📊 Sentiment Distribution")
                col1, col2, col3, col4 = st.columns(4)
                
                sentiment_counts = df['m2_sentiment'].value_counts()
                with col1:
                    st.metric("Positive", sentiment_counts.get('positive', 0))
                with col2:
                    st.metric("Neutral", sentiment_counts.get('neutral', 0))
                with col3:
                    st.metric("Negative", sentiment_counts.get('negative', 0))
                with col4:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                
                # Visualization
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution (TextBlob)",
                    color_discrete_map={
                        'positive': 'green',
                        'neutral': 'gray',
                        'negative': 'red'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Show existing results
        if st.session_state.m2_results is not None:
            st.subheader("Current Results")
            df_results = st.session_state.m2_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Polarity", f"{df_results['m2_polarity'].mean():.3f}")
            with col2:
                st.metric("Avg Subjectivity", f"{df_results['m2_subjectivity'].mean():.3f}")
            with col3:
                if st.session_state.m2_time:
                    st.metric("Processing Time", f"{st.session_state.m2_time:.2f}s")

# ==================== MILESTONE 3: LLM ====================
elif page == "🤖 Milestone 3: LLM Analysis":
    st.header("🤖 Milestone 3: LLM-based Analysis")
    
    if st.session_state.cleaned_data is None:
        st.warning("⚠️ Please upload and clean data first!")
    else:
        df = st.session_state.cleaned_data.copy()
        
        st.info("""
        **LLM Analysis (RoBERTa):**
        - Twitter-RoBERTa-base model
        - Trained on 124M tweets
        - Higher accuracy than traditional methods
        - GPU acceleration if available
        """)
        
        # Configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            batch_size = st.select_slider("Batch Size", [8, 16, 32], value=16)
        with col2:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.metric("Device", device.upper())
        with col3:
            st.info(f"**Texts to process:** {len(df)}")
        
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        if st.button("🚀 Run LLM Analysis", type="primary"):
            with st.spinner("Loading model..."):
                try:
                    start_time = time.time()
                    
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    model.to(device)
                    model.eval()
                    
                    load_time = time.time() - start_time
                    st.success(f"✅ Model loaded in {load_time:.2f}s!")
                    
                    # Prepare dataset
                    texts = df['cleaned_text'].fillna('').astype(str).tolist()
                    dataset = TextDataset(texts, tokenizer, 128)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                    
                    # Process
                    inference_start = time.time()
                    progress_bar = st.progress(0)
                    predictions = []
                    confidences = []
                    
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(dataloader):
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                            
                            batch_preds = torch.argmax(probs, dim=-1).cpu().numpy()
                            batch_conf = torch.max(probs, dim=-1)[0].cpu().numpy()
                            
                            predictions.extend(batch_preds)
                            confidences.extend(batch_conf)
                            
                            progress = int((batch_idx + 1) / len(dataloader) * 100)
                            progress_bar.progress(progress)
                    
                    progress_bar.progress(100)
                    inference_time = time.time() - inference_start
                    total_time = time.time() - start_time
                    
                    # Store timing
                    st.session_state.m3_time = total_time
                    
                    # Map predictions (RoBERTa: 0=negative, 1=neutral, 2=positive)
                    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                    df['m3_sentiment'] = [label_map[p] for p in predictions]
                    df['m3_confidence'] = confidences
                    
                    # Merge with M2 results if available
                    if st.session_state.m2_results is not None:
                        df['m2_sentiment'] = st.session_state.m2_results['m2_sentiment']
                        df['m2_polarity'] = st.session_state.m2_results['m2_polarity']
                        df['m2_subjectivity'] = st.session_state.m2_results['m2_subjectivity']
                    
                    st.session_state.m3_results = df
                    
                    st.success(f"✅ LLM analysis complete in {total_time:.2f}s (inference: {inference_time:.2f}s)!")
                    st.balloons()
                    
                    # Show results
                    st.subheader("Results Preview")
                    preview_cols = ['cleaned_text', 'm3_sentiment', 'm3_confidence']
                    st.dataframe(df[preview_cols].head(10))
                    
                    # Statistics
                    st.subheader("📊 Sentiment Distribution")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    sentiment_counts = df['m3_sentiment'].value_counts()
                    with col1:
                        st.metric("Positive", sentiment_counts.get('positive', 0))
                    with col2:
                        st.metric("Neutral", sentiment_counts.get('neutral', 0))
                    with col3:
                        st.metric("Negative", sentiment_counts.get('negative', 0))
                    with col4:
                        st.metric("Total Time", f"{total_time:.2f}s")
                    
                    fig = px.bar(
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        labels={'x': 'Sentiment', 'y': 'Count'},
                        title="Sentiment Distribution (LLM)",
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'positive': 'green',
                            'neutral': 'gray',
                            'negative': 'red'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ==================== COMPREHENSIVE COMPARISON ====================
elif page == "📊 Compare Results":
    st.header("📊 Comprehensive Comparison: TextBlob vs RoBERTa LLM")
    
    if st.session_state.m2_results is None or st.session_state.m3_results is None:
        st.warning("⚠️ Please complete both Milestone 2 and Milestone 3 analyses first!")
    else:
        df = st.session_state.m3_results.copy()
        has_ground_truth = 'ground_truth' in df.columns and df['ground_truth'].notna().any()
        
        # ========== PERFORMANCE METRICS ==========
        st.subheader("⚡ Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("TextBlob Time", 
                     f"{st.session_state.m2_time:.2f}s" if st.session_state.m2_time else "N/A")
        with col2:
            st.metric("LLM Time", 
                     f"{st.session_state.m3_time:.2f}s" if st.session_state.m3_time else "N/A")
        with col3:
            if st.session_state.m2_time and st.session_state.m3_time:
                speedup = st.session_state.m3_time / st.session_state.m2_time
                st.metric("Speed Ratio", f"{speedup:.2f}x")
                st.caption("LLM is slower" if speedup > 1 else "LLM is faster")
        with col4:
            if st.session_state.m2_time and st.session_state.m3_time:
                m2_speed = len(df) / st.session_state.m2_time
                m3_speed = len(df) / st.session_state.m3_time
                st.metric("TextBlob", f"{m2_speed:.0f} texts/s")
                st.metric("LLM", f"{m3_speed:.0f} texts/s")
        
        st.markdown("---")
        
        # ========== ACCURACY COMPARISON ==========
        if has_ground_truth:
            st.subheader("🎯 Accuracy Comparison")
            
            # Calculate metrics
            m2_accuracy = accuracy_score(df['ground_truth'], df['m2_sentiment'])
            m3_accuracy = accuracy_score(df['ground_truth'], df['m3_sentiment'])
            
            # Precision, Recall, F1
            labels = ['negative', 'neutral', 'positive']
            m2_precision, m2_recall, m2_f1, _ = precision_recall_fscore_support(
                df['ground_truth'], df['m2_sentiment'], labels=labels, average='weighted', zero_division=0
            )
            m3_precision, m3_recall, m3_f1, _ = precision_recall_fscore_support(
                df['ground_truth'], df['m3_sentiment'], labels=labels, average='weighted', zero_division=0
            )
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📊 Accuracy")
                st.metric("TextBlob", f"{m2_accuracy*100:.2f}%")
                st.metric("LLM (RoBERTa)", f"{m3_accuracy*100:.2f}%", 
                         delta=f"{(m3_accuracy-m2_accuracy)*100:+.2f}pp")
                
                # Winner badge
                if m3_accuracy > m2_accuracy:
                    st.markdown('<div class="winner-box">🏆 LLM Wins on Accuracy!</div>', unsafe_allow_html=True)
                elif m2_accuracy > m3_accuracy:
                    st.markdown('<div class="winner-box">🏆 TextBlob Wins on Accuracy!</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🎯 Precision & Recall")
                st.metric("TextBlob Precision", f"{m2_precision*100:.2f}%")
                st.metric("LLM Precision", f"{m3_precision*100:.2f}%",
                         delta=f"{(m3_precision-m2_precision)*100:+.2f}pp")
                st.metric("TextBlob Recall", f"{m2_recall*100:.2f}%")
                st.metric("LLM Recall", f"{m3_recall*100:.2f}%",
                         delta=f"{(m3_recall-m2_recall)*100:+.2f}pp")
            
            with col3:
                st.markdown("### 📈 F1 Score")
                st.metric("TextBlob F1", f"{m2_f1*100:.2f}%")
                st.metric("LLM F1", f"{m3_f1*100:.2f}%",
                         delta=f"{(m3_f1-m2_f1)*100:+.2f}pp")
                
                # Overall improvement
                improvement = ((m3_accuracy - m2_accuracy) / m2_accuracy * 100) if m2_accuracy > 0 else 0
                st.metric("Relative Improvement", f"{improvement:+.1f}%")
            
            st.markdown("---")
            
            # ========== CONFUSION MATRICES ==========
            st.subheader("🔍 Confusion Matrices")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### TextBlob Confusion Matrix")
                cm_m2 = confusion_matrix(df['ground_truth'], df['m2_sentiment'], labels=labels)
                fig_cm2 = px.imshow(
                    cm_m2,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=labels,
                    y=labels,
                    color_continuous_scale='Blues',
                    text_auto=True,
                    aspect="auto"
                )
                fig_cm2.update_layout(height=400)
                st.plotly_chart(fig_cm2, use_container_width=True)
            
            with col2:
                st.markdown("#### LLM (RoBERTa) Confusion Matrix")
                cm_m3 = confusion_matrix(df['ground_truth'], df['m3_sentiment'], labels=labels)
                fig_cm3 = px.imshow(
                    cm_m3,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=labels,
                    y=labels,
                    color_continuous_scale='Reds',
                    text_auto=True,
                    aspect="auto"
                )
                fig_cm3.update_layout(height=400)
                st.plotly_chart(fig_cm3, use_container_width=True)
            
            st.markdown("---")
            
            # ========== PER-CLASS PERFORMANCE ==========
            st.subheader("📊 Per-Class Performance Comparison")
            
            # Get detailed metrics per class
            m2_report = classification_report(df['ground_truth'], df['m2_sentiment'], 
                                             labels=labels, output_dict=True, zero_division=0)
            m3_report = classification_report(df['ground_truth'], df['m3_sentiment'],
                                             labels=labels, output_dict=True, zero_division=0)
            
            # Create comparison dataframe
            comparison_data = []
            for label in labels:
                comparison_data.append({
                    'Sentiment': label.capitalize(),
                    'TextBlob Precision': f"{m2_report[label]['precision']*100:.1f}%",
                    'LLM Precision': f"{m3_report[label]['precision']*100:.1f}%",
                    'TextBlob Recall': f"{m2_report[label]['recall']*100:.1f}%",
                    'LLM Recall': f"{m3_report[label]['recall']*100:.1f}%",
                    'TextBlob F1': f"{m2_report[label]['f1-score']*100:.1f}%",
                    'LLM F1': f"{m3_report[label]['f1-score']*100:.1f}%",
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualize per-class F1 scores
            fig_f1 = go.Figure()
            
            for label in labels:
                fig_f1.add_trace(go.Bar(
                    name=f'{label.capitalize()} - TextBlob',
                    x=['TextBlob'],
                    y=[m2_report[label]['f1-score']],
                    marker_color='lightblue'
                ))
                fig_f1.add_trace(go.Bar(
                    name=f'{label.capitalize()} - LLM',
                    x=['LLM'],
                    y=[m3_report[label]['f1-score']],
                    marker_color='lightcoral'
                ))
            
            fig_f1.update_layout(
                title='F1 Score Comparison by Sentiment Class',
                yaxis_title='F1 Score',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_f1, use_container_width=True)
        
        else:
            st.info("ℹ️ No ground truth labels available. Showing distribution comparison only.")
        
        st.markdown("---")
        
        # ========== SENTIMENT DISTRIBUTION COMPARISON ==========
        st.subheader("📊 Sentiment Distribution Comparison")
        
        m2_dist = df['m2_sentiment'].value_counts()
        m3_dist = df['m3_sentiment'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### TextBlob Distribution")
            fig_m2 = px.pie(
                values=m2_dist.values,
                names=m2_dist.index,
                color=m2_dist.index,
                color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            )
            st.plotly_chart(fig_m2, use_container_width=True)
        
        with col2:
            st.markdown("#### LLM Distribution")
            fig_m3 = px.pie(
                values=m3_dist.values,
                names=m3_dist.index,
                color=m3_dist.index,
                color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            )
            st.plotly_chart(fig_m3, use_container_width=True)
        
        # Side-by-side bar comparison
        comparison_df = pd.DataFrame({
            'TextBlob': m2_dist,
            'LLM (RoBERTa)': m3_dist
        }).fillna(0)
        
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name='TextBlob',
            x=comparison_df.index,
            y=comparison_df['TextBlob'],
            marker_color='lightblue'
        ))
        fig_compare.add_trace(go.Bar(
            name='LLM (RoBERTa)',
            x=comparison_df.index,
            y=comparison_df['LLM (RoBERTa)'],
            marker_color='lightcoral'
        ))
        fig_compare.update_layout(
            barmode='group',
            title='Sentiment Count Comparison',
            xaxis_title='Sentiment',
            yaxis_title='Count',
            height=400
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        
        st.markdown("---")
        
        # ========== AGREEMENT ANALYSIS ==========
        st.subheader("🤝 Agreement Analysis")
        
        # Calculate agreement
        agreement = (df['m2_sentiment'] == df['m3_sentiment']).sum()
        agreement_pct = (agreement / len(df)) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", len(df))
        with col2:
            st.metric("Models Agree", f"{agreement} ({agreement_pct:.1f}%)")
        with col3:
            st.metric("Models Disagree", f"{len(df)-agreement} ({100-agreement_pct:.1f}%)")
        
        # Agreement matrix
        agreement_matrix = pd.crosstab(
            df['m2_sentiment'], 
            df['m3_sentiment'],
            rownames=['TextBlob'],
            colnames=['LLM (RoBERTa)']
        )
        
        fig_agreement = px.imshow(
            agreement_matrix.values,
            labels=dict(x="LLM Prediction", y="TextBlob Prediction", color="Count"),
            x=agreement_matrix.columns,
            y=agreement_matrix.index,
            color_continuous_scale='Greens',
            text_auto=True,
            aspect="auto"
        )
        fig_agreement.update_layout(
            title='Prediction Agreement Matrix',
            height=400
        )
        st.plotly_chart(fig_agreement, use_container_width=True)
        
        st.markdown("---")
        
        # ========== SAMPLE COMPARISONS ==========
        st.subheader("🔍 Sample Prediction Comparisons")
        
        # Show where models disagree
        disagreements = df[df['m2_sentiment'] != df['m3_sentiment']]
        
        if len(disagreements) > 0:
            st.markdown(f"**Found {len(disagreements)} disagreements. Showing samples:**")
            
            comparison_cols = ['cleaned_text', 'm2_sentiment', 'm3_sentiment']
            if has_ground_truth:
                comparison_cols.append('ground_truth')
            if 'm3_confidence' in df.columns:
                comparison_cols.append('m3_confidence')
            
            st.dataframe(disagreements[comparison_cols].head(20), use_container_width=True)
        else:
            st.success("✅ Perfect agreement between models!")
        
        # Show all predictions sample
        st.markdown("**Random Sample of All Predictions:**")
        sample_cols = ['cleaned_text', 'm2_sentiment', 'm3_sentiment']
        if has_ground_truth:
            sample_cols.append('ground_truth')
        st.dataframe(df[sample_cols].sample(min(10, len(df))), use_container_width=True)
        
        st.markdown("---")
        
        # ========== SUMMARY INSIGHTS ==========
        st.subheader("💡 Summary Insights")
        
        insights = []
        
        if has_ground_truth:
            if m3_accuracy > m2_accuracy:
                insights.append(f"✅ **LLM outperforms TextBlob** by {(m3_accuracy-m2_accuracy)*100:.2f} percentage points")
            elif m2_accuracy > m3_accuracy:
                insights.append(f"✅ **TextBlob outperforms LLM** by {(m2_accuracy-m3_accuracy)*100:.2f} percentage points")
            else:
                insights.append("✅ **Both models perform equally** on this dataset")
        
        if st.session_state.m2_time and st.session_state.m3_time:
            if st.session_state.m2_time < st.session_state.m3_time:
                speedup = st.session_state.m3_time / st.session_state.m2_time
                insights.append(f"⚡ **TextBlob is {speedup:.1f}x faster** than LLM")
            else:
                speedup = st.session_state.m2_time / st.session_state.m3_time
                insights.append(f"⚡ **LLM is {speedup:.1f}x faster** than TextBlob")
        
        insights.append(f"🤝 **Models agree on {agreement_pct:.1f}%** of predictions")
        
        if len(disagreements) > 0:
            insights.append(f"🔍 **{len(disagreements)} disagreements** found - worth investigating")
        
        for insight in insights:
            st.markdown(insight)

# ==================== EMAIL REPORT ====================
elif page == "📧 Generate & Email Report":
    st.header("📧 Generate & Email Report")
    
    if st.session_state.m2_results is None:
        st.warning("⚠️ Please complete at least Milestone 2 analysis first!")
    else:
        df = st.session_state.m2_results if st.session_state.m3_results is None else st.session_state.m3_results
        has_ground_truth = 'ground_truth' in df.columns and df['ground_truth'].notna().any()
        
        # Generate comprehensive report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("SENTIMENT ANALYSIS PIPELINE - COMPREHENSIVE COMPARISON REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Documents Analyzed: {len(df)}")
        
        # ========== PERFORMANCE METRICS ==========
        report_lines.append(f"\n{'='*80}")
        report_lines.append("PERFORMANCE METRICS")
        report_lines.append("="*80)
        
        if st.session_state.m2_time:
            report_lines.append(f"TextBlob Processing Time: {st.session_state.m2_time:.2f} seconds")
            report_lines.append(f"TextBlob Speed: {len(df)/st.session_state.m2_time:.1f} texts/second")
        
        if st.session_state.m3_time:
            report_lines.append(f"LLM Processing Time: {st.session_state.m3_time:.2f} seconds")
            report_lines.append(f"LLM Speed: {len(df)/st.session_state.m3_time:.1f} texts/second")
        
        if st.session_state.m2_time and st.session_state.m3_time:
            speedup = st.session_state.m3_time / st.session_state.m2_time
            report_lines.append(f"Speed Ratio (LLM/TextBlob): {speedup:.2f}x")
        
        # ========== MILESTONE 2 RESULTS ==========
        report_lines.append(f"\n{'='*80}")
        report_lines.append("MILESTONE 2: TEXTBLOB ANALYSIS")
        report_lines.append("="*80)
        
        m2_dist = df['m2_sentiment'].value_counts()
        for sentiment, count in m2_dist.items():
            percentage = (count / len(df)) * 100
            report_lines.append(f"{sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        report_lines.append(f"\nAverage Polarity: {df['m2_polarity'].mean():.4f}")
        report_lines.append(f"Average Subjectivity: {df['m2_subjectivity'].mean():.4f}")
        report_lines.append(f"Polarity Std Dev: {df['m2_polarity'].std():.4f}")
        
        # ========== MILESTONE 3 RESULTS ==========
        if st.session_state.m3_results is not None and 'm3_sentiment' in df.columns:
            report_lines.append(f"\n{'='*80}")
            report_lines.append("MILESTONE 3: LLM (RoBERTa) ANALYSIS")
            report_lines.append("="*80)
            
            m3_dist = df['m3_sentiment'].value_counts()
            for sentiment, count in m3_dist.items():
                percentage = (count / len(df)) * 100
                report_lines.append(f"{sentiment.capitalize()}: {count} ({percentage:.1f}%)")
            
            report_lines.append(f"\nAverage Confidence: {df['m3_confidence'].mean():.4f}")
            report_lines.append(f"Confidence Std Dev: {df['m3_confidence'].std():.4f}")
            report_lines.append(f"Min Confidence: {df['m3_confidence'].min():.4f}")
            report_lines.append(f"Max Confidence: {df['m3_confidence'].max():.4f}")
            
            # ========== AGREEMENT ANALYSIS ==========
            report_lines.append(f"\n{'='*80}")
            report_lines.append("AGREEMENT ANALYSIS")
            report_lines.append("="*80)
            
            agreement = (df['m2_sentiment'] == df['m3_sentiment']).sum()
            agreement_pct = (agreement / len(df)) * 100
            report_lines.append(f"Models Agree: {agreement} predictions ({agreement_pct:.1f}%)")
            report_lines.append(f"Models Disagree: {len(df)-agreement} predictions ({100-agreement_pct:.1f}%)")
            
            # ========== ACCURACY COMPARISON ==========
            if has_ground_truth:
                report_lines.append(f"\n{'='*80}")
                report_lines.append("ACCURACY COMPARISON (WITH GROUND TRUTH)")
                report_lines.append("="*80)
                
                from sklearn.metrics import precision_recall_fscore_support
                
                m2_acc = accuracy_score(df['ground_truth'], df['m2_sentiment'])
                m3_acc = accuracy_score(df['ground_truth'], df['m3_sentiment'])
                
                report_lines.append(f"\nOverall Accuracy:")
                report_lines.append(f"  TextBlob: {m2_acc*100:.2f}%")
                report_lines.append(f"  LLM (RoBERTa): {m3_acc*100:.2f}%")
                report_lines.append(f"  Improvement: {(m3_acc-m2_acc)*100:+.2f} percentage points")
                
                if m3_acc > 0:
                    relative_improvement = ((m3_acc - m2_acc) / m2_acc * 100) if m2_acc > 0 else 0
                    report_lines.append(f"  Relative Improvement: {relative_improvement:+.1f}%")
                
                # Per-class metrics
                labels = ['negative', 'neutral', 'positive']
                
                m2_precision, m2_recall, m2_f1, _ = precision_recall_fscore_support(
                    df['ground_truth'], df['m2_sentiment'], labels=labels, average='weighted', zero_division=0
                )
                m3_precision, m3_recall, m3_f1, _ = precision_recall_fscore_support(
                    df['ground_truth'], df['m3_sentiment'], labels=labels, average='weighted', zero_division=0
                )
                
                report_lines.append(f"\nWeighted Average Metrics:")
                report_lines.append(f"  TextBlob - Precision: {m2_precision*100:.2f}%, Recall: {m2_recall*100:.2f}%, F1: {m2_f1*100:.2f}%")
                report_lines.append(f"  LLM - Precision: {m3_precision*100:.2f}%, Recall: {m3_recall*100:.2f}%, F1: {m3_f1*100:.2f}%")
                
                # Detailed per-class report
                m2_report = classification_report(df['ground_truth'], df['m2_sentiment'], 
                                                 labels=labels, zero_division=0, output_dict=True)
                m3_report = classification_report(df['ground_truth'], df['m3_sentiment'],
                                                 labels=labels, zero_division=0, output_dict=True)
                
                report_lines.append(f"\nPer-Class Performance:")
                for label in labels:
                    report_lines.append(f"\n  {label.upper()}:")
                    report_lines.append(f"    TextBlob - P: {m2_report[label]['precision']*100:.1f}%, R: {m2_report[label]['recall']*100:.1f}%, F1: {m2_report[label]['f1-score']*100:.1f}%")
                    report_lines.append(f"    LLM - P: {m3_report[label]['precision']*100:.1f}%, R: {m3_report[label]['recall']*100:.1f}%, F1: {m3_report[label]['f1-score']*100:.1f}%")
                
                # Winner declaration
                report_lines.append(f"\n{'='*80}")
                report_lines.append("CONCLUSION")
                report_lines.append("="*80)
                
                if m3_acc > m2_acc:
                    report_lines.append(f"🏆 WINNER: LLM (RoBERTa) with {(m3_acc-m2_acc)*100:.2f}pp higher accuracy")
                elif m2_acc > m3_acc:
                    report_lines.append(f"🏆 WINNER: TextBlob with {(m2_acc-m3_acc)*100:.2f}pp higher accuracy")
                else:
                    report_lines.append("🏆 TIE: Both models perform equally")
                
                if st.session_state.m2_time and st.session_state.m3_time:
                    if st.session_state.m2_time < st.session_state.m3_time:
                        report_lines.append(f"⚡ TextBlob is {st.session_state.m3_time/st.session_state.m2_time:.1f}x faster")
                    else:
                        report_lines.append(f"⚡ LLM is {st.session_state.m2_time/st.session_state.m3_time:.1f}x faster")
        
        report_lines.append(f"\n{'='*80}")
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        report_content = "\n".join(report_lines)
        
        # Display report
        st.subheader("📄 Report Preview")
        st.text_area("Report Content", report_content, height=500)
        
        # Email configuration
        st.subheader("📧 Email Configuration")
        
        recipient_email = st.text_input(
            "Recipient Email Address",
            placeholder="recipient@example.com"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Download Report", type="secondary"):
                st.download_button(
                    "Click to Download",
                    report_content,
                    "sentiment_analysis_comprehensive_report.txt",
                    "text/plain"
                )
        
        with col2:
            if st.button("📧 Send Email", type="primary"):
                if not recipient_email:
                    st.error("Please enter a recipient email address!")
                elif not re.match(r"[^@]+@[^@]+\.[^@]+", recipient_email):
                    st.error("Please enter a valid email address!")
                else:
                    with st.spinner("Sending email..."):
                        success = send_email_report(
                            recipient_email,
                            report_content,
                            report_content.encode(),
                            "sentiment_analysis_report.txt"
                        )
                        
                        if success:
                            st.success(f"✅ Email sent successfully to {recipient_email}!")
                            st.balloons()
                        else:
                            st.error("Failed to send email. Please check your connection.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    📊 Sentiment Analysis Pipeline | Comprehensive Comparison Dashboard<br>
    TextBlob vs RoBERTa LLM | Performance, Accuracy & Distribution Analysis
    </div>
    """,
    unsafe_allow_html=True
)