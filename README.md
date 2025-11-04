# 📊 Parallel Text Handling Processor

A Streamlit app comparing TextBlob vs RoBERTa LLM for sentiment analysis with authentication, data processing, and comprehensive reporting.

## 🌟 Features

- **Dual Analysis**: TextBlob (traditional) vs RoBERTa LLM (modern)
- **Complete Pipeline**: Upload → Clean → Analyze → Compare → Report
- **Authentication**: Secure login/registration with password hashing
- **Metrics**: Accuracy, precision, recall, F1-score, confusion matrices
- **Export**: Download reports or email them directly

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA (optional, for GPU acceleration)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd sentiment-analysis-pipeline
```

### Step 2: Install Dependencies

```bash
pip install streamlit pandas numpy torch transformers textblob
pip install plotly scikit-learn seaborn matplotlib sqlite3
```

### Step 3: Download NLTK Data (for TextBlob)

```python
python -c "import nltk; nltk.download('brown'); nltk.download('punkt')"
```

## 📦 Project Structure

```
sentiment-analysis-pipeline/
│
├── app.py                          # Main application file
├── auth.py                         # Authentication module
├── users.db                        # User database (auto-created)
├── analysis_results.db             # Analysis results database (auto-created)
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## 🎯 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Workflow

#### 1. **Authentication**

- Create a new account or login with existing credentials
- Password requirements: minimum 6 characters
- Valid email format required

#### 2. **Data Upload & Cleaning**

- Navigate to "📁 Data Upload & Cleaning"
- Upload a CSV file with text data
- Select the text column
- Optionally select ground truth label column
- Choose sample size (0 = all rows)
- Click "🧹 Clean Data"

**Supported Label Formats:**

- Numeric: 0 (negative), 2 (neutral), 4 (positive) - Twitter format
- Binary: 0 (negative), 1 (positive)
- Text: "negative", "neutral", "positive"

#### 3. **Milestone 2: TextBlob Analysis**

- Navigate to "🎯 Milestone 2: TextBlob"
- Click "🚀 Run TextBlob Analysis"
- View polarity, subjectivity, and sentiment classification
- Results stored in SQLite database

#### 4. **Milestone 3: LLM Analysis**

- Navigate to "🤖 Milestone 3: LLM Analysis"
- Configure batch size (8, 16, or 32)
- Click "🚀 Run LLM Analysis"
- RoBERTa model automatically downloads on first run
- GPU acceleration used if available

#### 5. **Compare Results**

- Navigate to "📊 Compare Results"
- View comprehensive comparison metrics:
  - Performance metrics (speed, throughput)
  - Accuracy comparison (if ground truth available)
  - Confusion matrices
  - Per-class performance
  - Agreement analysis
  - Sample predictions

#### 6. **Generate & Email Report**

- Navigate to "📧 Generate & Email Report"
- Review comprehensive text report
- Download report or send via email
- Configure recipient email address

## 📊 Analysis Metrics

### TextBlob Metrics

- **Polarity**: -1 (most negative) to +1 (most positive)
- **Subjectivity**: 0 (objective) to 1 (subjective)
- **Sentiment**: Classified as negative, neutral, or positive

### LLM Metrics

- **Confidence Score**: Model confidence (0 to 1)
- **Sentiment**: Classified as negative, neutral, or positive
- **Processing Speed**: Texts per second

### Comparison Metrics (with Ground Truth)

- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results
- **Agreement Rate**: Percentage of matching predictions

## ⚙️ Configuration

### Email Settings (in app.py)

```python
sender_email = "your-email@gmail.com"
sender_password = "your-app-password"  # Use app-specific password
```

**Gmail Setup:**

1. Enable 2-factor authentication
2. Generate app-specific password
3. Replace credentials in `send_email_report()` function

### Authentication Secret (in auth.py)

```python
SECRET_KEY = "your-secret-key-here"  # Change for production
```

### Model Configuration

```python
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
batch_size = 16  # Adjust based on GPU memory
max_length = 128  # Maximum token length
```

## 🔧 Troubleshooting

### Common Issues

**Issue: TextBlob ImportError**

```bash
pip install textblob
python -m textblob.download_corpora
```

**Issue: PyTorch CUDA not available**

- Install CUDA-enabled PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

**Issue: Model download fails**

- Check internet connection
- Ensure sufficient disk space (~500MB for RoBERTa)
- Manually download from Hugging Face

**Issue: Email sending fails**

- Use app-specific password for Gmail
- Enable "Less secure app access" (if not using 2FA)
- Check SMTP server and port settings

**Issue: Database locked error**

- Close all database connections
- Delete `.db` files and restart

## 📈 Performance Tips

### Speed Optimization

- **Reduce sample size** for faster testing
- **Increase batch size** (if GPU memory allows)
- **Use GPU** for LLM analysis when available
- **Parallel processing** possible by modifying code

### Memory Management

- Large datasets: Process in chunks
- Clear session state between analyses
- Use sampling for initial exploration

## 🛡️ Security Considerations

### Production Deployment

1. **Change SECRET_KEY** in auth.py
2. **Use environment variables** for sensitive data
3. **Enable HTTPS** for secure connections
4. **Implement rate limiting** for API calls
5. **Regular database backups**
6. **Update dependencies** regularly

### Password Security

- Passwords hashed with PBKDF2-HMAC-SHA256
- 100,000 iterations for key derivation
- Unique salt per password
- No plaintext password storage

## 📝 Data Privacy

- User data stored locally in SQLite
- No external data transmission (except email)
- Authentication tokens expire after 24 hours
- Clear session on logout

## 🤝 Contributing

Contributions welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🔮 Future Enhancements

- [ ] Multiple LLM model support (BERT, GPT, etc.)
- [ ] Real-time sentiment tracking
- [ ] API endpoint for external integration
- [ ] Advanced visualization dashboards
- [ ] Multi-language support
- [ ] Cloud deployment templates
- [ ] Batch processing from cloud storage
- [ ] Custom model fine-tuning interface

## 📞 Support

For issues, questions, or contributions:

- Open an issue on GitHub
- Check documentation at `/docs`
- Review troubleshooting section above

## 🙏 Acknowledgments

- **TextBlob**: Pattern library for text processing
- **Hugging Face**: Pre-trained transformer models
- **Streamlit**: Web application framework
- **Cardiff NLP**: RoBERTa sentiment model

## 📚 References

- [TextBlob Documentation](https://textblob.readthedocs.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)

---

**Built with ❤️ for sentiment analysis enthusiasts**

_Last Updated: November 2025_
