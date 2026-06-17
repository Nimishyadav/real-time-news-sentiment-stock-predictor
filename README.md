# 📈 Real-Time News Sentiment Stock Predictor

A real-time financial sentiment analysis platform that collects market news, analyzes sentiment using FinBERT, and generates stock-specific sentiment signals.

The project combines financial news sentiment with NSE stock price data to generate Bullish, Bearish, and Neutral signals through an interactive Streamlit dashboard.

---

## 🚀 Features

- Live news aggregation from NewsAPI and RSS feeds
- Financial sentiment analysis using FinBERT
- Bullish / Bearish / Neutral signal generation
- Historical sentiment storage using SQLite
- NSE stock price integration using yFinance
- Interactive Streamlit dashboard
- Multi-stock monitoring and ranking
- Auto-refresh support
- Historical price and sentiment visualization

---

## 🛠️ Tech Stack

- Python
- FinBERT (Transformers)
- Streamlit
- SQLite
- Pandas
- NumPy
- yFinance
- NewsAPI
- Plotly

---

## 📊 Dashboard Highlights

### Signal Generation
- Bullish Signals
- Bearish Signals
- Neutral Signals

### Market Insights
- Live sentiment score
- Price vs Sentiment visualization
- Daily stock rankings
- Headline feed monitoring

### Controls
- Stock selector
- Historical date range
- Auto refresh options

---

## 🏗️ System Architecture

NewsAPI + RSS Feeds
        ↓
 Data Collection
        ↓
 FinBERT NLP Pipeline
        ↓
 Sentiment Aggregation
        ↓
 SQLite Database
        ↓
 Streamlit Dashboard

---

## 📂 Project Structure

```text
project/
│
├── dashboard.py
├── fetch_news.py
├── sentiment_engine.py
├── database.py
├── requirements.txt
├── .env
├── sentiment.db
│
├── assets/
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/real-time-news-sentiment-stock-predictor.git

cd real-time-news-sentiment-stock-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
NEWS_API_KEY=YOUR_API_KEY
```

---

## ▶️ Run the Project

Step 1: Fetch latest news

```bash
python fetch_news.py
```

Step 2: Run sentiment analysis

```bash
python sentiment_engine.py
```

Step 3: Launch dashboard

```bash
streamlit run dashboard.py
```

---



## 🔮 Future Improvements

- Streamlit Cloud Deployment
- Additional News Sources
- Advanced Sentiment Weighting
- Historical Signal Accuracy Tracking
- Portfolio-Level Sentiment Monitoring

---

## ⚠️ Disclaimer

This project is intended for educational and learning purposes only.

The generated signals should not be considered financial advice or investment recommendations.

---

<img width="1913" height="1020" alt="Screenshot 2026-06-17 142944" src="https://github.com/user-attachments/assets/387f8fea-4c2f-40d5-a976-c52f3843c3f0" />
<img width="1917" height="900" alt="Screenshot 2026-06-17 143433" src="https://github.com/user-attachments/assets/c3d309db-967f-4d7b-b2da-f256d8a2b397" />
<img width="1918" height="1018" alt="Screenshot 2026-06-17 143354" src="https://github.com/user-attachments/assets/1b220445-ecd0-447e-b1be-9f2d883c3e7f" />
<img width="1913" height="1020" alt="Screenshot 2026-06-17 143327" src="https://github.com/user-attachments/assets/7b651745-1595-43b0-a484-1a1da099fb0d" />

