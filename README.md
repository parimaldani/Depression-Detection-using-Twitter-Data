# 🧠 Depression Detection using Twitter Data

This project demonstrates the use of **Natural Language Processing (NLP)** and **Machine Learning** to detect signs of depression in tweets. It leverages the Twitter API for real-time data collection and applies text classification using a Naive Bayes algorithm.

## 🚀 Overview

- Built a lightweight and effective sentiment classification model to identify whether a tweet suggests depressive symptoms.
- Utilized the **Tweepy** API to gather tweets and **NLTK** for preprocessing (tokenization, lemmatization, POS tagging).
- Trained a **Naive Bayes classifier**, achieving **99% accuracy** on labeled data.

## 🧰 Technologies Used

- Python
- Tweepy (Twitter API)
- NLTK (Natural Language Toolkit)
- Scikit-learn
- Pandas, NumPy

## 🗂️ Project Structure
├── tweepy_streamer.py # Collects tweets using Twitter API ├── twitter_credentials.py # Stores Twitter API keys ├── NLTK_Sentiment_Analysis.py # Preprocessing and sentiment classification ├── tweets.csv # Stored tweets for training/testing


## 🧪 How it Works

1. **Data Collection**:  
   Uses `tweepy_streamer.py` to stream tweets and save them to `tweets.csv`.

2. **Preprocessing**:  
   Applies tokenization, lemmatization, and POS tagging using NLTK to clean and standardize the text.

3. **Model Training**:  
   Trains a Naive Bayes classifier to categorize tweets as *depressed* or *not depressed* based on labeled keywords and patterns.

4. **Prediction**:  
   The trained model can classify new, unseen tweets with high accuracy.

## ✅ Results

Achieved a classification **accuracy of 99%**, demonstrating strong potential for real-world mental health monitoring applications using social media data.

## 📌 Applications

- Mental health trend analysis  
- Content moderation systems  
- Real-time user sentiment tracking  
- Social media behavioral analysis

## 💡 Inspiration

Inspired by the increasing role of AI in content understanding and moderation, this project reflects the potential of applying machine learning to social media for real-time human well-being insights.

---

> 🔐 **Note**: This project is for educational purposes only. Ethical use of user data and mental health prediction should always follow privacy guidelines and medical oversight.

## 📬 Contact

Feel free to reach out or connect via [LinkedIn](https://www.linkedin.com/in/parimaldani/)
