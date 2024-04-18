import praw
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import matplotlib.pyplot as plt

# Initialize PRAW
reddit = praw.Reddit(client_id='your_client_id',
                     client_secret='your_client_secret',
                     user_agent='script by your_username',
                     username='your_username',
                     password='your_password')

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

# Streamlit application
st.title('Sentiment Analysis of Reddit Posts')

# User inputs
company = st.text_input("Enter the keyword to search:", "Nike")
start_date = pd.to_datetime(st.date_input("Start date", pd.to_datetime("2023-01-01")))
end_date = pd.to_datetime(st.date_input("End date", pd.to_datetime("2023-12-31")))

if st.button("Analyze Sentiment"):
    # Fetch data
    posts = []
    search_query = f'{company}'
    for submission in reddit.subreddit('all').search(search_query, limit=1000):
        created_time = pd.to_datetime(submission.created_utc, unit='s')
        if start_date <= created_time <= end_date:
            posts.append([submission.title, submission.selftext, created_time])

    # Create DataFrame
    if posts:
        df = pd.DataFrame(posts, columns=['title', 'text', 'created'])
        df['created'] = pd.to_datetime(df['created'])
        df['quarter'] = df['created'].dt.to_period('Q')

        # Sentiment analysis function
        def sentiment_analysis(text):
            try:
                encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
                with torch.no_grad():
                    output = model(**encoded_input)
                scores = output[0][0].softmax(0)
                labels = ['negative', 'neutral', 'positive']
                sentiment_score = dict(zip(labels, scores.numpy()))
                return max(sentiment_score, key=sentiment_score.get)
            except Exception as e:
                return "error"

        # Apply sentiment analysis
        df['sentiment'] = df['text'].apply(sentiment_analysis)

        # Group data by quarter and sentiment
        sentiment_counts = df.groupby(['quarter', 'sentiment']).size().unstack(fill_value=0)

        # Calculate percentages
        quarter_totals = sentiment_counts.sum(axis=1)
        sentiment_percentages = sentiment_counts.div(quarter_totals, axis=0) * 100

        # Plotting sentiment percentages per quarter
        sentiment_percentages.plot(kind='bar', stacked=True)
        plt.xlabel('Quarter')
        plt.ylabel('Percentage')
        plt.title('Quarterly Sentiment Analysis')
        plt.legend(title='Sentiment')
        st.pyplot(plt)

    else:
        st.write("No posts found for the specified criteria. Please adjust your filters or verify API usage.")
