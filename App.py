import streamlit as st
import pandas as pd
import plotly.express as px
import re
import string
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Function to clean text
def clean_text(text):
    return re.sub('[^a-zA-Z]', ' ', text).lower()

# Function to lemmatize text
def lemmatize_text(token_list):
    lemmatizer = WordNetLemmatizer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    return " ".join([lemmatizer.lemmatize(token) for token in token_list if not token in set(all_stopwords)])

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("tripadvisor_hotel_reviews.csv")
    return data

# Preprocess data
def preprocess_data(data):
    data['cleaned_text'] = data['review'].apply(clean_text)
    data['cleaned_text'] = data['cleaned_text'].apply(tokenize_text)
    data['lemmatized_review'] = data['cleaned_text'].apply(lemmatize_text)
    data['label'] = data['rating'].map({1.0:0, 2.0:0, 3.0:1, 4.0:1, 5.0:1})
    data['review_len'] = data['review'].apply(lambda x: len(x.split()))
    data['punct'] = data['review'].apply(count_punct)
    return data

# Tokenize text
def tokenize_text(text):
    return text.split()

# Count punctuation
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / (len(text) - text.count(" ")), 3) * 100

# Main function to run the app
def main():
    st.title("Hotel Review Analysis")
    st.subheader("Data Exploration")

    # Load data
    data = load_data()

    # Preprocess data
    df = preprocess_data(data)

    # Show data
    if st.checkbox("Show Data"):
        st.write(df.head())

    # Show distribution of ratings
    st.subheader("Distribution of Ratings")
    st.bar_chart(df['rating'].value_counts())

    # Show distribution of review length
    st.subheader("Distribution of Review Length")
    fig, ax = plt.subplots()
    ax.hist(df['review_len'], bins=50)
    ax.set_title('Distribution of Review Length')
    ax.set_xlabel('Review Length')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Filter positive, neutral, and negative reviews
    filtered_positive = " ".join(df[df['label'] == 1]['lemmatized_review'])
    filtered_negative = " ".join(df[df['label'] == 0]['lemmatized_review'])

    # Show word cloud for positive reviews
    st.subheader("Positive Reviews Word Cloud")
    wordcloud = WordCloud(max_font_size=160, margin=0, background_color="white", colormap="Greens").generate(filtered_positive)
    plt.figure(figsize=[10, 10])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.title("Positive Reviews Word Cloud")
    st.pyplot(plt)

    # Show word cloud for negative reviews
    st.subheader("Negative Reviews Word Cloud")
    wordcloud = WordCloud(max_font_size=160, margin=0, background_color="white", colormap="Reds").generate(filtered_negative)
    plt.figure(figsize=[10, 10])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.title("Negative Reviews Word Cloud")
    st.pyplot(plt)

# Second Page for Sentiment Analysis
def sentiment_analysis():
    st.title("Logistic Regression")
    st.subheader("Predict Sentiment")

    # Load the pre-trained model and TfidfVectorizer
    with open('modellr.sav', 'rb') as f:
        modellr = pickle.load(f)

    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    with open('PCA.pkl', 'rb') as f:
        pca = pickle.load(f)

    comment = st.text_input('Enter your review text here:', value="")
    
    def preprocess_text(text):
        if isinstance(text, str):
            processed_text = clean_text(text)
            review_len = len(text) - text.count(" ")
            punct = count_punct(text)
            tokenized_text = tokenize_text(processed_text)
            lemmatized_review = lemmatize_text(tokenized_text)
            return lemmatized_review, review_len, punct
        return "", 0, 0

    def predict_sentiment(input_text):
        lemmatized_review, review_len, punct = preprocess_text(input_text)
        X_comment = tfidf_vectorizer.transform([lemmatized_review])
        X_comment = pca.transform(X_comment)
        X_numerical = pd.DataFrame({'review_len': [review_len], 'punct': [punct]})
        X_vect = pd.concat([X_numerical, pd.DataFrame(X_comment)], axis=1)
        X_vect.columns = X_vect.columns.astype(str)

        # Perform predictions
        prediction_lr = modellr.predict(X_vect)

        # Return predictions
        return prediction_lr

    if st.button('Predict'):
        if comment:
            prediction_lr = predict_sentiment(comment)

            # Convert predictions to sentiment labels
            sentiment_lr = 'Positive' if prediction_lr == 1 else 'Negative'
            
            st.write("Prediction:", sentiment_lr)
        else:
            st.write("Please enter a review text.")

if __name__ == "__main__":
    pages = {"Data Exploration": main, "Sentiment Analysis": sentiment_analysis}
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()
