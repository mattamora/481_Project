"""
the main file.
Im basically done all i need to do is the slides and report.

This is the 481 ai version 
"""


import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


"""
State Space Representation of My Project:
- States (N): 
  - S0: Raw Review Text
  - S1: Cleaned Review
  - S2: Vectorized Review (TF-IDF)
  - S3: Classified Sentiment (Positive/Negative)

- Actions (A): 
  - A1: Clean Text using clean_and_lemmatize()
  - A2: Vectorize Text using TfidfVectorizer()
  - A3: Classify using ML Model (Logistic Regression and Naive Bayes for comparison)

- Start State (S): 
  - S0 (Raw review input from Yelps dataset)

- Goal State (GD): 
  - S3 (Final classified label for sentimentm, either Positive or Negative)

Transition Path: 
S0 → A1 → S1 → A2 → S2 → A3 → S3

Overview:
S0: Raw Review Text  →  A1: clean_and_lemmatize()  →  S1: Cleaned review
S1: Cleaned review  →  A2: TF-IDF transform  →  S2: Vectorized review
S2: Vectorized review  →  A3: Logistic Regression  →  S3: Sentiment Label
"""


# Load data
# S0: Raw Review Text
df = pd.read_csv('restaurant_reviews_filtered.csv')

# Relabel sentiment: 3.5 and above is Positive, 3.0 and below is Negative
def label_sentiment(stars):
    return 'Positive' if stars >= 3.5 else 'Negative'

df['nb_sentiment'] = df['stars'].apply(label_sentiment)
df = df[df['nb_sentiment'].isin(['Positive', 'Negative'])]


# A1: Clean Text using clean_and_lemmatize()
lemmatizer = WordNetLemmatizer()
def clean_and_lemmatize(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# S1 Cleaned Review
df['cleaned_text'] = df['text'].apply(clean_and_lemmatize)

# A2: Vectorize Text using TfidfVectorizer()
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,         # Remove words that appear in fewer than 2 reviews
    max_df=0.85       # Remove words that appear in more than 85% of reviews
)

# S2: Vectorized Review (TF-IDF)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['nb_sentiment']

# A3: Classify using ML Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# A3: Classify using ML Model 
# S3 (Final classified label for sentimentm, either Positive or Negative)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Printing S3, Goal State
#print("\nLogistic Regression Classification Report:\n")
#print(classification_report(y_test, y_pred))

# Logistic Regression Probabilistic Reasoning
y_proba_log = model.predict_proba(X_test)
#print("\nExample Logistic Regression Probabilities for First 5 Reviews:")
#for i, probs in enumerate(y_proba_log[:5]):
    #print(f"Review {i+1}: Positive={probs[1]:.2f}, Negative={probs[0]:.2f}")



""" 
# Using Naive Bayes with lemmatization and bigrams (part4.py)
# Used to comapre to Logistic Regression Approach
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X = tfidf.fit_transform(df['cleaned_text'])
y = df['nb_sentiment']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train)
y_pred = NB_model.predict(X_test)

#print("\nNaïve Bayes Classification Report:\n")
#print(classification_report(y_test, y_pred))

# Naive Bayes Probabilistic Reasoning
y_proba_nb = NB_model.predict_proba(X_test)
#print("\nExample Naïve Bayes Probabilities for First 5 Reviews:")
#for i, probs in enumerate(y_proba_nb[:5]):
    #print(f"Review {i+1}: Positive={probs[1]:.2f}, Negative={probs[0]:.2f}")

"""


 
# Test the model on a new review
custom_review = "Would not want to come back here. The place stank and the food was average."
# Preprocess and transform
cleaned = clean_and_lemmatize(custom_review)
X_custom = tfidf.transform([cleaned])

# Predict
prediction = model.predict(X_custom)[0]
print(f"\nSentiment Prediction for custom review:\n\"{custom_review}\"\n→ {prediction}\n")



