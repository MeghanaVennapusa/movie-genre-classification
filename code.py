import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('movie_genre_dataset.csv')  # Ensure your data has 'plot' and 'genre' columns


def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

df['clean_plot'] = df['plot'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_plot']).toarray()
y = df['genre']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:\n', classification_report(y_test, y_pred))


def predict_genre(plot):
    cleaned_plot = clean_text(plot)
    vectorized_plot = vectorizer.transform([cleaned_plot]).toarray()
    genre = model.predict(vectorized_plot)[0]
    return genre


new_plot = "A young wizard embarks on an epic adventure to defeat a dark lord."
print(f'Predicted Genre: {predict_genre(new_plot)}')
