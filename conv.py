import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Get all TXT files in the TR2 directory
files = [f for f in os.listdir('TR2') if f.endswith('.txt')]

# Read and process the text from each file
texts = []
labels = []
for file in files:
    text = open('TR2/' + file, 'r').read()
    texts.append(text)

    # Define your labels here. For example:
    if 'category1' in file:
        labels.append('category1')
    else:
        labels.append('category2')

# Process texts with TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Convert labels to numerical values
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Save the processed data
np.save('DATA/X.npy', X.toarray())
np.save('DATA/y.npy', y)

# Save the vectorizer and the encoder
with open('DATA/vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('DATA/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

