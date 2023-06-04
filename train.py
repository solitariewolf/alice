import os
import nltk
import numpy as np
import textract
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

# Get all PDF and TXT files in the TR directory
files = [f for f in os.listdir('TR') if f.endswith('.pdf') or f.endswith('.txt')]

# Read and process the text from each file
texts = []
for file in files:
    if file.endswith('.pdf'):
        pdf = PdfReader(open('TR/' + file, 'rb'))
        text = ''
        for page in range(len(pdf.pages)):
            text += pdf.pages[page].extract_text()
        texts.append(text)
    elif file.endswith('.txt'):
        text = open('TR/' + file, 'r').read()
        texts.append(text)
        
# Read and process the text from each file
texts = []
for file in files:
    if file.endswith('.pdf'):
        pdf = PdfReader(open('TR/' + file, 'rb'))
        text = ''
        for page in range(len(pdf.pages)):
            text += pdf.pages[page].extract_text()
        texts.append(text)
    elif file.endswith('.txt'):
        text = open('TR/' + file, 'r').read()
        texts.append(text)

# Create TR2 directory if it doesn't exist
if not os.path.exists('TR2'):
    os.makedirs('TR2')

# Write texts to TR2 directory
for i, text in enumerate(texts):
    with open('TR2/text' + str(i) + '.txt', 'w') as f:
        f.write(text)

# TODO: Process texts to create training data

# TODO: Train a machine learning model with the training data

# TODO: Save the trained model

