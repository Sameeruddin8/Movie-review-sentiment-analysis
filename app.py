from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization
from keras.models import load_model
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
app = Flask(__name__)
new_model = load_model('Model/MovieSentiment.h5')

def preprocess(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
    
data = pd.read_csv('processed_movie.csv')
X = data['text']
Max_feature = 10000
vectorization = TextVectorization(max_tokens=Max_feature+1, output_sequence_length=1800, output_mode='int')
vectorization.adapt(X.values)

@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict", methods = ['GET','POST'])
def predict():
    input_text = preprocess(request.form.get('review'))
    vector = vectorization(input_text)
    res = new_model.predict(np.expand_dims(vector,0))
    print(res)
    return render_template('index.html', pred = "According to the above review it is a good movie" if res > 0.5 else "According to the above review it is a bad movie")
    
if __name__ == '__main__':
    app.run(debug = True)

