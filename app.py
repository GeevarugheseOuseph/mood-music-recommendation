from flask import Flask, render_template, request

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

import pandas as pd

app = Flask(__name__)

# SIMPLE DATASET

data = {

    'text': [

        'I am very happy',
        'I feel amazing',
        'I am sad',
        'I feel terrible',
        'I am angry',
        'I hate this',
        'I feel calm',
        'I am relaxed'

    ],

    'emotion': [

        'happy',
        'happy',
        'sad',
        'sad',
        'angry',
        'angry',
        'calm',
        'calm'

    ]
}

df = pd.DataFrame(data)

# TRAIN MODEL

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['text'])

y = df['emotion']

model = LogisticRegression()

model.fit(X, y)

# SONG DATABASE

songs = {

    'happy': [
        'Happy - Pharrell Williams',
        'On Top Of The World - Imagine Dragons',
        'Dancing Queen - ABBA'
    ],

    'sad': [
        'Fix You - Coldplay',
        'Let Her Go - Passenger',
        'Back to friends - Sombr'
    ],

    'angry': [
        'Believer - Imagine Dragons',
        'With Rage - Yoshimasa Terui',
        'Disappear - Bullet for my Valentine'
    ],

    'calm': [
        'Weightless - Marconi Union',
        'ethereal - inertia., Nadav Cohen',
        'sova - Theo Aabel, after noon'
    ]
}

# HOME PAGE

@app.route('/')
def home():

    return render_template('index.html')

# PREDICTION PAGE

@app.route('/predict', methods=['POST'])
def predict():

    user_text = request.form['text']

    text_vector = vectorizer.transform([user_text])

    prediction = model.predict(text_vector)[0]

    recommended_songs = songs[prediction]

    return render_template(

        'result.html',

        emotion=prediction,

        songs=recommended_songs

    )

if __name__ == '__main__':

    app.run(debug=True)