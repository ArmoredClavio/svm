from flask import Flask, render_template, request, jsonify
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

app = Flask(__name__,template_folder='Templates')
ps = PorterStemmer()

model = pickle.load(open('model.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'FAKE REVIEW' if model.predict(review_vect) == -1 else 'GENUINE REVIEW'
    return prediction

@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)


@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    app.run()