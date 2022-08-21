from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
ps = PorterStemmer()

app = Flask(__name__)

nltk.download('punkt')
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    rough_list = []
    for i in text:
        if i.isalnum():
            rough_list.append(i)
    text = rough_list[:]        # We have to clone the list because list is a mutable datatype.
    rough_list.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            rough_list.append(i)
    text = rough_list[:]
    rough_list.clear()
    for i in text:
        rough_list.append(ps.stem(i))  
    return " ".join(rough_list) # return the string 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    email = request.form.get('email_content')
    transformed_email_content = transform_text(email)
    vector_input = tfidf.transform([transformed_email_content])
    result = model.predict(vector_input)[0]
    if result == 1:
        return render_template('index.html', prediction='Spam')
    elif result == 0:
        return render_template('index.html', prediction='Not Spam!')
    else:
        return render_template('index.html', prediction=' ')

if __name__=="__main__":
    app.run(debug=True)