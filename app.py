from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from custom_preprocessing import CustomPreProcessing
from custom_preprocessing import PreProcessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	TEXT = "Data Policy"
	LABEL = "label"

	def cleantext(TEXT):
	    stop_word = np.loadtxt("stopwords_en.txt", dtype=str)
	    TEXT = TEXT.apply(lambda x : preproc.remove_stop_words(x, stop_word))
	    TEXT = TEXT.apply(preprocess.remove_upper_case)
	    TEXT = TEXT.apply(preproc.remove_URL)
	    TEXT = TEXT.apply(preproc.remove_html)
	    TEXT = TEXT.apply(preproc.remove_emoji)
	    TEXT = TEXT.apply(lambda x: x.replace("'ve", " have"))
	    TEXT = TEXT.apply(lambda x: x.replace("n't", " not"))
	    return TEXT

	# ---- Create object to preprocess the text 
	preprocess = CustomPreProcessing()
	preproc = PreProcessing()
	df = pd.read_csv('../data/Data Policy small dataset.csv',encoding = "ISO-8859-1")


	df[TEXT] = cleantext(df[TEXT])
	tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
	tfidf_vect.fit(df[TEXT])
	xtrain_tfidf =  tfidf_vect.transform(df[TEXT])

	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(xtrain_tfidf,df[LABEL])
	
	if request.method == 'POST':
		url = request.form['message']
		import requests 
		from bs4 import BeautifulSoup 

		# getting response object 
		res = requests.get(url) 
		# Initialize the object with the document 
		soup = BeautifulSoup(res.content, "html.parser") 
		# Get the whole body tag 
		tag = soup.body 
		output = "" 
		# Print each string recursivey 
		for string in tag.strings: 
		    output += string
		data = [message]
		d = {'Data Policy': [output]}
		df_test = pd.DataFrame(data=d)
		df_test['Data Policy'] = cleantext(df_test['Data Policy'])
		xtest_tfidf =  tfidf_vect.transform(df_test[TEXT])
		my_prediction = clf.predict(xtest_tfidf)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
