import streamlit as st  # for making the web app
import numpy as np  # for using arrays
import re # reegular expretion used for making patern
import pandas as pd  #for cleaning dataset
from nltk.corpus import stopwords # to remove (the,for,of,in.with) in the dataset
from nltk.stem.porter import PorterStemmer # it is used to make every word easy like (loved,loving ==> love)
from sklearn.feature_extraction.text import TfidfVectorizer # used to convert word into vector (fall ==> [0,0])
from sklearn.model_selection import train_test_split # to sPlit the data
from sklearn.linear_model import LogisticRegression # for logistic regression model
from sklearn.metrics import accuracy_score



news_df = pd.read_csv('train.csv') 
news_df = news_df.fillna(' ') # to fill missing values with empty strings
news_df['content'] = news_df['author'] + ' ' + news_df['title'] 
X = news_df.drop('label', axis=1) # droping the lable column
y = news_df['label'] 

# stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content) # this will evry thingelse insted of a-z & A-Z  (^  this stands for negation)
    stemmed_content = stemmed_content.lower() 
    stemmed_content = stemmed_content.split() 
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]  #for cleaning dataset
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming) 

# Vectorize data
X = news_df['content'].values 
y = news_df['label'].values
vector = TfidfVectorizer() 
vector.fit(X) 
X = vector.transform(X) # convert text to vector

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2) # stratify=y ensures that the proportion of labels is maintained in both train and test sets

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train,Y_train) # fit the model on training data


# website
st.title('Fake News Detector')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 0:
        st.header("This news is Real")
    else:
        st.header("This news is Fake")


st.write("Developed by :- ")
st.write("Rakshit Diwani")