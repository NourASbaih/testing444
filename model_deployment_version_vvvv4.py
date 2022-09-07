#!/usr/bin/env python
# coding: utf-8

# In[3]:


from numpy import array
#Word Tokenization
#from keras.preprocessing.sequence import pad_sequences
#from keras.models import Sequential
#from keras.layers.core import Activation, Dropout, Dense
#from keras.layers import Flatten, LSTM
#from keras.layers import GlobalMaxPooling1D
#from keras.models import Model
#from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
#from keras.preprocessing.text import Tokenizer
#from keras.layers import Input
#from keras.layers.merge import Concatenate
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import inflect
#lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#stopwords
from nltk.corpus import stopwords
#for GloVe Model
from numpy import array
from numpy import asarray
from numpy import zeros
import tensorflow as tf
import nltk

import sklearn
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


# In[4]:


# training data
train = pd.read_csv('nfr_dataset.csv')
#train = pd.read_csv('C:/Users/Nour Ashraf/Desktop/nfr_dataset.csv')


# In[5]:


train.head()


# In[6]:


# import the inflect library
# Convert number to text function
p = inflect.engine()

# convert number into words
def convert_number(text):
    # split string into list of words
    temp_str = text.split()
    # initialise empty list
    new_string = []

    for word in temp_str:
        # if word is a digit, convert the digit
        # to numbers and append into the new_string list
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)

        # append the word as it is
        else:
            new_string.append(word)

    # join the words of new_string to form a string
    temp_str = ' '.join(new_string)
    return temp_str


# In[7]:


#lemmatization function 
lemmatizer = WordNetLemmatizer()
# lemmatize string
def lemmatize_word(text):
    word_tokens = word_tokenize(text)
    # provide context i.e. part-of-speech
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens]
    return lemmas


# In[8]:


#stop word function 
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('wordnet')
stopwords=set(stopwords.words('english'))
#stopwords = ['ourselves', 'hers', 'between']
def remove_stopwords(data):
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            if word.lower() not in stopwords:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array


# In[9]:


#Data preprocessing as the following :
#sub >>>> Replaces one or many matches with a string
    # 1)Removing single character , we won't use it 
    # 2)Text to Lowercase
    # 3)Transform percent sign to text 'percent'
    # 4)Convert / if it lies between 2 numbers to 'divided by'
    # 5)Convert . if it lies between 2 numbers to 'point'
    # 6)Convert - if it lies between 2 words to ' ' space to avoid mix the words togather 
    # 7)Remove the odd symbloe
    # 8)Handle the 10x10 symbole (it dosen'tworkwith other functions)
    # 9)Removing punctuations
    # 10)Converting the numbers into words
    # 11)Removing extra spaces
    # 12)lemmatize_word  -- <<stemming dosen't work with this function>> .. 
    # 13)Remove stop  words 
    # 14)Before merge the text for each sentence >> we need to check if any of them are empty to be deleted 
    # 15)Use join to merge the text again togather


def preprocess_text(sen):
    tempArr = []
    #sen = re.sub(r"\s+[a-zA-Z]\s+", ' ', sen)  #1
    sen = re.sub('([A-Z]+)', lambda m: m.group(0).lower(), sen)  #2
    sen = re.sub(r'\%', ' percent', sen)   #3 
    sen = re.sub(r"(\d)(\/)", r"\g<1> divided by ", sen) #4
    sen = re.sub(r"(\d)(\.)", r"\g<1> point ", sen)      #5
    sen = re.sub(r'\-', ' ', sen) #6
    sen = re.sub(r'�',' ', sen)  #7
    sen = re.sub('10x10','ten by ten feet screen',sen) #8
    sen = re.sub(r'[' + string.punctuation + ']', ' ', sen)   #9
    sen = convert_number(sen)  #10
    sen = re.sub(r'\s\s+', '', sen)   #11
    sen = lemmatize_word(sen)   #12
    sen = remove_stopwords(sen)   #13
    sen = list(filter(None, sen )) #14
    return " ".join(sen)


# In[10]:


# clean training data
#train_tweet = preprocess_text(train["Requirements"])
#train_tweet = pd.DataFrame(train_tweet)

#<requirement, label>
#this step should be performed on both train and test data becuase they are in a different files 

#train data
#Train_Before_Preprocess for comparasion 
Train_Before_Preprocess = train["RequirementText"]
train_clean_text = []
#store proccessed text 
train_clean_text = list(map(preprocess_text,train["RequirementText"]))
#y_train = train_labels.values

#Train_Before_Preprocess = train["RequirementText"]
#test_tweet = []
#store proccessed text 
#test_tweet = list(map(preprocess_text,test["RequirementText"]))

#test data
#Test_Before_Preprocess for comparasion 
#Test_Before_Preprocess = test["Requirements"]
#X_test = []
#store proccessed text 
#X_test = list(map(preprocess_text,test["Requirements"]))
#y_test = test_labels.values


# In[11]:


# append cleaned tweets to the training data
train["RequirementText"] = train_clean_text

# compare the cleaned and uncleaned tweets
train.head(10)


# In[12]:


train['class'].value_counts()


# In[13]:


train['class'].value_counts().plot(kind='bar')


# In[14]:


features = train.drop(columns=['class']).columns
print(features)


# In[15]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=888)
X_resampled, y_resampled = ros.fit_resample(train[features], train['class'])
y_resampled.value_counts()


# In[16]:


y_resampled.value_counts().plot(kind='bar')


# In[17]:


from sklearn.model_selection import train_test_split

# extract the labels from the train data
#y = train.class.values
y = y_resampled

# use 70% for the training and 30% for the test
x_train, x_test, y_train, y_test = train_test_split(X_resampled.RequirementText.values, y, 
                                                    stratify=y, 
                                                    random_state=1, 
                                                    test_size=0.3, shuffle=True)


# In[18]:


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(y_train)
Test_Y = Encoder.fit_transform(y_test)
                
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(train['RequirementText'])
Train_X_Tfidf = Tfidf_vect.transform(x_train)
Test_X_Tfidf = Tfidf_vect.transform(x_test)
                
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression

y = train["class"].values
cv = CountVectorizer()
ctmTr = cv.fit_transform(x_train)
X_test_dtm = cv.transform(x_test)

lr = LogisticRegression()
lr.fit(ctmTr, y_train)
lr_score = lr.score(X_test_dtm, y_test)
y_pred_lr = lr.predict(X_test_dtm)

############################################################## Model GUI ######################################3
############################################################## Model GUI ######################################3
############################################################## Model GUI ######################################3
############################################################## Model GUI ######################################3
import pickle
pickle.dump(lr , open('LR_model.pk1' , 'wb'))

#model = pickle.load(open('model.pk1' , 'rb'))

seq = ["The product is expected to run on Windows CE and Palm operating systems."]
#a = preprocess_text(t)
Xnew = cv.transform(seq)
#ynew = svm.predict(Xnew)


model =pickle.load(open('LR_model.pk1','rb'))
print(model.predict(Xnew))


# In[ ]:





# In[20]:


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(y_train)
Test_Y = Encoder.fit_transform(y_test)
                
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(train['RequirementText'])
Train_X_Tfidf = Tfidf_vect.transform(x_train)
Test_X_Tfidf = Tfidf_vect.transform(x_test)
                
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(Train_X_Tfidf, Train_Y)

y_pred_knn = knn.predict(Test_X_Tfidf)

############################################################## Model GUI ######################################3
############################################################## Model GUI ######################################3
############################################################## Model GUI ######################################3
############################################################## Model GUI ######################################3
import pickle
pickle.dump(knn , open('KNN_model.pk1' , 'wb'))


seq = ["The product is expected to run on Windows CE and Palm operating systems."]
#a = preprocess_text(t)
Xnew = Tfidf_vect.transform(seq)
#ynew = svm.predict(Xnew)


model =pickle.load(open('KNN_model.pk1','rb'))
print(model.predict(Xnew))


# In[21]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
text_rfc = Pipeline([('tfidf', TfidfVectorizer(lowercase=True)),
                     ('clf', RandomForestClassifier(class_weight='balanced'))])

text_rfc.fit(x_train, y_train)

############################################################## Model GUI ######################################3
############################################################## Model GUI ######################################3
############################################################## Model GUI ######################################3
############################################################## Model GUI ######################################3
import pickle
pickle.dump(text_rfc , open('RF_model.pk1' , 'wb'))

#model = pickle.load(open('model.pk1' , 'rb'))

seq = ["The product is expected to run on Windows CE and Palm operating systems."]
#a = preprocess_text(t)
#Xnew = Tfidf_vect.transform(seq)
#ynew = svm.predict(Xnew)


model =pickle.load(open('RF_model.pk1','rb'))
print(model.predict(seq))


# In[ ]:





# In[ ]:





# In[ ]:


import pickle
import streamlit as st
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def main():
    st.title("Requirement Type Prediction App")
    st.subheader("This app predicts the Requirement type using Deep Learning Algorithms.")
    
    menu = ["Home","Experiment Models Summary","NLP","SVM Prediction Model","KNN Prediction Model","LogisticRegression Prediction Model","RandomForest Prediction Model"]
    choice = st.sidebar.selectbox("Menu" , menu)
    
    if choice == "Home":
        st.subheader("Dataset")
        st.write("The datset contains Functional requirements & and 11 type of Non-functional requirements or different types of software products.")
        st.write("CSV of the original dataset file can be found [here](https://drive.google.com/file/d/1hJZsMKpM2QRNa-Q3RDpeCOl4E97vZcua/view?usp=sharing)")
        st.write("RandomOverSampler method was used to oversample the minority classes of given the datasetin order to gain prediction model with high accuracy")
        st.write("CSV of the original dataset file can be found [here](https://drive.google.com/file/d/1g3LVHWR8eewbVssPwBQzbC-mmKl-fBa6/view?usp=sharing)")
        st.write("Requirements Data types: ")
        st.write("1-Functional (F)")
        st.write("2-Availability (A)")
        st.write("3-Fault Tolerance (FT)")
        st.write("4-Legal (L)")
        st.write("5-Look & Feel (LF)")
        st.write("6-Maintainability (MN)")
        st.write("7-Operational (O)")
        st.write("8-Performance (PE)")
        st.write("9-Portability (PO)")
        st.write("10-Scalability (SC)")
        st.write("11-Security (SE)")
        st.write("12-Usability (US)")
        
    if choice == "NLP":
        st.subheader("Natural Language Preproccessing") 
        with st.form(key='nlpForm' , clear_on_submit=True):
            before_nlp = st.text_area("Enter Text Here:")
            submit_text = st.form_submit_button(label='Preproccess text')
            
            #submit
            arr = [""]
            arr2 = [""] 
            if submit_text:
                str = before_nlp
                arr = str.split()  
                
                for i in range(0, len(arr)):    
                    arr2[0] = arr2[0] +" "+ arr[i];  
                
                t = arr2[0]
                after_nlp = preprocess_text(t)
            
                st.success("Your Data has been submitted successfully!!")
                
                st.info("Original Text is :"+" "+ before_nlp)
                        
                st.success("Text After Preproccessing : "+" "+after_nlp)

            
        
        
    if choice == "SVM Prediction Model": 
        st.subheader("Prediction using SVM")
        st.info("This model was trained using the SVM algorithm , TF-IDF vectorizor and the oversampled dataset. Below are the evaluation metrics for this model.")
        #df = pd.read_csv("C:/Users/Nour Ashraf/Desktop/New/SVM.csv")

        #Method 1
        #st.dataframe(df)
        st.info("The accuracy of this model reaches up to 99% !")
        
        with st.form(key='nlpForm' , clear_on_submit=True):
            message = st.text_area("Enter Text Here:")
            submit_message = st.form_submit_button(label='Predict')
            
            #submit
            if submit_message:
                
                
                Encoder = LabelEncoder()
                Train_Y = Encoder.fit_transform(y_train)
                Test_Y = Encoder.fit_transform(y_test)

                Tfidf_vect = TfidfVectorizer(max_features=5000)
                Tfidf_vect.fit(train['RequirementText'])
                Train_X_Tfidf = Tfidf_vect.transform(x_train)
                Test_X_Tfidf = Tfidf_vect.transform(x_test)

                # Classifier - Algorithm - SVM
                # fit the training dataset on the classifier
                SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
                SVM.fit(Train_X_Tfidf,Train_Y)
                # predict the labels on validation dataset
                predictions_SVM = SVM.predict(Test_X_Tfidf)
                # Use accuracy_score function to get the accuracy
                print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


                r = ""
                f = ""
                empty = ""
                arr = [""]
                arr[0] = message
                
                Xnew = Tfidf_vect.transform(arr)
                pickle.dump(SVM , open('svm_model.pk1' , 'wb'))
                model = pickle.load(open('svm_model.pk1','rb'))
                
                result = model.predict(Xnew)
                if result == 0:
                    f= "Non-Functional Requirement (Availability)"
                if result == 1:
                    f= "Functional Requirement"
                if result == 2:
                    f= "Non-Functional Requirement (Fault Tolerance)"
                if result == 3:
                    f= "Non-Functional Requirement (Legal Requirement)"
                if result == 4:
                    f= "Non-Functional Requirement (Look & Feel)"
                if result == 5:
                    f= "Non-Functional Requirement (Maintainability)"
                if result == 6:
                    f= "Non-Functional Requirement (Operational)"
                if result == 7:
                    f= "Non-Functional Requirement (Performance)"
                if result == 8:
                    f= "Non-Functional Requirement (Portability)"
                if result == 9:
                    f= "Non-Functional Requirement (Scalability)"
                if result == 10:
                    f= "Non-Functional Requirement (Security)"
                if result == 11:
                    f= "Non-Functional Requirement (Usability)"
               
                
                st.success("Your Data has been submitted successfully!!")
                
                st.info("Original Requirement Text is :"+" "+ message)
                #st.write(message)
                        
                #st.success(result)
                st.success("The Requirement Type is:"+" "+f)
                
    
    if choice == "KNN Prediction Model": 
        st.subheader("Prediction using KNN")
        st.info("This model was trained using the KNN algorithm , TF-IDF vectorizor and the oversampled dataset. Below are the evaluation metrics for this model.")
        st.info("The accuracy of this model reaches up to 99% !")
        
        with st.form(key='nlpForm' , clear_on_submit=True):
            message = st.text_area("Enter Text Here:")
            submit_message = st.form_submit_button(label='Predict')
            
            #submit
            if submit_message:
                
                
                Encoder = LabelEncoder()
                Train_Y = Encoder.fit_transform(y_train)
                Test_Y = Encoder.fit_transform(y_test)

                Tfidf_vect = TfidfVectorizer(max_features=5000)
                Tfidf_vect.fit(train['RequirementText'])
                Train_X_Tfidf = Tfidf_vect.transform(x_train)
                Test_X_Tfidf = Tfidf_vect.transform(x_test)
                
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(Train_X_Tfidf, Train_Y)
                y_pred_knn = knn.predict(Test_X_Tfidf)

                r = ""
                f = ""
                empty = ""
                arr = [""]
                arr[0] = message
                
                Xnew = Tfidf_vect.transform(arr)
                pickle.dump(knn , open('KNN_model.pk1' , 'wb'))
                model = pickle.load(open('KNN_model.pk1','rb'))
                
                result = model.predict(Xnew)
                if result == 0:
                    f= "Non-Functional Requirement (Availability)"
                if result == 1:
                    f= "Functional Requirement"
                if result == 2:
                    f= "Non-Functional Requirement (Fault Tolerance)"
                if result == 3:
                    f= "Non-Functional Requirement (Legal Requirement)"
                if result == 4:
                    f= "Non-Functional Requirement (Look & Feel)"
                if result == 5:
                    f= "Non-Functional Requirement (Maintainability)"
                if result == 6:
                    f= "Non-Functional Requirement (Operational)"
                if result == 7:
                    f= "Non-Functional Requirement (Performance)"
                if result == 8:
                    f= "Non-Functional Requirement (Portability)"
                if result == 9:
                    f= "Non-Functional Requirement (Scalability)"
                if result == 10:
                    f= "Non-Functional Requirement (Security)"
                if result == 11:
                    f= "Non-Functional Requirement (Usability)"
               
                
                st.success("Your Data has been submitted successfully!!")
                
                st.info("Original Requirement Text is :"+" "+ message)
                #st.write(message)
                        
                #st.success(result)
                st.success("The Requirement Type is:"+" "+f)
        
        
    if choice == "LogisticRegression Prediction Model": 
        st.subheader("Prediction using LogisticRegression")
        st.info("This model was trained using the LogisticRegression algorithm , TF-IDF vectorizor and the oversampled dataset. Below are the evaluation metrics for this model.")
        st.info("The accuracy of this model reaches up to 99% !")
        
        with st.form(key='nlpForm' , clear_on_submit=True):
            message = st.text_area("Enter Text Here:")
            submit_message = st.form_submit_button(label='Predict')
            
            #submit
            if submit_message:
                
                y = train["class"].values
                cv = CountVectorizer()
                ctmTr = cv.fit_transform(x_train)
                X_test_dtm = cv.transform(x_test)

                lr = LogisticRegression()
                lr.fit(ctmTr, y_train)
                lr_score = lr.score(X_test_dtm, y_test)
                y_pred_lr = lr.predict(X_test_dtm)
                

                r = ""
                f = ""
                empty = ""
                arr = [""]
                arr[0] = message
                
                Xnew = cv.transform(arr)
                pickle.dump(lr , open('LR_model.pk1' , 'wb'))
                model =pickle.load(open('LR_model.pk1','rb'))
                
                
                result = model.predict(Xnew)
                r = result
                if r == 'F':
                    f = "Functional Requirement"
                if r == 'A':
                    f = "Non-Functional Requirement (Availability)"
                if r == 'FT':
                    f = "Non-Functional Requirement (Fault Tolerance)"
                if r == 'L':
                    f = "Non-Functional Requirement (Legal Requirement)"
                if r == 'LF':
                    f = "Non-Functional Requirement (Look & Feel)"
                if r == 'MN':
                    f = "Non-Functional Requirement (Maintainability)"
                if r == 'O':
                    f = "Non-Functional Requirement (Operational)"
                if r == 'PE':
                    f = "Non-Functional Requirement (Performance)"
                if r == 'PO':
                    f = "Non-Functional Requirement (Portability)"
                if r == 'SC':
                    f = "Non-Functional Requirement (Scalability)"
                if r == 'SE':
                    f = "Non-Functional Requirement (Security)"
                if r == 'US':
                    f= "Non-Functional Requirement (Usability)"
               
                
                st.success("Your Data has been submitted successfully!!")
                
                st.info("Original Requirement Text is :"+" "+ message)
                #st.write(message)
                        
                #st.success(result)
                st.success("The Requirement Type is:"+" "+f)
        
        
        
       
    if choice == "RandomForest Prediction Model": 
        st.subheader("Prediction using RandomForest")
        st.info("This model was trained using the RandomForest algorithm , TF-IDF vectorizor and the oversampled dataset. Below are the evaluation metrics for this model.")
        st.info("The accuracy of this model reaches up to 99% !")
        
        with st.form(key='nlpForm' , clear_on_submit=True):
            message = st.text_area("Enter Text Here:")
            submit_message = st.form_submit_button(label='Predict')
            
            #submit
            if submit_message:
                
                text_rfc = Pipeline([('tfidf', TfidfVectorizer(lowercase=True)),
                     ('clf', RandomForestClassifier(class_weight='balanced'))])

                text_rfc.fit(x_train, y_train)
                

                r = ""
                f = ""
                empty = ""
                arr = [""]
                arr[0] = message
                
                pickle.dump(text_rfc , open('RF_model.pk1' , 'wb'))
                model =pickle.load(open('RF_model.pk1','rb'))
                
                
                result = model.predict(arr)
                r = result
                if r == 'F':
                    f = "Functional Requirement"
                if r == 'A':
                    f = "Non-Functional Requirement (Availability)"
                if r == 'FT':
                    f = "Non-Functional Requirement (Fault Tolerance)"
                if r == 'L':
                    f = "Non-Functional Requirement (Legal Requirement)"
                if r == 'LF':
                    f = "Non-Functional Requirement (Look & Feel)"
                if r == 'MN':
                    f = "Non-Functional Requirement (Maintainability)"
                if r == 'O':
                    f = "Non-Functional Requirement (Operational)"
                if r == 'PE':
                    f = "Non-Functional Requirement (Performance)"
                if r == 'PO':
                    f = "Non-Functional Requirement (Portability)"
                if r == 'SC':
                    f = "Non-Functional Requirement (Scalability)"
                if r == 'SE':
                    f = "Non-Functional Requirement (Security)"
                if r == 'US':
                    f= "Non-Functional Requirement (Usability)"
               
                
                st.success("Your Data has been submitted successfully!!")
                
                st.info("Original Requirement Text is :"+" "+ message)
                #st.write(message)
                        
                #st.success(result)
                st.success("The Requirement Type is:"+" "+f)
        
        
            
    if choice == "Experiment Models Summary":
        st.subheader("Models Summary") 
        img = Image.open('C:/Users/Nour Ashraf/Desktop/toolPagedrawio.png')
        st.image(img)
        
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with' , ‘they’, ‘own’, ‘an’, ‘be’, ‘some’, ‘for’, ‘do’, ‘its’, ‘yours’, ‘such’, ‘into’, ‘of’, ‘most’, ‘itself’, ‘other’, ‘off’, ‘is’, ‘s’, ‘am’, ‘or’, ‘who’, ‘as’, ‘from’, ‘him’, ‘each’, ‘the’, ‘themselves’, ‘until’, ‘below’, ‘are’, ‘we’, ‘these’, ‘your’, ‘his’, ‘through’, ‘don’, ‘nor’, ‘me’, ‘were’, ‘her’, ‘more’, ‘himself’, ‘this’, ‘down’, ‘should’, ‘our’, ‘their’, ‘while’, ‘above’, ‘both’, ‘up’, ‘to’, ‘ours’, ‘had’, ‘she’, ‘all’, ‘no’, ‘when’, ‘at’, ‘any’, ‘before’, ‘them’, ‘same’, ‘and’, ‘been’, ‘have’, ‘in’, ‘will’, ‘on’, ‘does’, ‘yourselves’, ‘then’, ‘that’, ‘because’, ‘what’, ‘over’, ‘why’, ‘so’, ‘can’, ‘did’, ‘not’, ‘now’, ‘under’, ‘he’, ‘you’, ‘herself’, ‘has’, ‘just’, ‘where’, ‘too’, ‘only’, ‘myself’, ‘which’, ‘those’, ‘i’, ‘after’, ‘few’, ‘whom’, ‘t’, ‘being’, ‘if’, ‘theirs’, ‘my’, ‘against’, ‘a’, ‘by’, ‘doing’, ‘it’, ‘how’, ‘further’, ‘was’, ‘here’, ‘than’} 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




