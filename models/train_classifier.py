import sys
import nltk
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score
import pickle

nltk.download(['stopwords','punkt','wordnet'])


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+ database_filepath)
    database_name = database_filepath.split('/')[-1].split('.')[0] # split given path to get just the name
    df = pd.read_sql("SELECT * FROM "+database_name, engine)
    X = df['message'] 
    y = df.drop(['id','message','original','genre'],axis=1)
    return X, y, y.columns


def tokenize(text):
    # text lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenizing text
    words = word_tokenize(text)
    # Removing stop words
    words = [x for x in words if x not in nltk.corpus.stopwords.words('english')]
    # Lemmatize
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words] # for nouns
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words] # for verbs
    
    return lemmed


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
        ('tdidf', TfidfTransformer()),
        ('multi', MultiOutputClassifier(estimator=RandomForestClassifier()))
        ])
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    result=[]
    precision = []
    recall=[]
    f1=[]
    for i in range(y_pred.shape[1]):
        #result.append(classification_report(np.array(y_test.iloc[:,i]),y_pred[:,i]))
        precision.append(precision_score(np.array(y_test.iloc[:,i]),y_pred[:,i],average='weighted'))
        recall.append(recall_score(np.array(y_test.iloc[:,i]),y_pred[:,i],average='weighted'))
        f1.append(recall_score(np.array(y_test.iloc[:,i]),y_pred[:,i],average='weighted'))
    df_result=pd.DataFrame(data=[precision,recall,f1],index=['precision','recall','f1'], columns=y_test.columns)
    return df_result[category_names]


def save_model(model, model_filepath):
    classifier_pkl=open(model_filepath,'wb')
    pickle.dump(model,classifier_pkl)
    classifier_pkl.close()
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()