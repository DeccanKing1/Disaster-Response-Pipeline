#lets import necessary libraries
import sys
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report


nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    
    
    """
    Function loads data from databse file
    
    Arguments:
        database_filepath- path to SQLite db
    Output:
        X- feature DataFrame
        Y -DataFrame for labels
        category_names
    """
    
    #engine = create_engine('sqlite:///' + database_filepath)
    #table_name = os.path.basename(database_filepath)
    
    print('testing')
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    print(engine.table_names())
    df = pd.read_sql_table("df",engine)
    #df = pd.read_sql_table(table_name,engine)
    
    X = df['message']
    Y = df.iloc[:,4:]
    
    return X, Y, df.columns[4:]
    
    
def tokenize(text):
    
    
    """
    Tokenizer function to make data ready for NLP task
    
    Arguments:
        text- text message list
    Output:
        tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    
    """
    Function builds the model
    
    NLP pipeline is implemented here with gridsearch to select best hyper parameters
    
    Arguments: None
    Returns: Pipeline object model
    """
   
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
       
        'features__text_pipeline__vect__max_features': (None, 5000),
        'features__text_pipeline__tfidf__use_idf': (True, False)
        }         
    cv = GridSearchCV(pipeline, param_grid = parameters)
    return cv

    
    
    
    
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Function to print class accuracy
    """
    
    y_pred_test = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred_test))

def save_model(model, model_filepath):
    """ Save model as pickle file"""
    import pickle
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    
    """  Main function to execute NLP pipline using GridSearch"""
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