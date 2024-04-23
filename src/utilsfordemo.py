import numpy as np
import pandas as pd
# import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import contractions
from num2words import num2words
# from gensim.models import Word2Vec
import re
import pickle

def preProcessText(df, column):
    df[column] = df[column].astype(str)
    df[column] = df[column].str.lower()
    
    df[column] = df[column].apply(word_tokenize)
    punctuation = set(string.punctuation)
    df[column] = df[column].apply(lambda tokens: [token for token in tokens if token not in punctuation])

    stop_words = set(stopwords.words('english'))
    df[column] = df[column].apply(lambda tokens: [token for token in tokens if token not in stop_words])

    lemmatizer = WordNetLemmatizer()
    df[column] = df[column].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

    df[column] = df[column].apply(lambda tokens: [num2words(token) if token.isdigit() else token for token in tokens])

    df[column] = df[column].apply(lambda tokens: [contractions.fix(token) for token in tokens])

    df[column] = df[column].apply(lambda tokens: [token for token in tokens if token.isalnum()])

    df[column] = df[column].apply(lambda tokens: ' '.join(tokens))

    return df[column]

def dropUnwantedTechsAndKeys(df):
    with open('src/unWantedTechs.pkl', 'rb') as f:
        unwantedTechs = pickle.load(f)
    df = df.drop(unwantedTechs, axis = 1)
    
    with open('src/unwantedKeywords.pkl', 'rb') as f:
        unwantedKeywords = pickle.load(f)
    df = df.drop(unwantedKeywords, axis = 1)
    
    return df

def convStrToList(df, column):
    df[column] = df[column].fillna('')
    df[column] = df[column].str.split(',')
    df[column] = [[col.strip() for col in sublist] for sublist in df[column]]
    
    allColumns = [col.strip() for sublist in df[column] for col in sublist]
    allColumns = set([col for col in allColumns if col != ''])
    allColumns = sorted(allColumns)
    
    return df[column]


def unPickelData(df):
    import pickle
    with open('src/seoDes.pkl', 'rb') as f:
        seoPickeler = pickle.load(f)
        
    seoMatrix = seoPickeler.transform(df['SEO Description'])
    seoColumns = seoPickeler.get_feature_names_out()
    seoDummies = pd.DataFrame(seoMatrix.toarray(), columns = seoColumns)
    df.reset_index(drop=True, inplace=True)
    seoDummies.reset_index(drop=True, inplace=True)

    # Industries
    with open('src/industry.pkl', 'rb') as f:
        industryPickeler = pickle.load(f)
        
    df['Industry'] = df['Industry'].fillna('')
    industryMatrix = industryPickeler.transform(df['Industry'])
    industryColumns = industryPickeler.classes_
    industryDummies = pd.DataFrame(industryMatrix, columns=industryColumns)
    industryDummies.reset_index(drop=True, inplace=True)

    # Technologies
    with open('src/technology.pkl', 'rb') as f:
        techPickler = pickle.load(f)
    
    df['Technologies'] = df['Technologies'].fillna('')
    techMatrix = techPickler.transform(df['Technologies'])
    techColumns = techPickler.classes_
    techDummies = pd.DataFrame(techMatrix, columns = techColumns)
    techDummies.reset_index(drop=True, inplace=True)

   # Keywords
    with open('src/keyword.pkl', 'rb') as f:
        keywordPickler = pickle.load(f)
        
    df['Keywords'] = df['Keywords'].fillna('')
    keywordMatrix = keywordPickler.transform(df['Keywords'])
    keywordColumns = keywordPickler.classes_
    keywordDummies = pd.DataFrame(keywordMatrix, columns = keywordColumns)
    keywordDummies.reset_index(drop=True, inplace=True)
    
    data = pd.concat([ industryDummies,techDummies, keywordDummies, seoDummies], axis=1) 

    return data

def dropDuplicateColumns(df):
    duplicateColumns = df.columns[df.columns.duplicated()]
    temp = pd.DataFrame()

    for col in duplicateColumns:
        duplicateIndices = df.columns.get_loc(col)
        temp[col] = df.iloc[:, duplicateIndices].any(axis=1)
    
    df = df.drop(duplicateColumns, axis =1)
    df = pd.concat([df,temp],axis = 1)
    
    return df


def preProcessData(df):
    df['Technologies'] = convStrToList(df, 'Technologies')
    df['Keywords'] = convStrToList(df, 'Keywords')
    df['SEO Description'] = preProcessText(df, 'SEO Description')

    return df


def testModel(test):
    test_bckp = test.copy()
    test = test.drop(['Employees', 'Company City', 'Company State', 'Company Country', 'Total Funding', 'Latest Funding', 'Latest Funding Amount', 'Last Raised At', 'Annual Revenue','Number of Retail Locations', 'Founded Year'], axis = 1)
    test = preProcessData(test)
    test = unPickelData(test)
    test = dropDuplicateColumns(test)
    test = dropUnwantedTechsAndKeys(test)
    
    with open('src/model.pkl', 'rb') as f:
        loadedModel = pickle.load(f)
        
    test['output'] = loadedModel.predict(test)
    test_bckp['ICP'] = test['output'].apply(lambda x: 'ICP' if x == 1 else 'Not an ICP')

    return test_bckp
