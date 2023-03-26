import re
import pandas as pd

def preprocess(data):
    # data is a Pandas series (a mini dataframe)
    # #making all the data lowercase
    data = data.str.lower()
    data = data.str.replace('[^a-zA-Z ]+','',regex=True)
    return data

def preprocess_text(data):
    # data is a string
    # #making all the data lowercase
    data = str(data.lower())
    data = re.sub('[^a-zA-Z ]+','', data)
    return data