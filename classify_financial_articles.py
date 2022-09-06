# pickel is a python module that allows you to save and load objects

# pandas is a python module that allows you to work with dataframes (tables)
# and data analysis, modeling and manipulation usually is used in machine learning

from cProfile import label
from gravityai import gravityai as grav
import pickle
import pandas as pd


model = pickle.load(open(''))
tfidf_vectorizer = pickle.load(open(''))
label_encoder = pickle.load(open(''))


def process(inPath, outPath):
    # read the csv file
    input_df = pd.read_csv(inPath)
    # vectorize the data
    features = tfidf_vectorizer.transform(input_df['text'])
    # predict the clases
    predictions = model.predict(features)
    # convert the output labels to categories
    input_df['category'] = label_encoder.inverse_transform(predictions)
    output_df = input_df[['id', 'category']]
    output_df.to_csv(outPath, index=False)


grav.wait_for_requests(process)
