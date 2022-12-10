# authors: Luke Yang, HanChen Wang
# date: 2022-12-10

"""This script read the raw data from the input file and splits it on a 20-80 train test split ratio. Some feature cleaning are performed to remove ambiguity. Eventually it saves the cleaned data in .csv files in the output folder.

Usage: data_preprocessor.py --input=<input> --output=<output> ...

Arguments:
  --input=<input>        path to raw data file
  --output=<output>      folder to save the preprocessed and split dataset (as train.csv and test.csv)

Make sure you call this script in the repo's root path.
Example:

python src/data_preprocessor.py --input=data/raw/AB_NYC_2019.csv --output=data/processed/

"""
from docopt import docopt
import numpy as np
import pandas as pd
import requests, os
from sklearn.model_selection import train_test_split

# Split the data into train.csv and test.csv
def preprocess_data(raw_data_path, out_folder):
    if not (os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Data cleaning
    df = pd.read_csv(raw_data_path)
    df = df.query('not reviews_per_month.isnull()')
    train_df, test_df = train_test_split(df, test_size=0.8, random_state=573)
    latest_day = max(pd.to_datetime(train_df["last_review"]))
    train_df["days_from_last_review"] = (
        latest_day - (pd.to_datetime(train_df["last_review"]))
    ).dt.days
    test_df["days_from_last_review"] = (
        latest_day - (pd.to_datetime(test_df["last_review"]))
    ).dt.days
    train_df.to_csv(out_folder + 'train_cleaned.csv')
    test_df.to_csv(out_folder + 'test_cleaned.csv')





if __name__ == '__main__':
    arguments = docopt(__doc__)

    input_path = arguments['--input'] # Points to the raw dataset
    out_folder = arguments['--output'][0]

    # Tests arguments
    assert input_path.endswith('.csv')
    assert 'data' in out_folder and 'processed' in out_folder

    preprocess_data(input_path, out_folder)

    # Tests that the files are generated as expected
    assert os.path.exists('data/processed/test_cleaned.csv')
    assert os.path.exists('data/processed/train_cleaned.csv')

    print("-- Cleaned data available at: {}".format(out_folder))
