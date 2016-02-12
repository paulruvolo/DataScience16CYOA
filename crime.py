import pandas as pd
import numpy as np

def load_train():
    """
    Load the train dataset
    """
    return pd.read_csv('train.csv')

def load_test():
    """
    Load the test dataset
    """
    return pd.read_csv('test.csv')

def clean_data(df):
    """
    Clean the given data frame
    """
    # Create Year, Month, Day, Hour, and Minute columns (seconds are always 00)
    df['Year'] = df.Dates.apply(lambda x: int(x.split(' ')[0].split('-')[0]))
    df['Month'] = df.Dates.apply(lambda x: int(x.split(' ')[0].split('-')[1]))
    df['Day'] = df.Dates.apply(lambda x: int(x.split(' ')[0].split('-')[2]))
    df['Hour'] = df.Dates.apply(lambda x: int(x.split(' ')[1].split(':')[0]))
    df['Minute'] = df.Dates.apply(lambda x: int(x.split(' ')[1].split(':')[1]))

    # Encode DayOfWeek as integers
    day_of_week = {
        'Sunday': 0,
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6
    }
    df['DoW'] = df.DayOfWeek.replace(day_of_week)

    # Encode PdDistrict as integers
    pd_district = {
        'NORTHERN': 0,
        'PARK': 1,
        'INGLESIDE': 2,
        'BAYVIEW': 3,
        'RICHMOND': 4,
        'CENTRAL': 5,
        'TARAVAL': 6,
        'TENDERLOIN': 7,
        'MISSION': 8,
        'SOUTHERN': 9
    }
    df['PdD'] = df.PdDistrict.replace(pd_district)

    # Set the invalid X and Y values to the district medians
    pdd_x = [df.X[df.PdD == i].median() for i in range(10)]
    pdd_y = [df.Y[df.PdD == i].median() for i in range(10)]
    df.loc[df.X == -120.5, 'X'] = df.PdD[df.X == -120.5].replace(pdd_x)
    df.loc[df.Y == 90, 'Y'] = df.PdD[df.Y == -120.5].replace(pdd_y)

    # Encode Category as integers
    categories = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY',
       'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE',
       'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
       'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING',
       'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING',
       'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES',
       'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE',
       'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE',
       'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE',
       'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT',
       'WARRANTS', 'WEAPON LAWS']
    if 'Category' in df.columns:
        df['CategoryNumber'] = df.Category.apply(lambda x: categories.index(x))

    return df

def load_cleaned_train():
    return clean_data(load_train())

def load_cleaned_test():
    return clean_data(load_test())

def logloss(y, p):
    """
    Given the correct categories (pandas series y: N by 1) encoded as integers
    and predictions (numpy array p: N by M) where N is the number of trials
    M is number of possible categories, return the multi-class logarithmic
    loss as described here: https://www.kaggle.com/c/sf-crime/details/evaluation
    """
    # Normalize each row and avoid extremes
    p /= p.sum(axis=1)[:,None]
    p = p.clip(1e-15, 1 - 1e-15)

    # Calculate logloss
    logloss = 0
    for i in range(len(p)):
        logloss += np.log(p[i, y.iloc[i]])
    logloss /= float(-len(p))

    return logloss
