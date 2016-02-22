import pandas as pd
import numpy as np

CATEGORIES = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY',
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

def load_train():
    """
    Load the train dataset
    """
    try:
        return pd.read_csv('train.csv')
    except:
        return pd.read_csv('../train.csv')

def load_test():
    """
    Load the test dataset
    """
    try:
        return pd.read_csv('test.csv')
    except:
        return pd.read_csv('../test.csv')

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

    # Create columns indicating potentially bogus reports filed at 00:01 and 12:00
    df['BogusReport'] = df.apply(lambda x: x.Hour == 0 and x.Minute == 1, axis=1)
    df['NBogusReport'] = df.apply(lambda x: x.Hour == 12 and x.Minute == 0, axis=1)

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

    # Encode Time of Day
    df['Morning'] = df.Hour.apply(lambda x: int(x >= 5 and x < 12))
    df['Afternoon'] = df.Hour.apply(lambda x: int(x >= 12 and x < 17))
    df['Evening'] = df.Hour.apply(lambda x: int(x >= 17 and x < 24))
    df['Night'] = df.Hour.apply(lambda x: int(x >= 0 and x < 5))

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

    # Create one-hot encoding of PdDistrict
    df = pd.concat((df, pd.get_dummies(df['PdD'], prefix='PdD')), axis=1)

    # Set the invalid X and Y values to the district medians
    pdd_x = {i: df.X[df.PdD == i].median() for i in range(10)}
    pdd_y = {i: df.Y[df.PdD == i].median() for i in range(10)}
    df.loc[df.X == -120.5, 'X'] = df.PdD[df.X == -120.5].replace(pdd_x)
    df.loc[df.Y == 90, 'Y'] = df.PdD[df.Y == 90].replace(pdd_y)

    # Encode Category as integers
    if 'Category' in df.columns:
        df['CategoryNumber'] = df.Category.apply(lambda x: CATEGORIES.index(x))

    
    #Encode information from addresses
    on_corner = []

    for address in df.Address:
        on_corner.append(int('/' in address))
    df['CornerCrime'] = on_corner

    streets = {}
    for element in df.Address:
        address = element.split(' / ')
        streets[address[0]] = streets.get(address[0],0)+1
        try:
            streets[address[1]] = streets.get(address[1],0)+1
        except:
            pass

    crime_streets = sorted(streets,key=streets.get,reverse=True)[0:1]

    count = 0
    for street in crime_streets:
        label = 'ST_'+str(count)
        df[label] = [int(street in element) for element in df.Address]
        count += 1

    # Drop unnecessary colummns
    df.drop(['Dates', 'DayOfWeek', 'PdDistrict', 'Address'], inplace=True, axis=1)
    try:
        df.drop(['Descript', 'Resolution'], inplace=True, axis=1)
    except ValueError:
        pass  # the test data won't have these columns

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

def create_submission(alg, X_train, y_train, X_test, predictors, filename):
    alg.fit(X_train[predictors], y_train)
    predictions = alg.predict_proba(X_test[predictors])

    submission = pd.DataFrame({
        'Id': X_test.Id
    })
    for i in range(predictions.shape[1]):
        submission[CATEGORIES[i]] = predictions[:,i]
    submission.to_csv(filename, index=False)

if __name__ == '__main__':
    test = load_cleaned_test()
    test.info()
