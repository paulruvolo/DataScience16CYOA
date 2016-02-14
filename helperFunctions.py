import pandas as pd 
from datetime import datetime

def recodeData(df, isTrain = False):
	'''This function takes in the dataframe that we get from loading in the 
	SF crime data and returns a re-coded dataframe that has all the 
	additional features we want to add and the categorical features recoded 
	and cleaned.
	'''

	#since the modifications are done in-place we don't return the dataframe. 
	#we do, however, return the list of all the columns we added.
	newDate = recodeDates(df)
	newDistrict = recodePoliceDistricts(df)

	addedColumns = newDate + newDistrict

	if (isTrain):
		columnsToDrop = ['Descript', 'Resolution']
		df.drop(columnsToDrop, axis=1, inplace=True)

	return df, addedColumns

	

def recodeDates(df):
	'''This function takes in a dataframe and recodes the date field into 
	useable values. Here, we also recode the day of week.'''
	#Recode the dates column to year, month, day and hour columns
	df['DateTime'] = df['Dates'].apply(
		lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

	df['Year'] = df['DateTime'].apply(lambda x: x.year)
	df['Month'] = df['DateTime'].apply(lambda x: x.month)
	df['Day'] = df['DateTime'].apply(lambda x: x.day)
	df['Hour'] = df['DateTime'].apply(lambda x: x.hour)

	#Recode the day of week into a number
	daysOfWeek =['Sunday', 'Monday', 'Tuesday', 'Wednesday', 
             'Thursday', 'Friday', 'Saturday']
	df['DayOfWeekRecode'] = df['DayOfWeek'].apply(
		lambda x: daysOfWeek.index(x))

	return ['Year', 'Month', 'Day', 'Hour', 'DayOfWeekRecode']

def recodePoliceDistricts(df):
	'''This function recodes the police district to a one-hot encoding 
	scheme.'''
	districts = df['PdDistrict'].unique().tolist()
	newColumns = []
	for district in districts:

		newColumns.append('District' + district)
		df['District' + district] = df['PdDistrict'].apply(
			lambda x: int(x == district))

	return newColumns

# def recodeCrimeCategory(df):



