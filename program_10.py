#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
#
# This script servesa as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.
#
"""Due April 10, 2020
Created on Tue Apr 07 14:18:40 2020
by Hannah Walcek
Assignment 10 - Descriptive Statistics and Environmental Metrics

This program uses two text files WildcatCreek_Discharge_03335000_19540601-20200315.txt
and TippecanoeRiver_Discharge_03331500_19431001-20200315.txt. It cleans the
data and calculates various metrics including Tqmean, Richards-Baker Flashiness Index,
seven day low flow, and instances of 3x median for both monthly and annual
periods. The result is four files of monthly metrics, annual metrics, monthly averages,
and annual averages for both rivers.
"""

import pandas as pd
import scipy.stats as stats
import numpy as np

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # remove negative values
    DataDF.loc[~(DataDF['Discharge']>0), 'Discharge'] = np.nan
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
       
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    # clip data to whatever time span
    DataDF = DataDF.loc[startDate:endDate]
    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    
    # drop nan values
    drop_Tqmean = Qvalues.dropna()
    
    # find mean of array
    mean_streamflow = np.mean(drop_Tqmean)
    
    # create array of values greater than mean
    exceed = drop_Tqmean[drop_Tqmean>mean_streamflow]
    
    # find fraction unless drop_Tqmean is 0
    if len(drop_Tqmean) == 0:
        Tqmean = np.nan
    else:
        Tqmean = len(exceed)/len(drop_Tqmean)
        
    return ( Tqmean )

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""
    
    # drop nan values
    drop_RBindex = Qvalues.dropna()
    
    # calculate flowchange
    flowchange = drop_RBindex.diff()
    flowchange = flowchange.dropna()
    
    # calculate absolute value of flowchange
    abstQ = abs(flowchange)
    total_flowchange = abstQ.sum()
    
    # sum of flowchange
    total = drop_RBindex.sum()
    
    # caluclate RBindex
    RBindex = total_flowchange/total
        
    return ( RBindex )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
    
    # drop nan values
    drop_7Q = Qvalues.dropna()
    
    # use rolling mean to find lowest average flow with 7 day window
    val7Q = drop_7Q.rolling(window=7).mean().min()
    
    return ( val7Q )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""
    
    # drop nan values
    drop_median3x = Qvalues.dropna()
    
    # find median of array
    mean_streamflow = np.median(drop_median3x)
    
    # create array of values greater than mean
    greater = drop_median3x[drop_median3x>(3*mean_streamflow)]
    
    # find fraction
    median3x = len(greater)
    
    return ( median3x )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    
    # name columns
    annual_columns = ['Mean Flow', 'Peak Flow', 'Median Flow', 'Coeff Var', 'Skew', 'Tqmean', 'R-B Index', '7Q', '3xMedian']
    WYDataDF = pd.DataFrame(columns = annual_columns)
    
    # calculations for columns using water year
    WYDataDF['Mean Flow'] = DataDF['Discharge'].resample("AS-OCT").mean()
    WYDataDF['Peak Flow'] = DataDF['Discharge'].resample("AS-OCT").max()
    WYDataDF['Median Flow'] = DataDF['Discharge'].resample("AS-OCT").median()
    WYDataDF['Coeff Var'] = (DataDF['Discharge'].resample("AS-OCT").std()/WYDataDF['Mean Flow']) *100
    WYDataDF['Skew'] = DataDF['Discharge'].resample("AS-OCT").apply(stats.skew)
    WYDataDF['Tqmean'] = DataDF['Discharge'].resample("AS-OCT").apply(CalcTqmean)
    WYDataDF['R-B Index'] = DataDF['Discharge'].resample("AS-OCT").apply(CalcRBindex)
    WYDataDF['7Q'] = DataDF['Discharge'].resample("AS-OCT").apply(Calc7Q)
    WYDataDF['3xMedian'] = DataDF['Discharge'].resample("AS-OCT").apply(CalcExceed3TimesMedian)
    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
    
    # name columns
    monthly_columns = ['Mean Flow', 'Coeff Var', 'Tqmean', 'R-B Index']
    
    # calculations for columns
    MoDataDF = pd.DataFrame(columns = monthly_columns)
    
    MoDataDF['Mean Flow'] = DataDF['Discharge'].resample("M").mean()
    MoDataDF['Coeff Var'] = DataDF['Discharge'].resample("M").std()/MoDataDF['Mean Flow']*100
    MoDataDF['Tqmean'] = DataDF['Discharge'].resample("M").apply(CalcTqmean)
    MoDataDF['R-B Index'] = DataDF['Discharge'].resample("M").apply(CalcRBindex)
    
    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    
    # average annual dataframe
    AnnualAverages = WYDataDF.mean(axis = 0)
    
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    # average monthly dataframe
    MonthlyAverages = MoDataDF
    MonthlyAverages['Date'] = MonthlyAverages.index.month 
    MonthlyAverages = MoDataDF.groupby('Date').mean()
   
    return( MonthlyAverages )

# calculate monthly and annual stats for Wildcat
DataDF, MissingValues = ReadData("WildcatCreek_Discharge_03335000_19540601-20200315.txt")
DataDF, MissingValues = ClipData(DataDF, '1969-10-01', '2019-09-30')
Wildcat_WYDataDF = GetAnnualStatistics(DataDF)
Wildcat_WYDataDF = Wildcat_WYDataDF.assign(Station = 'Wildcat')
Wildcat_MoDataDF = GetMonthlyStatistics(DataDF)
Wildcat_MoDataDF = Wildcat_MoDataDF.assign(Station = 'Wildcat')

# calculate monthly and annual stats for Tippe
DataDF, MissingValues = ReadData("TippecanoeRiver_Discharge_03331500_19431001-20200315.txt")
DataDF, MissingValues = ClipData(DataDF, '1969-10-01', '2019-09-30')
Tippe_WYDataDF = GetAnnualStatistics(DataDF)
Tippe_WYDataDF = Tippe_WYDataDF.assign(Station = 'Tippe')
Tippe_MoDataDF = GetMonthlyStatistics(DataDF)
Tippe_MoDataDF = Tippe_MoDataDF.assign(Station = 'Tippe')

# create Annual_Metrics.csv
Annual_Metrics = Wildcat_WYDataDF
Annual_Metrics = Annual_Metrics.append(Tippe_WYDataDF)
Annual_Metrics.to_csv('Annual_Metrics.csv', sep="\t", index =True)

# create Monthly_Metrics.csv
Monthly_Metrics = Wildcat_MoDataDF
Monthly_Metrics = Monthly_Metrics.append(Tippe_MoDataDF)
Monthly_Metrics.to_csv('Monthly_Metrics.csv', sep="\t", index=True)

# calculate annual averages for both rivers
Wildcat_AnnualAverages = GetAnnualAverages(Wildcat_WYDataDF)
Wildcat_AnnualAverages = Wildcat_AnnualAverages.to_frame()
Wildcat_AnnualAverages = Wildcat_AnnualAverages.transpose()
Wildcat_AnnualAverages = Wildcat_AnnualAverages.assign(Station = 'Wildcat')
Tippe_AnnualAverages = GetAnnualAverages(Tippe_WYDataDF)
Tippe_AnnualAverages = Tippe_AnnualAverages.to_frame()
Tippe_AnnualAverages = Tippe_AnnualAverages.transpose()
Tippe_AnnualAverages = Tippe_AnnualAverages.assign(Station = 'Tippe')

# create Average_Annual_Metrics.txt
Average_Annual_Metrics = Wildcat_AnnualAverages
Average_Annual_Metrics = Average_Annual_Metrics.append(Tippe_AnnualAverages)
Average_Annual_Metrics.to_csv('Average_Annual_Metrics.txt', sep = "\t", index = None)

# calculate monthly averages for both rivers
Wildcat_MonthlyAverages = GetMonthlyAverages(Wildcat_MoDataDF)
Wildcat_MonthlyAverages = Wildcat_MonthlyAverages.assign(Station = 'Wildcat')
Tippe_MonthlyAverages = GetMonthlyAverages(Tippe_MoDataDF)
Tippe_MonthlyAverages = Tippe_MonthlyAverages.assign(Station = 'Tippe')

# create Average_Monthly_Metrics.txt
Average_Monthly_Metrics = Wildcat_MonthlyAverages
Average_Monthly_Metrics = Average_Monthly_Metrics.append(Tippe_MonthlyAverages)
Average_Monthly_Metrics.to_csv('Average_Monthly_Metrics.txt', sep = "\t", index = None)


# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
        