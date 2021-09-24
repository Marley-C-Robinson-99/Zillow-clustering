###################################         IMPORTS        ###################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date
import os
import env

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


###################################         ACQUIRE        ###################################
# get database url for acquiring data
def get_db_url(db):
    ''' 
    Function to acquire database url using hostid, username, and password from
    my environment file
    '''
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{db}'
    return url


# acquire zillow data using the SQL query
def acquire_zillow():
    ''' 
    Pull data from MySQL database and cache file unless file is already present
    '''
    url = get_db_url('zillow')
    filename = 'raw_zillow.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        sql = """
        SELECT prop.*,
            pred.logerror,
            pred.transactiondate,
            air.airconditioningdesc,
            arch.architecturalstyledesc,
            build.buildingclassdesc,
            heat.heatingorsystemdesc,
            landuse.propertylandusedesc,
            story.storydesc,
            construct.typeconstructiondesc
        FROM properties_2017 prop
            INNER JOIN (SELECT parcelid,
                   Max(transactiondate) transactiondate
                   FROM   predictions_2017
                   GROUP  BY parcelid) pred
            USING (parcelid)
        JOIN predictions_2017 as pred USING (parcelid, transactiondate)
        LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
        LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
        LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
        LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
        LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
        LEFT JOIN storytype story USING (storytypeid)
        LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
        WHERE  prop.latitude IS NOT NULL
            AND prop.longitude IS NOT NULL
        """
        df = pd.read_sql(sql, url, index_col='id')
        df.to_csv(filename, index=False)
        return df

###################################         PREPARE HELPER FUNCS        ###################################

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
    '''
    Helper function to drop rows or columns based on the percent of values that are missing
    from cols and rows given a specified threshold.
    '''
    threshold = int(round(prop_required_column*len(df.index),0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def remove_columns(df, cols_to_remove):
	'''
    Helper function to drop a list of columns from a dataframe
    '''
    df = df.drop(columns=cols_to_remove)
    return df

def outlier_function(df, cols, k):
	'''
    Helper function to detect and handle oulier using IQR rule
    '''
    for col in df[cols]:
        q1 = df.annual_income.quantile(0.25)
        q3 = df.annual_income.quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df

def month_sold(df):
    '''
    Creates a month sold feature based on sale_date
    '''
    # Converting date to string for splitting in month_sold function
    df['sale_date'] = df['sale_date'].astype(str)

    # Splitting the date to select month into a df
    month_sold = df.sale_date.str.split(pat ='-', expand = True)
    
    # Creating month col
    df['month_sold'] = month_sold[1] # grabs date col of df
    
    # Replaces month numbers with strings
    # df['month_sold'] = df['month_sold'].replace(['05', '06', '07', '08'], ['may', 'jun', 'jul', 'aug'])
    
    # Recasting sale_date as datetime int
    df['sale_date'] = df['sale_date'].astype('O')
    df['month_sold'] = df['month_sold'].astype(int)
    return df

def yearly_tax(df):
    ''' 
    Creates a rounded yearly_tax feature.
    Equation = tax value / (current year - year built)
    '''
    # Getting current year
    curr_year = int(f'{str(date.today())[:4]}')

    # Creating column
    df['yearly_tax'] = df.tax_value / (curr_year - df.year_built)

    df.yearly_tax = round(df.yearly_tax.astype(float), 0)

def impute_year(df):   
    '''
    Helper function to impute year built using mode
    '''
    imp = SimpleImputer(strategy='most_frequent')  # build imputer

    imp.fit(df[['yearbuilt']]) # fit to df of col yearbuilt

    # transform the data
    df[['year_built']] = imp.transform(df[['yearbuilt']])
    return df

def tax_rate(df):
    '''
    Creates a tax_rate column by calculating taxamount(yearly)/tax_rate(total) 
    '''
    # Calc tax_rate
    tax_rate =  df.yearly_tax / df.tax_value
    
    # Creating col
    df['tax_rate'] = tax_rate
    
    return df
###################################         PREPARE/SPLIT         ###################################

def wrangle_zillow():
    '''
    Utilizes acquire_zillow function to pull raw dataframe
    and then passes it through a number of functions to impute missing vals,
    drop nulls, and rename cols where needed.
    '''
    
    df = acquire_zillow()

    # Restrict df to only properties that meet single use criteria
    single_use = [261, 262, 263, 264, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]

    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]

    # Impute missing vals in yearbuilt
    df = impute_year(df)

    # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)

    # Add column for counties
    df['county'] = df['fips'].apply(
        lambda x: 'Los Angeles' if x == 6037\
        else 'Orange' if x == 6059\
        else 'Ventura')

    # drop unnecessary columns
    dropcols = ['parcelid',
         'calculatedbathnbr',
         'finishedsquarefeet12',
         'fullbathcnt',
         'heatingorsystemtypeid',
         'propertycountylandusecode',
         'propertylandusetypeid',
         'propertyzoningdesc',
         'censustractandblock',
         'propertylandusedesc',
         'fips',
         'yearbuilt']
    
    df = remove_columns(df, dropcols)
    
    # Rename cols for understandability
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'home_area',
                              'taxvaluedollarcnt':'tax_value', 
                              'transactiondate':'sale_date',
                              'taxamount':'yearly_tax',
                              'rawcensustractandblock':'census_tb',
                              'structuretaxvaluedollarcnt':'structure_tax',
                              'landtaxvaluedollarcnt':'land_tax_value',
                              'heatingorsystemdesc':'heating_type',
                              'regionidzip':'zip_code',
                              'regionidcity':'city_id',
                              'regionidcounty':'county_id',
                              'lotsizesquarefeet':'lot_area',
                              'buildingqualitytypeid':'quality_id',
                              'assessmentyear':'year_assessed'})

    # Create tax_rate col
    df = tax_rate(df)

    # Create Month_sold col
    month_sold(df)

    # replace nulls in unitcnt with 1
    df.unitcnt.fillna(1, inplace = True)

    # assume that since this is Southern CA, null means 'None' for heating system
    df.heating_type.fillna('None', inplace = True)

    # replace nulls with median values for select columns
    df.lot_area.fillna(7313, inplace = True)
    df.quality_id.fillna(6.0, inplace = True)

    # Columns to look for outliers
    df = df[df.tax_value < 5_000_000]
    df = df[df.home_area < 8000]

    # Removing outliers with func
    # df = outlier_function(df, [outlier_cols], k)

    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()

    return df

def train_validate_test_split(df, target = None, seed=1312):
    '''
    ---------------------------------------------------
    This function takes a dataframe, optionally the name of the target variable
    (for X/y spit if one is provided), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    ---------------------------------------------------

    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 

    ---------------------------------------------------

    Output:
        - The function returns, in this order, train, validate and test dataframes. 
          Or, if target is provided, in this order, X_train, X_validate, X_test, y_train, y_validate, y_test
    '''

    if target == None:
        train_validate, test = train_test_split(df, test_size=0.2, 
                                                random_state=seed)
        train, validate = train_test_split(train_validate, test_size=0.3, 
                                                random_state=seed)
        return train, validate, test
    else:
        X = df.drop(columns = target)
        y = df[target]
        X_train_validate, X_test, y_train_validate, y_test  = train_test_split(X, y, 
                                    test_size=0.2, 
                                    random_state=seed)
        X_train, X_validate, y_train, y_validate  = train_test_split(X_train_validate, y_train_validate, 
                                    test_size=.25, 
                                    random_state=seed)
        return X_train, X_validate, X_test, y_train, y_validate, y_test

###################################         DATA SUMMARY        ###################################

def nulls_by_col(df):
    '''
    Determines the number of nulls for each column
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

def nulls_by_row(df):
    '''
    Determines the number of nulls for each row
    '''
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'customer_id': 'num_rows'}).reset_index()
    return rows_missing

def col_desc(df):
    '''
    Preforms a .describe on a dataframe as well as adding the range of each col
    '''
    stats_df = df.describe().T
    stats_df['range'] = stats_df['max'] - stats_df['min']
    return stats_df

def summarize(df):
    '''
    Function to summarize a dataframe, will take in a single argument (a pandas dataframe)
    and output to console various statistics on said dataframe, including:
    - .info()
    - .describe()
    - .value_counts()
    - Observation of nulls in the dataframe
    - Column stats
    '''
    print('=====================================================\n\n')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    print(df.describe())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================\n\n')
    print('Dataframe Value Counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
    print('=====================================================\n\n')
    print('Nulls by Column: ')
    print(nulls_by_col(df))
    print('=====================================================\n\n')
    print('Nulls by Row: ')
    print(nulls_by_row(df))
    print('=====================================================\n\n')
    print('Column Stats: ')
    print(col_desc(df))
    print('============================================')
    