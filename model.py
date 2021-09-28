import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

from wrangle import train_validate_test_split
from explore import scaling, encode_cols
from scipy import stats

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.feature_selection import f_regression 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures


import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import math

def lin_reg(X_train, X_validate, X_test, y_train, y_validate, y_test, x, y, model_type = LinearRegression(normalize = True), show_metrics = True):
    '''
    Creates and fits a linear regression model given 
    a driver and target variable.
    
    Parameters
    ----------------
    - x : driver variable(s)

    - y : target variable

    - model_type : Default is LinearRegression (OLM) with normalize = True, 
            can use 3 of sklearns linear_models, LinearRegression(OLS), LassoLars, TweedieRegressor (GLM)
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Outputs
    -------------------
    Returns a predictions dataframe for train, validate and test with columns:
    - x variable(s)
    - y actual
    - ŷ (predictions of the target (Y) based upon driver (X))
    - mean_baseline (baseline predictions of Y)
    - residuals between y actual and ŷ
    - residuals squared
    - residual baseline
    - residual baseline squared
    '''

    # Create model
    model = model_type
    
    # Creating y_train and validate dfs with neccesary cols
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    # Scaling X_train validate and test
    scaler, X_train_scaled, X_validate_scaled, X_test_scaled = scaling(X_train, X_validate, X_test)
    for feature in x:
        X_train_scaled[f'{feature}_train'] = X_train_scaled[feature]
        X_validate_scaled[f'{feature}_validate'] = X_validate_scaled[feature]
        X_test_scaled[f'{feature}_test'] = X_test_scaled[feature]
        X_train_scaled.drop(columns = [feature])
        X_validate_scaled.drop(columns = [feature])
        X_test_scaled.drop(columns = [feature])


    # Creating train df
    dft = pd.DataFrame()
    for feature in x:
        dft[f'{feature}_train'] = X_train_scaled[f'{feature}_train']
    

    # Creating validate df
    dfv = pd.DataFrame()
    for feature in x:
        dfv[f'{feature}_validate'] = X_validate_scaled[f'{feature}_validate']
    
    # Creating test df
    dfs = pd.DataFrame()
    for feature in x:
        dfs[f'{feature}_test'] = X_test_scaled[f'{feature}_test']

    # Creating ŷ predictions column
    if model == PolynomialFeatures():
        X_train_scaled = model.fit_transform(dft)
        X_validate_scaled = model.transform(dfv)
        X_test_scaled = model.transform(dfs)

        lm = LinearRegression(normalize=True)
        lm.fit(dft, y_train[y])
        dft['yhat_train'] = lm.predict(dft)
        yhat_train = dft['yhat_train']
        dft[f'{y}_train'] = y_train[y]

        dfv['yhat_validate'] = lm.predict(dfv)
        yhat_validate = dfv['yhat_validate']
        dfv[f'{y}_validate'] = y_validate[y]

        dfs['yhat_test'] = lm.predict(dfs)
        yhat_test = dfs['yhat_test']
        dfs[f'{y}_test'] = y_test[y]

        model = lm
    
    model.fit(dft, y_train[y])

    dft['yhat_train'] = model.predict(dft)
    yhat_train = dft['yhat_train']
    dft[f'{y}_train'] = y_train[y]

    dfv['yhat_validate'] = model.predict(dfv)
    yhat_validate = dfv['yhat_validate']
    dfv[f'{y}_validate'] = y_validate[y]

    dfs['yhat_test'] = model.predict(dfs)
    yhat_test = dfs['yhat_test']
    dfs[f'{y}_test'] = y_test[y]

    # Create baseline predictions column
    dft['yhat_baseline'] = dft[f'{y}_train'].mean()
    dfv['yhat_baseline'] = dft[f'{y}_train'].mean()
    dfs['yhat_baseline'] = dft[f'{y}_train'].mean()
    # Creating yhat_train and yhat_validate mean and median
    dfv['yhat_median'] = dft['yhat_train'].median()
    dft['yhat_median'] = dft['yhat_train'].median()
    dfs['yhat_median'] = dft['yhat_train'].median()
    # residual train df
    dft['residuals'] = residuals(dft[f'{y}_train'], yhat_train)
    dft['residual^2'] = residuals(dft[f'{y}_train'], yhat_train) ** 2
    dft['residual_baseline'] = residuals(dft[f'{y}_train'], dft['yhat_baseline'])
    dft['residual_baseline^2'] = residuals(dft[f'{y}_train'], dft['yhat_baseline']) ** 2
    # residual validate df
    dfv['residuals'] = residuals(dfv[f'{y}_validate'], yhat_validate)
    dfv['residual^2'] = residuals(dfv[f'{y}_validate'], yhat_validate) ** 2
    dfv['residual_baseline'] = residuals(dfv[f'{y}_validate'], dfv['yhat_baseline'])
    dfv['residual_baseline^2'] = residuals(dfv[f'{y}_validate'], dfv['yhat_baseline']) ** 2   
    # residual test df
    dfs['residuals'] = residuals(dfs[f'{y}_test'], yhat_test)
    dfs['residual^2'] = residuals(dfs[f'{y}_test'], yhat_test) ** 2
    dfs['residual_baseline'] = residuals(dfs[f'{y}_test'], dfs['yhat_baseline'])
    dfs['residual_baseline^2'] = residuals(dfs[f'{y}_test'], dfs['yhat_baseline']) ** 2 
    
    df_eval, df_sig = get_metrics(dft, dfv, dfs, x, y)

    train_pred = dft
    validate_pred = dfv
    test_pred = dfs
    if show_metrics == True:
        return train_pred, validate_pred, test_pred, df_eval, df_sig
    else:
        return train_pred, validate_pred, test_pred

def get_metrics(dft, dfv, dfs, x, y, *model_type):
    ''' 
    Takes in scaled train, validate and test dataframes from my linear regression
    funcion and driver vars (x) and a target var(y). 

    Outputs 2 dataframes:
    - Evaluation df with evaluation metrics such as RMSE, MSE and SSE
    - Model significance df with metrics for determining significance such as r^2 score.
    '''
    
    # helper var assignmert
    yt = dft[f'{y}_train']
    xt = {}
    for i, feature in enumerate(x):
        xt['xt{0}'.format(i)] = dft[f'{feature}_train']
    
    yv = dfv[f'{y}_validate']
    xv = {}
    for i, feature in enumerate(x):
        xv['xv{0}'.format(i)] = dfv[f'{feature}_validate']
    
    ys = dfs[f'{y}_test']
    xs = {}
    for i, feature in enumerate(x):
        xs['xs{0}'.format(i)] = dfs[f'{feature}_test']

    
    yhat_train = dft['yhat_train']
    yhat_validate = dfv['yhat_validate']
    yhat_test = dfs['yhat_test']
    yhat_baseline = dft['yhat_baseline']
    
    # eval math var assignment
    SSE_train = sse(yt,yhat_train)
    MSE_train = mse(yt,yhat_train)
    RMSE_train = rmse(yt,yhat_train)
    SSE_validate = sse(yv,yhat_validate)
    MSE_validate = mse(yv,yhat_validate)
    RMSE_validate = rmse(yv,yhat_validate)
    SSE_test = sse(ys,yhat_test)
    MSE_test = mse(ys,yhat_test)
    RMSE_test = rmse(ys,yhat_test)
    # eval baseline vars
    SSE_baseline = sse(yt, yhat_baseline)
    MSE_baseline = mse(yt, yhat_baseline)
    RMSE_baseline = rmse(yt, yhat_baseline)

    # eval df
    df_eval = pd.DataFrame(np.array(['SSE_train','MSE_train','RMSE_train', 'SSE_validate', 'MSE_validate', 'RMSE_validate', 'SSE_test', 'MSE_test', 'RMSE_test']), columns=['metric'])
    df_eval['model_error'] = np.array([SSE_train, MSE_train, RMSE_train, SSE_validate, MSE_validate, RMSE_validate, SSE_test, MSE_test, RMSE_test])
    # baseline eval df
    df_baseline_eval = pd.DataFrame(np.array(['SSE_baseline','MSE_baseline','RMSE_baseline', 'SSE_baseline', 'MSE_baseline', 'RMSE_baseline', 'SSE_baseline', 'MSE_baseline', 'RMSE_baseline']), columns=['metric'])
    df_baseline_eval['model_error'] = np.array([SSE_baseline, MSE_baseline, RMSE_baseline, SSE_baseline, MSE_baseline, RMSE_baseline, SSE_baseline, MSE_baseline, RMSE_baseline])

    # train error delta
    df_eval['error_delta'] = df_eval.model_error - df_baseline_eval.model_error

    # model significance vars
    ESS_train = ess(yt,yhat_train)
    R2_train = r2_score(yt,yhat_train)
    TSS = tss(yt)
    ESS_validate = ess(yv,yhat_validate)
    R2_validate = r2_score(yv,yhat_validate)
    ESS_test = ess(ys,yhat_test)
    R2_test = r2_score(ys,yhat_test)


    ESS_baseline = ess(yt, yhat_baseline)
    TSS_baseline = ESS_baseline + SSE_baseline
    R2_baseline = r2_score(yt, yhat_baseline)

    # model significance df
    df_sig = pd.DataFrame(np.array(['ESS_train', 'R2_train', 'TSS', 'ESS_validate', 'R2_validate', 'ESS_test', 'R2_test']), columns = ['metric'])
    df_sig['model_significance'] = np.array([ESS_train, R2_train, TSS, ESS_validate, R2_validate, ESS_test, R2_test])
    # model baseline significance df
    df_baseline_sig = pd.DataFrame(np.array(['ESS_baseline', 'TSS_baseline', 'R^2_baseline']), columns = ['metric'])
    df_baseline_sig['model_significance'] = np.array([ESS_baseline, TSS_baseline, R2_baseline])

    df_eval = pd.concat([df_eval, df_baseline_eval], axis = 0)
    df_sig = pd.concat([df_sig, df_baseline_sig], axis = 0)
    
    return df_eval, df_sig


def plot_actual_vs_pred(y_actual, yhat_baseline, yhat_model1, yhat_model2, yhat_model3):
    plt.figure(figsize=(16,8))
    plt.plot(y_actual, yhat_baseline, alpha=1, color="gray", label='_nolegend_')
    plt.annotate("Baseline: Predict Using Mean", (16, 9.5))
    plt.plot(y_actual, y_actual, alpha=1, color="blue", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (.5, 3.5), rotation=15.5)

    plt.scatter(y_actual, yhat_model1, 
            alpha=.5, color="red", s=100, label="Model: LinearRegression")
    plt.scatter(y_actual, yhat_model2, 
            alpha=.5, color="yellow", s=100, label="Model: Lasso + Lars")
    plt.scatter(y_actual, yhat_model3, 
            alpha=.5, color="green", s=100, label="Model: TweedieRegressor")
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()

def plot_errors(y_actual, yhat_baseline, yhat_model1):
    plt.figure(figsize=(16,8))
    plt.axhline(label="No Error")
    plt.scatter(y_actual.sample(200), (yhat_model1 - y_actual).sample(200), 
           alpha=.5, color="red", s=100, label="Model: LinearRegression")
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Residual/Error: Predicted - Actual")
    plt.title("Plotting Errors in Predictions")
    plt.show()
###################################         MATH EVAL         ###################################

def residuals(y, ŷ = 'yhat', df = None):
    if df == None:
        return ŷ - y
    else:
        return df[ŷ] - df[y]

def sse(y, ŷ):
    return (residuals(y, ŷ) **2).sum()

def mse(y, ŷ):
    n = y.shape[0]
    return sse(y, ŷ) / n

def rmse(y, ŷ):
    return math.sqrt(mse(y, ŷ))

def ess(y, ŷ):
    return ((ŷ - y.mean()) ** 2).sum()

def tss(y):
    return ((y - y.mean()) ** 2).sum()

def r2_score(y, ŷ):
    return ess(y, ŷ) / tss(y)

def t_stat(corr):
    t = (corr * sqrt(n - 2)) / sqrt(1 - corr**2)
    return t

def regression_errors(y, ŷ):
    return pd.Series({
        'sse': sse(y, ŷ),
        'ess': ess(y, ŷ),
        'tss': tss(y),
        'mse': mse(y, ŷ),
        'rmse': rmse(y, ŷ),
    })

def baseline_mean_errors(y):
    predicted = y.mean()
    return {
        'sse': sse(y, ŷ),
        'mse': mse(y, ŷ),
        'rmse': rmse(y, ŷ),
    }

def better_than_baseline(y, ŷ, r2_actual, r2_baseline):
    rmse_baseline = rmse(y, y.mean())
    rmse_model = rmse(y, ŷ)
    if rmse_model < rmse_baseline:
        if r2_actual > r2_baseline:
            return "RMSE better than baseline\nR^2 better than baseline"
        elif r2_actual == r2_baseline:
            return "RMSE better than baseline\nR^2 same as baseline"
        else:
            return "RMSE better than baseline\nR^2 worse than baseline"
    elif rmse_model == rmse_baseline:
        return "Same as baseline"
    elif rmse_model > rmse_baseline:
        if r2_actual > r2_baseline:
            return "Worse than baseline\nR^2 better than baseline"
        elif r2_actual == r2_baseline:
            return "Worse than baseline\nR^2 same as baseline"
        else:
            return "Worse than baseline\nR^2 worse than baseline"
###################################         AUTO-FEATURE SELECTION    ###################################

def kbest_features(df, k, target, show_scores = True, scaler_type = MinMaxScaler()):
    '''
    Takes a dataframe and uses SelectKBest to select for
    the most relevant drivers of target.
    
    Parameters:
    -----------
    k : Number of features to select

    show_scores : If true, outputs a dataframe containing the top (k) features
            and their respective f-scores

    scaler_type : Default is StandardScaler, determines the type 
            of scaling applied to the df before
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Output:
    -------------
    features : A list of features of (k) length that SelectKBest has selected to be the
    main drivers of the target.

    fs_sorted : A sorted dataframe that combines each feature with its respective score.
    '''

    # only selects numeric cols and separates target
    X_train, X_validate, X_test, y_train, y_validate, y_test = train_validate_test_split(df, target = target)

    scaler, X_train_scaled, X_validate_scaled, X_test_scaled = scaling(X_train, X_validate, X_test)

    # creating SelectKBest object with {k} selected features
    kbest = SelectKBest(f_regression, k= k)
    
    # fitting object
    kbest.fit(X_train_scaled, y_train)
    
    # assigning features to var
    features = X_train.columns[kbest.get_support()]
    if show_scores == True:
        # getting feature scores
        scores = kbest.scores_[kbest.get_support()]

        # creating zipped list of feats and their scores
        feat_scores = list(zip(features, scores))
    
        fs_df = pd.DataFrame(data = feat_scores, columns= ['feat_names','F_Scores'])
    
        fs_sorted = fs_df.sort_values(['F_Scores'], ascending = [False])

        return fs_sorted
    else:
        return list(features)


def rfe_features(df, n, target, est_model = LinearRegression(), scaler_type = MinMaxScaler(), show_rank = True):
    '''
    Takes a dataframe and uses Recursive Feature Elimination to select for
    the most relevant drivers of target.
    
    Parameters:
    -----------
    n : Number of features to select
            
    est_model : Defailt is LinearRegression, determines the estimator
            used by the RFE function
    
    scaler_type : Default is StandardScaler, determines the type 
            of scaling applied to the df before
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Output:
    -------------
    A list of features of (n) length that RFE has selected to be the
    main drivers of the target.
    '''

    # train, validate, test split
    X_train, X_validate, X_test, y_train, y_validate, y_test = train_validate_test_split(df, target = target)
    
    # scaling data
    scaler, X_train_scaled, X_validate_scaled, X_test_scaled = scaling(X_train, X_validate, X_test)
    
    # creating RFE object with {n} selected features
    rfe = RFE(estimator= est_model, n_features_to_select=n)
    
    # fitting object
    rfe.fit(X_train_scaled, y_train)
    
    # assigning features to var
    features = X_train.columns[rfe.support_]
    
    if show_rank == True:
        # getting feature scores
        rank = rfe.ranking_

        # creating zipped list of feats and their scores
        feat_rank = list(zip(X_train.columns, rank))
    
        fr_df = pd.DataFrame(data = feat_rank, columns= ['feat_names','F_rank'])

        fr_sorted = fr_df.sort_values(['F_rank'], ascending = [True])

        return fr_sorted
    else:
        return list(features)