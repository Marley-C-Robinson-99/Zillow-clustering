import pandas as pd
import wrangle as wr

import re
import itertools
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
import numpy as np

###################################         HELPER FUNCS         ###################################

def separate_feats(df):
    ''' 
    Creates a combination of all possible features as well as separates quant vars and cat vars.

    Outputs, in this order, list of all features, categorical features, quantitative features
    '''

    cat_feats = []
    for col in df.columns:
        if (df[col].dtype == object) | (df[col].dtype == 'uint8') | (df[col].nunique() <= 12):
            cat_feats.append(col)

    # Filters for quantitative variables
    quant_feats = [col for col in df.columns if col not in cat_feats]

    # General features that are not objects
    feats = [col for col in df.columns]

    return feats, cat_feats, quant_feats


def pairing(df):
    ''' 
    Helper function for vizualizations, creates quant/cat var pairs. 
    Takes in a dataframe and outputs a list of unique pairs and quant/cat var pairings
    '''
    # Separating features
    feats, cat_feats, quant_feats = separate_feats(df)


    # Creating raw pairs of all features
    pairs = []
    pairs.extend(list(itertools.product(cat_feats, quant_feats)))
    pairs.extend(list(itertools.product(cat_feats, feats)))
    pairs.extend(list(itertools.product(quant_feats, feats)))
    
    # Whittling down pairs to unique combos of quant and cat vars
    unique_pairs = []
    cat_quant_pairs = []
    for pair in pairs:
        if pair[0] != pair[1]:
            if pair not in unique_pairs:
                if (pair[1], pair[0]) not in unique_pairs:
                    unique_pairs.append(pair)
                    if (pair[0] not in quant_feats) & (pair[1] not in cat_feats):
                        cat_quant_pairs.append(pair)


    return pairs, unique_pairs, cat_quant_pairs


def feature_combos(df):
    ''' 
    Creates a list of all possible feature combinations
    '''
    feats, cat_feats, quant_feats = separate_feats(df)
    combos = []
    for i in range(2, len(feats) + 1):
        combos.extend(list(itertools.combinations(feats, i)))
    
    return combos
    

def hist_combos(df):
    ''' Create a histplot for vars
    '''
    feats, cat_feats, quant_feats = separate_feats(df)
    
    
    
    for feat in feats:
        if (df[feat].dtype != object):
            plt.figure(figsize=(7, 5))
            # Title with column name.
            plt.title(feat)

            # Display histogram for column.
            df[feat].hist(bins=5)

            # Hide gridlines.
            plt.grid(False)
        
            # Hide scientific notation
            plt.ticklabel_format(style='plain')
            plt.tight_layout()
            plt.show()


def relplot_pairs(df):
    ''' 
    Plots each unique cat/quant pair using a relplot
    '''
    pairs, u_pairs, cq_pairs = pairing(df)
    for pair in u_pairs:
        sns.relplot(x= pair[0], y= pair[1], data=df)
        plt.title(f'{pair[0]} vs {pair[1]}')
        plt.show()
        print('_____________________________________________')


def lmplot_pairs(df):
    ''' 
    Plots each unique cat/quant pair using a lmplot
    '''
    pairs, u_pairs, cq_pairs = pairing(df)
    for pair in u_pairs:
        sns.lmplot(x= pair[0], y= pair[1], data=df, line_kws={'color': 'red'})
        plt.title(f'{pair[0]} vs {pair[1]}')
        plt.show()
        print('_____________________________________________')


def jointplot_pairs(df):
    ''' 
    Plots each unique cat/quant pair using a jointplot
    '''
    pairs, u_pairs, cq_pairs = pairing(df)
    for pair in u_pairs:
        sns.jointplot(x= pair[0], y= pair[1], data=df, kind = 'reg', height = 4, joint_kws = {'line_kws':{'color': 'red'}})
        plt.title(f'{pair[0]} vs {pair[1]}')
        plt.tight_layout()
        plt.show()
        print('_____________________________________________')


def pairplot_combos(df):
    ''' 
    Plots combinations of quant variables using pairplots where combo length greater or equal to 2
    '''

    # Get list of feats
    feats, cat_feats, quant_feats = separate_feats(df)
    quant = df[quant_feats]

    # From quantitative vars get combos to create pairplots of
    combos = feature_combos(quant)
    for combo in combos:
        plt.figure(figsize=(5,5))
        sns.pairplot(quant, corner=True)
        plt.title(f'Pairplot of: {combo}')
        plt.tight_layout()
        plt.show()
        print('_____________________________________________')


def heatmap_combos(df):
    ''' 
    Create heatmaps for unique combos of vars where combo length is greater than 3
    '''
    feats, cat_feats, quant_feats = separate_feats(df)
    combos = feature_combos(df)

    for combo in combos:
        if len(combo) >= 3:
            plt.figure(figsize=(8,5))
            plt.title(f'Heatmap of {len(combo)} features')
            sns.heatmap(df[list(combo)].corr(), cmap = 'plasma', annot=True)
            plt.tight_layout()
            plt.show()
            print(f'Heatmap features: {combo}')
            print('_____________________________________________')

def encode_cols(train, validate, test, col_list):
    '''Function to preform one hot encoding on a list of columns in train, validate and test datasets '''
    encoded_train = pd.get_dummies(data = train, columns = col_list)
    encoded_validate = pd.get_dummies(data = validate, columns = col_list)
    encoded_test = pd.get_dummies(data = test, columns = col_list)

    return encoded_train, encoded_validate, encoded_test


def scaling(train, validate, test, scaler_type = MinMaxScaler()):
    """
    Takes in train, validate and test dfs or arrays
    Returns scaler, train_scaled, validate_scaled, test_scaled dfs/arrays
    """
    # Chooses scaler to use
    scaler = scaler_type

    # Checking for if train is an array or not
    if len(train.shape) > 1:
        feats, cat_feats, quant_feats = separate_feats(train)
        # Lists all object vars in cat feats
        obj_vars = []
        for col in train.columns:
            matches = [match for match in cat_feats if match == col]
            for match in matches:
                if (train[match].dtype == object) | (train[match].dtype == 'uint8'):
                    obj_vars.append(col)
        num_vars = [col for col in train.columns if col not in obj_vars]
        
        
        
        train_scaled = train.copy()
        validate_scaled = validate.copy()
        test_scaled = test.copy()

        # Scaling
        train_scaled[num_vars] = scaler.fit_transform(train_scaled[num_vars])
        validate_scaled[num_vars] = scaler.transform(validate_scaled[num_vars])
        test_scaled[num_vars] = scaler.transform(test_scaled[num_vars])

        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        train_scaled = train.copy()
        validate_scaled = validate.copy()
        test_scaled = test.copy()
        
        # Scaling
        train_scaled = scaler.fit_transform(train_scaled)
        validate_scaled = scaler.transform(validate_scaled)
        test_scaled = scaler.transform(test_scaled)
    
        return scaler, train_scaled, validate_scaled, test_scaled

###################################         EXPOLORE FUNCS         ###################################

def plot_categorical_and_continuous_vars(train, validate, test):
    ''' 
    Takes in scaled train, validate and testcales data and plots numerous visualizations
    '''
    df = train
    print('Histograms for each feature')
    hist_combos(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Relplots for each cat/quant feature pairing')
    relplot_pairs(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Lmplots for each cat/quant feature pairing')
    lmplot_pairs(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Jointplots for each cat/quant feature pairing')
    jointplot_pairs(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Heatmaps for feature combos (len >= 3)')
    heatmap_combos(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Pairplot of quant vars and target')
    pairplot_combos(df)

def explore_univariate(train, cat_vars, quant_vars):
    for var in cat_vars:
        explore_univariate_categorical(train, var)
        print('_________________________________________________________________')
    for col in quant_vars:
        p, descriptive_stats = explore_univariate_quant(train, col)
        plt.show(p)
        print(descriptive_stats)

def explore_bivariate(train, target, cat_vars, quant_vars):
    for cat in cat_vars:
        explore_bivariate_categorical(train, target, cat)
    for quant in quant_vars:
        explore_bivariate_quant(train, target, quant)

def explore_multivariate(train, target, cat_vars, quant_vars):
    '''
    '''
    plot_swarm_grid_with_color(train, target, cat_vars, quant_vars)
    plt.show()
    violin = plot_violin_grid_with_color(train, target, cat_vars, quant_vars)
    plt.show()
    pair = sns.pairplot(data=train, vars=quant_vars, hue=target)
    plt.show()
    plot_all_continuous_vars(train, target, quant_vars)
    plt.show()    

def co_linearity(df, target, cat_vars):
    '''returns a df of all p values and chi2
    scores based on the specified categorical variables and target'''
    cat = cat_vars.copy()
    cat.remove(target)
    
    chi2s = []
    p_score = []
    pdf = pd.DataFrame(columns = ['feature', 'p_values', 'chi2'])
    pdf['feature'] = list(cat)
    
    for i, var in enumerate(cat):
        observed = pd.crosstab(df[cat_vars[i]], df[target])
        chi2, p, degf, expected = stats.chi2_contingency(observed)
        p_score.append(p)
        chi2s.append(chi2)
    
    pdf['p_values'] = p_score
    pdf['chi2'] = chi2s
    pdf = pdf.sort_values(['p_values'], ascending = [True])
    return pdf.reset_index(drop = True)

def pearsons(df, target, quant_vars):
    '''Preforms pearsons r tests on each quant_var against target'''
    quant = quant_vars.copy()
    quant.remove(target)

    corrs = []
    p_score = []
    pdf = pd.DataFrame(columns = ['feature', 'p_values', 'corr'])
    pdf['feature'] = list(quant)
    
    for var in quant:
        corr, p = stats.pearsonr(df[target], df[var])
        p_score.append(p)
        corrs.append(corr)
    
    pdf['p_values'] = p_score
    pdf['corr'] = corrs
    pdf = pdf.sort_values(['p_values'], ascending = [True])
    return pdf.reset_index(drop = True)
### Univariate

def explore_univariate_categorical(train, cat_var):
    '''
    takes in a dataframe and a categorical variable and returns
    a frequency table and barplot of the frequencies. 
    '''
    frequency_table = freq_table(train, cat_var)
    plt.figure(figsize=(5,5))
    sns.barplot(x=cat_var, y='Count', data=frequency_table, color='lightseagreen')
    plt.title(cat_var)
    plt.show()

    print(frequency_table)

def explore_univariate_quant(train, quant_var):
    '''
    takes in a dataframe and a quantitative variable and returns
    descriptive stats table, histogram, and boxplot of the distributions. 
    '''
    descriptive_stats = train[quant_var].describe()
    plt.figure(figsize=(8,2))

    p = plt.subplot(1, 2, 1)
    p = plt.hist(train[quant_var], color='lightseagreen')
    p = plt.title(quant_var)

    # second plot: box plot
    p = plt.subplot(1, 2, 2)
    p = plt.boxplot(train[quant_var])
    p = plt.title(quant_var)
    return p, descriptive_stats

def freq_table(train, cat_var):
    '''
    for a given categorical variable, compute the frequency count and percent split
    and return a dataframe of those values along with the different classes. 
    '''
    class_labels = list(train[cat_var].unique())

    frequency_table = (
        pd.DataFrame({cat_var: class_labels,
                      'Count': train[cat_var].value_counts(normalize=False), 
                      'Percent': round(train[cat_var].value_counts(normalize=True)*100,2)}
                    )
    ).reset_index(drop=True)
    return frequency_table


#### Bivariate

def explore_bivariate_categorical(train, target, cat_var):
    '''
    takes in categorical variable and binary target variable, 
    returns a crosstab of frequencies
    runs a chi-square test for the proportions
    and creates a barplot, adding a horizontal line of the overall rate of the target. 
    '''
    print(cat_var, "\n_____________________\n")
    ct = pd.crosstab(train[cat_var], train[target], margins=True)
    chi2_summary, observed, expected = run_chi2(train, cat_var, target)
    p = plot_cat_by_target(train, target, cat_var)

    print(chi2_summary)
    print("\nobserved:\n", ct)
    print("\nexpected:\n", expected)
    plt.show(p)
    print("\n_____________________\n")

def explore_bivariate_quant(train, target, quant_var):
    '''
    descriptive stats by each target class. 
    compare means across 2 target groups 
    boxenplot of target x quant
    swarmplot of target x quant
    '''
    print(quant_var, "\n____________________\n")
    descriptive_stats = train.groupby(target)[quant_var].describe()
    average = train[quant_var].mean()
    mann_whitney = compare_means(train, target, quant_var)
    plt.figure(figsize=(4,4))
    boxen = plot_boxen(train, target, quant_var)
    swarm = plot_swarm(train, target, quant_var)
    plt.show()
    print(descriptive_stats, "\n")
    print("\nMann-Whitney Test:\n", mann_whitney)
    print("\n____________________\n")

## Bivariate Categorical

def run_chi2(train, cat_var, target):
    observed = pd.crosstab(train[cat_var], train[target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    chi2_summary = pd.DataFrame({'chi2': [chi2], 'p-value': [p], 
                                 'degrees of freedom': [degf]})
    expected = pd.DataFrame(expected)
    return chi2_summary, observed, expected

def plot_cat_by_target(train, target, cat_var):
    p = plt.figure(figsize=(2,2))
    p = sns.barplot(cat_var, target, data=train, alpha=.8, color='lightseagreen')
    overall_rate = train[target].mean()
    p = plt.axhline(overall_rate, ls='--', color='gray')
    return p


## Bivariate Quant

def plot_swarm(train, target, quant_var):
    average = train[quant_var].mean()
    p = sns.swarmplot(data=train, x=target, y=quant_var, color='lightgray')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_boxen(train, target, quant_var):
    average = train[quant_var].mean()
    p = sns.boxenplot(data=train, x=target, y=quant_var, color='lightseagreen')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

# alt_hyp = ‘two-sided’, ‘less’, ‘greater’

def compare_means(train, target, quant_var, alt_hyp='two-sided'):
    x = train[train[target]==0][quant_var]
    y = train[train[target]==1][quant_var]
    return stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)


### Multivariate

def plot_all_continuous_vars(train, target, quant_vars):
    '''
    Melt the dataset to "long-form" representation
    boxenplot of measurement x value with color representing the target variable. 
    '''
    my_vars = [item for sublist in [quant_vars, [target]] for item in sublist]
    sns.set(style="whitegrid", palette="muted")
    melt = train[my_vars].melt(id_vars=target, var_name="measurement")
    plt.figure(figsize=(8,6))
    p = sns.boxenplot(x="measurement", y="value", hue=target, data=melt)
    p.set(yscale="log", xlabel='')    
    plt.show()

def plot_violin_grid_with_color(train, target, cat_vars, quant_vars):
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.violinplot(x=cat, y=quant, data=train, split=True, 
                           ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()

def plot_swarm_grid_with_color(train, target, cat_vars, quant_vars):
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.swarmplot(x=cat, y=quant, data=train, ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()