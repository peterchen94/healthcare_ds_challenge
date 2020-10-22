import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def extract_unique_time_params(file_list, file_dir, params_col, non_time_params):
    '''
    get list of unique parameters that represents the union of parameters from all patients measured after time 00:00

    :param file_list: list of text files from a directory
    :param file_dir: directory of files
    :param params_col: column for parameter names
    :param non_time_params: parameters which are not part of the time series measurements
    :return: list of unique time parameters
    '''
    unique_time_params = set()
    for file in file_list:
        df = pd.read_csv(os.path.join(file_dir, file))
        params = set([param for param in df[params_col].unique() if param not in non_time_params])
        unique_time_params.update(params)

    return list(unique_time_params)


def extract_nontime_df(df, non_time_params):
    '''
    extract a single row from a patient text file where columns map to the non time params

    :param df: dataframe from raw text file of patient
    :param non_time_params: list of non time parameters
    :return: dataframe thats is pivoted with a single row corresponding to a single patient for non time params
    '''
    df = df[(df['Parameter'].isin(non_time_params)) & (df['Time'] == '00:00')]
    df_pivot = pd.pivot_table(df, values='Value', index='Time', columns='Parameter')
    df_pivot = df_pivot.reset_index(drop=True)

    return df_pivot


def extract_day_df(df, unique_time_params, day=1):
    '''
    extract a single row from a patient text file which has the aggregated features for a corresponding day

    aggfunc is mean for this assessment

    :param df: dataframe from raw text file of patient
    :param unique_time_params: columns that are required for output - all non time parameters
    :param day: day 1 or day 2
    :return: dataframe thats is pivoted with a single row corresponding to a single patient
    '''
    record_id = df[df['Parameter'] == 'RecordID'].iloc[0]
    if day == 1:
        df = df[(df['Time'] > '00:00') & (df['Time'] <= '24:00')]
    elif day == 2:
        df = df[(df['Time'] > '00:00')]

    else:
        raise ValueError('day parameter must be 1 or 2')

    df_pivot = pd.pivot_table(df, columns='Parameter', aggfunc=np.mean)
    df_pivot['RecordID'] = record_id

    undefined_cols = [col for col in unique_time_params if col not in df_pivot.columns]

    df_pivot = df_pivot.reindex(columns=df_pivot.columns.tolist() + undefined_cols)
    df_pivot = df_pivot[['RecordID'] + unique_time_params]
    df_pivot = df_pivot.reset_index(drop=True)

    return df_pivot


def get_null_percentages(df):
    '''
    return percentage of missing value in each corresponding column in df

    :param df: dataframe
    :return: dataframe of missing percentages
    '''
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    missing_value_df = missing_value_df.sort_values(by='percent_missing', ascending = False)
    return missing_value_df




def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt