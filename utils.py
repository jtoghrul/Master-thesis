import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.io # read matlab file
import lightgbm as lgb # Light GBM
import gc # garbigue collector

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# Recursive Feature Elimination
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

###############################################
# Read Matlab file and convert into DataFrame #
###############################################


def _read_matlab_file(file_name):
    '''Read in Matlab file and convert into Pandas DataFrame'''
    
    # read .mat file
    eeg = scipy.io.loadmat(file_name)

    # Get variable names of matlab file
    print(list(eeg.keys()))

    # convert .mat file to numpy array
    eeg_array = eeg['Simulation']
    
    # print message
    print('Matlab file is converted to DataFrame')
    
    return eeg_array

def _get_simulation_parameters(eeg_array):
    '''Extract first part (parameters of simulation) of the parameters'''
    
    # generate id for each simulation
    number_of_simulation = eeg_array['len'][0][0][0].shape[0]
    simulation_id = [x for x in range(0, number_of_simulation)]

    # combines variables
    simulation = np.column_stack((simulation_id,
                                  eeg_array['len'][0][0][0],
                                  eeg_array['distance_sources'][0][0][0],
                                  eeg_array['con'][0][0][0],
                                  eeg_array['snr'][0][0][0],
                                  eeg_array['depth_1'][0][0][0],
                                  eeg_array['depth_2'][0][0][0]))
    # convert to Data Frame
    simulation = pd.DataFrame(simulation)

    simulation.columns = ['id', 'len', 
                          'distance_source', 
                          'con', 'snr', 
                          'depth_1', 'depth_2']
    print('Independent variables have been read : ', simulation.shape)
              
    return simulation


def _get_simulation_result(file_name, result_type='fpr'):
    '''Read second part of the Matlab file and transform into required format.
    
    Result is the second part of the simulation, and for each simulation we have 4 target values, 
    for different algorithms (localization source and connectivity estimate):
    
    0 | 0.5 | 0 | 0
    0 | 1   | 1 | 1
    
    The goal is to transform Result matrix to column matrix, where each target value placed on the seperate row:
    
    localization_source | connectivity_estimate | target
    0 | 0 | 0
    0 | 1 | 0.5
    1 | 0 | 0
    1 | 1 | 0
    0 | 0 | 0
    0 | 1 | 1
    1 | 0 | 1
    1 | 1 | 1
    
    '''
    # get values
    eeg_array = _read_matlab_file(file_name)
    simulation = _get_simulation_parameters(eeg_array)
    
    # calculate variables about simulation
    number_of_simulation = simulation.shape[0]
    
    # replicate each simulation independent variables 4 times
    replicated_simulation = pd.concat([simulation] * 4).sort_index().reset_index()
    
    # get FPR and FNR from Result matrix
    if result_type == 'fpr':
        false_rate = np.transpose(eeg_array['Results'][0][0][0][0][0]) # False negative rate
    else:
        false_rate = np.transpose(eeg_array['Results'][0][0][0][0][1]) # False negative rate
    
    # encode 2 by 2 table with dummy variables: result contains 4 columns
    # localization_source | connectivity_estimate
    # 0 | 0
    # 0 | 1
    # 1 | 0
    # 1 | 1
    # total 4 possible values.
    two_by_two_table = [0, 0, 0, 1, 1, 0, 1, 1]

    # create numpy array from 2 by 2 table (list)
    numpy_tbt_table = np.array(two_by_two_table * number_of_simulation)

    # convert to correct size: from 1D to 2D
    numpy_tbt_table = numpy_tbt_table.reshape(4 * number_of_simulation, 2)
    
    # reshape result matrix
    fr_ = false_rate.reshape(false_rate.shape[0] * false_rate.shape[1])
    
    # combine and convert matrix to data frame
    df_fr = pd.DataFrame(np.column_stack([numpy_tbt_table, fr_]))
    
    # rename columns
    df_fr.rename(columns={0:'localization_source', 1:'connectivity_estimate', 2:'y'}, inplace=True)
    
    # combine with simulated dataframe
    everything = pd.concat([replicated_simulation, df_fr], axis=1)
    
    # remove missing values from target value
    # everything.dropna(axis=0, inplace=True)
    
    # delete (automatic) index column
    del everything['index']
    
    print('Simulation Result is read')
    
    return everything

def _save_simulation(df, place_to_save):
    '''Save simulation data into `csv`'''
    df.to_csv(place_to_save, index=False)
    

def read_and_save_simulation(file_name, save=None):
    '''Read Simulation file and save into csv file'''
    
    everything = _get_simulation_result(file_name)
    
    # save data frame at save file
    if save is not None:
        _save_simulation(everything, save)
    
    # return final result
    return everything

def random_sampling(frac=0.25):
    '''randomly sample dataset'''
    
    # random sampling
    simulation = simulation.sample(frac=frac)
    
    # reset index
    simulation.reset_index(drop=True, inplace=True)
    
    # return result
    return simulation


def univariate_selection(score_func, k, X, y, names):
    '''Perform Univariate feature selection
    
    @names : feature names
    '''
    
    # Suppose, we select 5 features with top 5 Fisher scores
    selector = SelectKBest(f_classif, k = 5)

    # New dataframe with the selected features for later use in the classifier. fit() method works too, 
    # if you want only the feature names and their corresponding scores
    X_new = selector.fit_transform(X, y)
    scores = selector.scores_[selector.get_support()]
    names_scores = list(zip(names, scores))
    ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'Scores'])

    # Sort the dataframe for better visualization
    ns_df_sorted = ns_df.sort_values(['Scores', 'Feat_names'], ascending = [False, True])
    
    return ns_df_sorted





