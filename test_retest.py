import json
import os
import fnmatch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calc_cvs(df, subject_list, session_list, subject_col, session_col, structs_of_interest, method='gluer'):
    """
    Given:
        - a dataframe; as returned by load_dataset() (`df`)
        - a list of subjects to iterate over (`subject_list`)
        - a list of sessions to iterate over (`session_list`) 
          - maclaren method assummes all subjects have all sessions and each session has 2 repeated meas
          - gluer method might assume 'all subjects have all sessions' (?)
            - this is ok for decnef
          - gluer doesn't assume all sessions have same num repeated measures
        - the column name in the dataset's tsv that denotes subject_id (`subject_col`)
        - the column name in the dataset's tsv that denotes session_id (`session_col`)
        - a list of columns names in the dataframe over which to compute CoVs (`structs_of_interest`)
        - method to employ to compute intra-session CoV ('maclaren' or 'gluer'; default='gluer')
    Produce:
        - a dataframe with the total coefficient of variation (CoV) for each element in `structs_of_interest` 
          (`total_cvs_df`)
        - a datafram with the intra-session coefficient of variation (CoV) for each element in `structs_of_interest`
          session_cvs_df  
    See: 
        - [1] Maclaren, Julian, et al. "Reliability of brain volume measurements: a test-retest dataset." 
          Scientific data 1.1 (2014): 1-9.
        - [2]: Glüer, C-C., et al. "Accurate assessment of precision errors: how to measure the 
          reproducibility of bone densitometry techniques." Osteoporosis international 5.4 (1995): 262-270.
    """

    # Holds the intra-session CoV for each subject/struct
    # Will eventually be a numpy array
    subject_session_cvs = None
    # Holds the total CoV for each subject/struct
    # Will eventually be a numpy array
    subject_total_cvs = None
    
    for subject in subject_list:
        # Select by subject; make a numpy array
        subject_df = df[df[subject_col]==subject]
        subject_level_vals = subject_df.loc[:,structs_of_interest].to_numpy()
        
        # same m used to compute sigma_s in [1]
        m = 0

        # Calculate total CoV for this subject
        total_cvs = 100 * np.std(subject_level_vals,axis=0)/np.mean(subject_level_vals,axis=0)
        if subject_total_cvs is None:
            subject_total_cvs = total_cvs
        else:
            subject_total_cvs = np.stack((subject_total_cvs,total_cvs))

        # Compute `subject_session_cvs` according to [1] or eq's 5 and 6 in [2]
        if (method == 'maclaren'):
            # Compute `session_cvs` a la Maclaren
            
            # To track the summation in $\sigma_s = \sqrt{\frac{\sum{(x_i'-x_i'')^2}}{2m}}$ in [1]
            # Will eventually be a numpy array
            subject_sum = None
        
            for session in session_list:
                # Select by subject ADN session; make a numpy array
                session_df = subject_df[subject_df[session_col]==session]
                session_level_vals = session_df.loc[:,structs_of_interest].to_numpy()
                
                # At least one of the pairs did not get processed properly; skip
                # this is fragile and probably needs more work
                vals_has_a_nan = np.isnan(np.sum(session_level_vals))
                if vals_has_a_nan:
                    continue
                # Number of sessions
                m += 1
                # np.diff() assumes only 2 measurments per session (as does MacLaren) and will break
                # if anything else is passed
                diff_squared = np.square(np.diff(session_level_vals,axis=0).flatten())
                if subject_sum is None:
                    subject_sum = diff_squared
                else:
                    subject_sum += diff_squared
            # eq 1 in [1]        
            sigma_s = np.sqrt(np.divide(subject_sum,2*m))
            # eq 2 in [1]
            session_cvs = 100 * sigma_s / np.mean(subject_level_vals,axis=0)
                
        elif (method=='gluer'):
            # Compute `session_cvs` a la Gluer
            
            m = len(session_list)
            # record the number of repeated measurements in each session and compute df
            n_meas = []
            for session in session_list:
                n_meas.append(subject_df[subject_df[session_col]==session].shape[0])
            # eq 7 in [2]
            deg_freedom = np.sum(np.subtract(n_meas,1))

            # counter for doube summation term of eq 6 in [2]
            std_ctr_div_df = None
            # counter for the summation term of eq 5 in [2]
            x_j_over_m = None

            for session in session_list:
                session_df = subject_df[subject_df[session_col]==session]
                session_level_vals = session_df.loc[:,structs_of_interest].to_numpy()
                # summation in eq 6 in [2]
                if std_ctr_div_df is None:
                    std_ctr_div_df = np.sum(np.square(np.mean(session_level_vals,axis=0) - session_level_vals)/deg_freedom,axis=0)
                else:
                    std_ctr_div_df += np.sum(np.square(np.mean(session_level_vals,axis=0) - session_level_vals)/deg_freedom,axis=0)
                # summation in eq 5 in [2]
                if x_j_over_m is None:
                    x_j_over_m = np.mean(session_level_vals,axis=0)/m
                else:
                    x_j_over_m += np.mean(session_level_vals,axis=0)/m                    
            # eq 6 in [2]
            sigma_s = np.sqrt(std_ctr_div_df)
            # eq 5 in [2]
            session_cvs = 100 * (sigma_s / x_j_over_m)            
        else:
            print('Balls')

        # Record this subect's intra-session CoVs   
        if subject_session_cvs is None:
            subject_session_cvs = session_cvs
        else:
            subject_session_cvs = np.stack((subject_session_cvs,session_cvs))            

    # We now have:
    # - `subject_session_cvs`: a n x k array of each subject's intra-session coefficient of variation for each 
    #    entry in `structs_of_interest`
    # - `subject_total_cvs`: a n x k array of each subject's total coefficient of variation for each 
    #    entry in `structs_of_interest`
    
    # Take the mean of the coefficients of variation across subject for each struct,
    # making sure to RMS average them together (not arithmetic avg)
    # eq 4a in [2]
    session_cvs = np.sqrt(np.mean(np.square(subject_session_cvs),axis=0))
    total_cvs = np.sqrt(np.mean(np.square(subject_total_cvs),axis=0))

    # Stuff results back into a dataframe    
    total_cvs_series = pd.Series(total_cvs, index=structs_of_interest)
    session_cvs_series = pd.Series(session_cvs, index=structs_of_interest)
    abs_diff_cvs_series = pd.Series(np.abs(total_cvs - session_cvs), index=structs_of_interest)
    # Ok dataframes are fun now..
    means = df[ (df[subject_col].isin(subject_list)) & \
                (df[session_col].isin(session_list)) ] \
                  [structs_of_interest].mean()
    idx = ['mean-vol','total-cov','session-cov','abs-diff-cov']
    list_of_series = [means, total_cvs_series, session_cvs_series, abs_diff_cvs_series]
    results = pd.DataFrame(list_of_series, columns=structs_of_interest, index=idx)
    return results

def session_permute(df, subject_list, subject_col, session_col):
    '''
    Given:
        - a datafram (`df`)
        - the list of subjects to operate over (`subject_list`)
        - the column name in the dataset's tsv that denotes subject_id (`subject_col`)
        - the column name in the dataset's tsv that denotes session_id (`session_col`)
    Produce:
        - a datafram where the session labels for every subject in `subject_list` has been randomly permuted
    '''
    
    new_df = None
    for subject in subject_list:
        subject_df = df[df[subject_col]==subject]
        session_list_random_permute = np.random.permutation(subject_df[session_col].to_numpy())
        sub_idx = subject_df.index
        subject_df_perm = subject_df.drop(session_col,axis=1)
        subject_df_perm.insert(1,session_col,session_list_random_permute)
        if new_df is None:
            new_df = subject_df_perm
        else:
            new_df = new_df.append(subject_df_perm)        
    return new_df

def monte_carlo_perm_test(df, subject_list, session_list, subject_col, session_col, structs_of_interest, n_itrs=100, method='gluer'):

    # Calculate the actual coefficients of variation for the dataset
    cvs_df = calc_cvs(df,subject_list,session_list,subject_col,session_col,structs_of_interest,method=method)

    # Now simulate how likely we are to observe an equal or greater difference 
    # by randomly permuting session_id's
    counter = np.zeros(cvs_df.loc['abs-diff-cov'].to_numpy().shape)
    for i in range(n_itrs):
        permuted_df = session_permute(df, subject_list, subject_col, session_col)
        simulated_cvs_df = calc_cvs(permuted_df,subject_list,session_list,subject_col,session_col,structs_of_interest,method=method)    
        counter += 1 * (simulated_cvs_df.loc['abs-diff-cov'] >= cvs_df.loc['abs-diff-cov'])

    pvals = (counter/n_itrs).rename('p-vals')
    cvs_with_pval = cvs_df.append(pvals)

    return cvs_with_pval
