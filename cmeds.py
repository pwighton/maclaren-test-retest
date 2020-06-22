import json
import os
import fnmatch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Helper functions for loading cmeds datasets into pandas dataframs

def find_json_files(filepath, pattern="*.json"):
    """
    Given:
    - a directory to search

    Produce:
    - a list of all .json files anywhere in the directory path
    - equivalient to the shell command `find /filepath -name '*.json'`
    """
    filelist = []
    for dName, sdName, fList in os.walk(filepath):
        for fileName in fList:
            if fnmatch.fnmatch(fileName, pattern):
                filelist.append(os.path.join(dName, fileName))
    return filelist

def load_json_file(filename):
    """
    Given:
    - a json file

    Produce:
    - dictionary
    """
    with open(filename) as infile:
        json_data = json.load(infile)
    return json_data

def load_dataset(jsonpath, demofile, drop_subjects=None):
    json_files = find_json_files(jsonpath)
    subject_data = {}
    vol_data = {}
    norm_data = {}
    subject_list = []
    for file in json_files:
        subname = os.path.basename(os.path.dirname(file))
        subject_data[subname] = load_json_file(file)
        if 'normative' in subject_data[subname] and 'volume' in subject_data[subname]['normative']:
            # Subject was processed without error
            vol_data[subname] = subject_data[subname]['measurements']['volume_percent_icv']
            norm_data[subname] = {}
            for vol in subject_data[subname]['normative']['volume']:
                norm_data[subname][vol] = subject_data[subname]['normative']['volume'][vol]['percentiles']['percentile']        
            subject_list.append(subname)
        else:
            # Subject was processed with error
            if drop_subjects is None:
                drop_subjects = []
            drop_subjects.append(subname)
            print('Ignoring Subject (did it error out?)', subname)

    demo_dataf = pd.read_csv(demofile, sep='\t', index_col='subject_id')
    vol_temp_df = pd.DataFrame.from_dict(data=vol_data, orient='index')
    norm_temp_df = pd.DataFrame.from_dict(data=norm_data, orient='index')
    if drop_subjects is not None:
        print('Dropping the following subejcts:', drop_subjects)
        demo_dataf.drop(drop_subjects, errors='ignore')
        vol_temp_df.drop(drop_subjects, errors='ignore')
        norm_temp_df.drop(drop_subjects, errors='ignore')
    # for convience, concatenate demographics info onto vol and norm dataframes
    vol_dataf = pd.concat([demo_dataf,vol_temp_df],axis=1)
    norm_dataf = pd.concat([demo_dataf,norm_temp_df],axis=1)

    return vol_dataf, norm_dataf
