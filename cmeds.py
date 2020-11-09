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

def add_regions(df,regions):
    """
    Given:
    - a pandas datafram
    - a list of regions to add together
      - The first element in the list is a list of regions to add together
      - The second element in the list is the name of the resulting region
      - eg: `regions = [ [ ['Right-Amygdala','Left-Amygdala'],'Amygdala'] ]`
    Produce:
    - a pandas dataframe with additional colums reulting from adding the regions together as specified
    """
    # Specify regions to add together as a list.
    
    for region in regions:
        regions_to_add = region[0]
        new_region_name = region[1]
        df[new_region_name] = df[regions_to_add].sum(axis=1)
    return df

def load_dataset(jsonpath, demofile, drop_subjects=[], vol_data_src='volume'):
    
    if (vol_data_src != 'volume_percent_icv') & \
       (vol_data_src != 'volume') & \
       (vol_data_src != 'samseg_volume'):
        print('`vol_data_src` must either be "volume" or "volume_percent_icv"')
        return None
            
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
            vol_data[subname] = subject_data[subname]['measurements'][vol_data_src]
            norm_data[subname] = {}
            for vol in subject_data[subname]['normative']['volume']:
                norm_data[subname][vol] = subject_data[subname]['normative']['volume'][vol]['percentiles']['percentile']        
            subject_list.append(subname)
        else:
            # Subject was processed with error

            drop_subjects.append(subname)
            print('Ignoring Subject (did it error out?)', subname)

    demo_dataf = pd.read_csv(demofile, sep='\t', index_col='subject_id')
    vol_temp_df = pd.DataFrame.from_dict(data=vol_data, orient='index')
    norm_temp_df = pd.DataFrame.from_dict(data=norm_data, orient='index')

    # for convience, concatenate demographics info onto vol and norm dataframes
    print('Dropping the following subjects', drop_subjects)
    vol_dataf = pd.concat([demo_dataf,vol_temp_df],axis=1).drop(drop_subjects, errors='ignore')
    norm_dataf = pd.concat([demo_dataf,norm_temp_df],axis=1).drop(drop_subjects, errors='ignore')

    return vol_dataf, norm_dataf


### -------------------------------------------
# Functions for loading in FreeSurfer data

def load_fs_dataset(statspath, demofile, structs_of_interest, drop_subjects=[], glob_pattern='aseg.stats'):
    aseg_stats_files = find_json_files(statspath,glob_pattern)
    vol_data = {}
    for aseg_stats_file in aseg_stats_files:
        df = get_aseg_stats_dataframe(aseg_stats_file)    
        # subject name is assumed to be the directory name one level above the location of aseg.stats (FS directory structure)
        sub_name = os.path.split(os.path.split(os.path.dirname(aseg_stats_file))[0])[1]
        sub_vol_data = {}
        for struct in structs_of_interest:
            sub_vol_data[struct] = float(df[df['StructName']==struct]['Volume_mm3'])
        vol_data[sub_name] = sub_vol_data

    vol_temp_df = pd.DataFrame.from_dict(data=vol_data, orient='index')
    demo_dataf = pd.read_csv(demofile, sep='\t', index_col='subject_id')

    # for convience, concatenate demographics info onto vol and norm dataframes
    print('Dropping the following subjects', drop_subjects)
    vol_dataf = pd.concat([demo_dataf,vol_temp_df],axis=1).drop(drop_subjects, errors='ignore')

    return vol_dataf

def load_fssamseg_dataset(statspath, demofile, structs_of_interest, drop_subjects=[], glob_pattern='samseg.stats'):
    samseg_stats_files = find_json_files(statspath,glob_pattern)
    vol_data = {}
    
    for filename in samseg_stats_files:
        # subject name is assumed to be the directory name one level above the location of aseg.stats (FS directory structure)
        sub_name = os.path.split(os.path.split(os.path.dirname(filename))[0])[1]
        sub_vol_data = {}
        with open(filename) as f:
            for line in f:
                for struct in structs_of_interest:
                    if struct in line:
                        vol = float(line.split(',')[-2])
                        sub_vol_data[struct] = vol
        vol_data[sub_name] = sub_vol_data
    vol_temp_df = pd.DataFrame.from_dict(data=vol_data, orient='index')
    demo_dataf = pd.read_csv(demofile, sep='\t', index_col='subject_id')

    # for convience, concatenate demographics info onto vol and norm dataframes
    print('Dropping the following subjects', drop_subjects)
    vol_dataf = pd.concat([demo_dataf,vol_temp_df],axis=1).drop(drop_subjects, errors='ignore')
    
    return vol_dataf

def add_gm_wm_to_dataframe(aseg_dataframe, aseg_stats_file):
    label_number_dict = {'Left-Cerebral-White-Matter': 2,
                         'Left-Cerebral-Cortex': 3,
                         'Right-Cerebral-White-Matter': 41,
                         'Right-Cerebral-Cortex': 42}

    label_name_dict = {'lhCerebralWhiteMatter': 'Left-Cerebral-White-Matter',
                       'lhCortex': 'Left-Cerebral-Cortex',
                       'rhCerebralWhiteMatter': 'Right-Cerebral-White-Matter',
                       'rhCortex': 'Right-Cerebral-Cortex'}

    def get_volume(line):
        return float(line.split(',')[-2])

    label_stats = []
    with open(aseg_stats_file) as f:
        for line in f:
            for label in label_name_dict:
                if label in line:
                    vol = get_volume(line)
                    struct = label_name_dict[label]
                    row = {'SegId': label_number_dict[struct],
                           'NVoxels': np.nan,
                           'Volume_mm3': vol,
                           'StructName': struct,
                           'normMean': np.nan,
                           'normStdDev': np.nan,
                           'normMin': np.nan,
                           'normMax': np.nan,
                           'normRange': np.nan}
                    label_stats.append(row)

    aseg_gm_wm_dataframe = aseg_dataframe.append(label_stats).reset_index(drop=True)

    return aseg_gm_wm_dataframe

def get_aseg_stats_dataframe(aseg_stats_file):
    column_headers = ['SegId',
                      'NVoxels',
                      'Volume_mm3',
                      'StructName',
                      'normMean',
                      'normStdDev',
                      'normMin',
                      'normMax',
                      'normRange']

    aseg_dataframe = pd.read_table(aseg_stats_file,
                                   delim_whitespace=True,
                                   header=None,
                                   comment='#',
                                   index_col=0,
                                   names=column_headers)

    aseg_gm_wm_dataframe = add_gm_wm_to_dataframe(aseg_dataframe, aseg_stats_file)

    return aseg_gm_wm_dataframe
