import os
import csv
import sys
import argparse
import pandas as pd
import numpy as np
from time import time
import utils as ut

# create feature dataset, ONLY LEVEL-4
# impute missing values with mean
def feature_dataset(datadir):
    # feature list and behavioral ehrs
    feat_list, idx_to_bt = _load_vocabulary(datadir, ut.file_names['vocab'])
    behr = _load_behr(datadir, ut.file_names['behr'], idx_to_bt)

    # split data into time frames
    t_behr = {k: {'F1': {},
                  'F2': {},
                  'F3': {},
                  'F4': {},
                  'F5': {}} 
              for k in behr.keys()}

    for id_subj, t_vect in behr.items():
        for el in t_vect:
            if 0 < float(el[0]) <= 2.5:
                for e in el[1:]:
                    t_behr[id_subj]['F1'].setdefault('::'.join(e.split('::')[:len(e.split('::'))-1]), 
                                                     list()).append(int(e.split('::')[-1]))
            elif 2.5 < float(el[0]) <= 6.0:
                for e in el[1::]:
                    t_behr[id_subj]['F2'].setdefault('::'.join(e.split('::')[:len(e.split('::'))-1]), 
                                                     list()).append(int(e.split('::')[-1]))
            elif 6.0 < float(el[0]) <= 13.0:
                for e in el[1::]:
                    t_behr[id_subj]['F3'].setdefault('::'.join(e.split('::')[:len(e.split('::'))-1]), 
                                                     list()).append(int(e.split('::')[-1]))
            elif 13.0 < float(el[0]) < 17.0:
                for e in el[1::]:
                    t_behr[id_subj]['F4'].setdefault('::'.join(e.split('::')[:len(e.split('::'))-1]), 
                                                     list()).append(int(e.split('::')[-1]))
            else:
                for e in el[1::]:
                    t_behr[id_subj]['F5'].setdefault('::'.join(e.split('::')[:len(e.split('::'))-1]), 
                                                     list()).append(int(e.split('::')[-1]))

    # create long and wide dataframes with scaled and raw values and save results
    # repeated values in the same time frame are averaged out
    df_dict = {k: {} for k in t_behr.keys()}
    for id_lab, t_dict in t_behr.items():
        for t in t_dict:
            for f in feat_list:
                try: 
                    df_dict[id_lab].setdefault(t, 
                                    list()).extend([np.mean(tehr[id_lab][t][f])])
                except KeyError:
                    df_dict[id_lab].setdefault(t, list()).extend([np.nan])

    df_list = []
    df_list_wide = []
    for id_subj, t_dict in df_dict.items():
        tmp_l = []
        feat_wide = []
        for t, el in t_dict.items():
            for f in feat_list:
                feat_wide.append('::'.join([t, f]))
            df_list.append([id_subj, t] + [e for e in el])
            tmp_l.extend([e for e in el])
        df_list_wide.append([id_subj] + tmp_l)
    df = pd.DataFrame(df_list, columns=['id_subj', 'time'] + feat)
    df = df.set_indes('id_subj')

    df_wide = pd.DataFrame(df_list_wide, columns=['id_subj'] + feat_wide)
    df_wide = df_wide.set_index('id_subj')
    df_wide_fi = df_wide.fillna(df_wide.mean()).dropna(axis=1)

    # scale data
    scaler = StandardScaler()

    df_wide_scaled = scaler.fit_transform(df_wide_fi)

    df_wide_scaled = pd.DataFrame(df_wide_scaled, columns=list(df_wide_fi.columns))
    df_wide_scaled['id_subj'] = df_wide.index
    df_wide_scaled = df_wide_scaled.set_index('id_subj')

    # save datsets
    df.to_csv(datadir + 'cohort-feat-long.csv', 
              header=True, index=True, index_label='id_subj'
    df_wide.to_csv(datadir + 'cohort-feat-wide.csv', 
                   header=True, index=True, index_label='id_subj')
    df_wide_scaled(datadir + 'cohort-feat-wide-scaled.csv', 
                   header=true, index=True, index_label='id_subj')


"""
Private Functions
"""


def _load_vocabulary(datadir, filename):
    with open(os.path.join(datadir, filename)) as f:
        rd = csv.reader(f)
        next(rd)
        feat = set()
        idx_to_vc = {}
        for r in rd:
            idx_to_vc[r[1]] = r[0]
            tmp_sp = r[0].split('::')
            feat.add('::'.join(tmp_sp[:len(tmp_sp)-1]))
        feat = sorted(list(feat))
    return feat, idx_to_vc



def _load_behr(datadir, filename, idx_to_bt):
    with open(os.path.join(datadir, filename)) as f:
    rd = csv.reader(f)
    next(rd)
    behr = {}
    for r in rd:
        behr.setdefault(r[0], 
        list()).append([float(r[1])] + sorted([idx_to_bt[t] for t in r[2::]]))
       

"""
Main Function
"""


def _process_args():
    parser = argparse.ArgumentParser(
             description='Create feature dataset from level 4 ')
    parser.add_argument(dest='datadir', help='data directory')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _process_args()
    print('')

    start = time()
    feature_dataset(args.datadir)
    print("Processing time: %d" % round(time() - start, 2)) 
