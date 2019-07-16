import csv
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from .dataset import data_path
from sklearn.preprocessing import StandardScaler


# Pinfo class to store subjects' demographics
class Pinfo():
    def __init__(self,
                 sex: str,
                 dob: datetime,
                 aoa: float):
        self.sex = sex
        self.dob = dob
        self.aoa = aoa


"""
Functions:

Create vocabulary on 4 different levels:
- level 1: deepest level subtests;
- level 2: aggregation of subtest scores;
- level 3: general indices;
- level 4: indices and subtests of interest.
"""


def create_tokens(df_dict):
    """
    Parameters
    ----------
    df_dict (dictionary)
        {keys=table_name, values=table dataframes}

    Return
    ------
    dictionary
        {keys=pid, values=[[table_name, Pinfo, doa-ordered list(tokens)]]}
    """
    raw_behr = {}
    for table_name, data in df_dict.items():
        for row in data.itertuples():
            if bool(re.match('ados', table_name)):
                subj_vec = [table_name] + \
                           [Pinfo(row.sex, row.date_birth,
                                  _eval_age(row.date_birth, row.date_ass))] + \
                           ['::'.join([table_name,
                                       df_dict[table_name].columns[idx - 1],
                                       str(row[idx])]) for idx in range(7, len(row))
                            if row[idx] != '']
                raw_behr.setdefault(row[1], list()).append(subj_vec)
            elif bool(re.match('vin|srs', table_name)):
                subj_vec = [table_name] + \
                           [Pinfo(row.sex, row.date_birth,
                                  _eval_age(row.date_birth, row.date_ass))] + \
                           ['::'.join([table_name,
                                       'caretaker',
                                       df_dict[table_name].columns[idx - 1],
                                       str(row[idx])]) for idx in range(7, len(row))
                            if row[idx] != '']
                raw_behr.setdefault(row[1], list()).append(subj_vec)
            elif bool(re.match('psi', table_name)):
                subj_vec = [table_name] + \
                           [Pinfo(row.sex, row.date_birth,
                                  _eval_age(row.date_birth, row.date_ass))] + \
                           ['::'.join([table_name,
                                       row[6],
                                       df_dict[table_name].columns[idx - 1],
                                       str(row[idx])]) for idx in range(7, len(row))
                            if row[idx] != '']
                raw_behr.setdefault(row[1], list()).append(subj_vec)
            elif not (bool(re.match('emotional', table_name))):
                subj_vec = [table_name] + \
                           [Pinfo(row.sex, row.date_birth,
                                  _eval_age(row.date_birth, row.date_ass))] + \
                           ['::'.join([table_name,
                                       df_dict[table_name].columns[idx - 1],
                                       str(row[idx])]) for idx in range(7, len(row))
                            if row[idx] != '']
                raw_behr.setdefault(row[1], list()).append(subj_vec)
    for string_vec in raw_behr.values():
        string_vec.sort(key=lambda x: x[1].aoa)

    seq_len = np.array([len(ins) for ins in raw_behr.values()])
    print("Average length of behavioral sequences: {0:.3f}\n".format(np.mean(seq_len)))

    return raw_behr


def behr_level1(raw_behr):
    """
    Parameters
    ----------
    raw_behr (dictionary)
        {keys=pid, values=list(ins_name, info, tokens)}

    Return
    ------
    dictionary
        {keys=pid, values=[[Pinfo, filtered tokens wrt level]]}
    """
    os.makedirs(os.path.join(data_path, 'level-1'),
                exist_ok=True)

    deep_behr = {}
    for pid, tkn_vec in raw_behr.items():
        for el in tkn_vec:
            if bool(re.match('leit', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('scaled',
                                                                   str(x))), el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['leiter'] +
                                                                     x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('vinel', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('scaled',
                                                                   str(x))), el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['vineland'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('wa|wi|wpp', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('scaled_(bd|si|mr|vc::|ss|oa|in|cd|co|pc::|pcn::)',
                                                                   str(x))), el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['wechsler',
                                                                      x.split('_')[1]]),
                                                 tkn_vec_rid))
            elif bool(re.match('ados-2modulo[1|toddler]', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('\.[a|b|d]',
                                                                   str(x))), el))
                tkn_vec_rid = list(map(lambda x: '::'.join([x.split('::')[0],
                                                            x.split('.')[1]]),
                                       tkn_vec_rid))
                for idx, t in enumerate(tkn_vec_rid):
                    try:
                        ss = t.split('::')[1]
                        sc = t.split('::')[2]
                        if ss == 'd1' or ss == 'b1' or ss == 'd2':
                            tkn_vec_rid[idx] = '::'.join(['ados', ss, sc])
                        else:
                            tkn_vec_rid[idx] = t
                    except IndexError:
                        pass
                tkn_vec_rid = [el[1]] + tkn_vec_rid
            elif bool(re.match('ados-2modulo[2|3|4]', el[0])):
                tkn_vec_rid = list(filter(lambda x: not (bool(re.search('tot|score|lang|algor',
                                                                        str(x)))),
                                          el))
                for idx, t in enumerate(tkn_vec_rid):
                    try:
                        ss = t.split('::')[1]
                        sc = t.split('::')[2]
                        if ss == 'd1' or ss == 'b1' or ss == 'd2':
                            tkn_vec_rid[idx] = '::'.join(['ados', ss, sc])
                        else:
                            tkn_vec_rid[idx] = t
                    except:
                        pass
                tkn_vec_rid.pop(0)
            elif bool(re.match('psi', el[0])):
                tkn_vec_rid = [el[1]] + list(filter(lambda x: bool(re.search('raw', str(x)))
                                                              and not (bool(re.search('raw_dr|raw_ts',
                                                                                      str(x)))),
                                                    el))
            elif bool(re.match('srs', el[0])):
                tkn_vec_rid = [el[1]] + list(filter(lambda x: bool(re.search('raw', str(x)))
                                                              and not (bool(re.search('raw_tot', str(x)))),
                                                    el))
            elif bool(re.match('griffiths', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('q_', str(x))),
                                          el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['gmds'] + x.split('::')[1::]),
                                                 tkn_vec_rid))

            deep_behr.setdefault(pid, list()).append(tkn_vec_rid)
    return deep_behr


def behr_level2(raw_behr):
    """
    Parameters
    ----------
    behr (dictionary)
        {keys=pid, values=list(ins_name, info, tokens)}

    Return
    ------
    dictionary
        {keys=pid, values=[[Pinfo, filtered tokens wrt level]]}
    """

    os.makedirs(os.path.join(data_path, 'level-2'),
                exist_ok=True)

    deep_behr = {}
    for pid, tkn_vec in raw_behr.items():
        for el in tkn_vec:
            if bool(re.match('wa|wi|wpp', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('sumScaled_[PR|VC|V|P|WM|PS|PO|GL]',
                                                                   str(x))), el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['wechsler', x.split('_')[1]]),
                                                 tkn_vec_rid))
            elif bool(re.match('leit', el[0])):
                tkn_vec_rid = list(
                    filter(lambda x: bool(re.search('scaled', str(x))),
                           el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['leiter'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('vinel', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('sum_',
                                                                   str(x))), el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['vineland'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('ados-2modulo[1|toddler]', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('\.sa_tot|\.rrb_tot',
                                                                   str(x))), el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['ados'] + x.split('.')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('ados-2modulo[2|3]', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('::rrb_tot|::sa_tot', str(x))),
                                          el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['ados'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('psi', el[0])):
                tkn_vec_rid = [el[1]] + list(filter(lambda x: bool(re.search('raw', str(x)))
                                                              and not (bool(re.search('raw_dr|raw_ts',
                                                                                      str(x)))),
                                                    el))
            elif bool(re.match('srs', el[0])):
                tkn_vec_rid = [el[1]] + list(filter(lambda x: bool(re.search('raw', str(x)))
                                                              and not (bool(re.search('raw_tot', str(x)))),
                                                    el))
            elif bool(re.match('griffiths', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('q_', str(x))),
                                          el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['gmds'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            deep_behr.setdefault(pid, list()).append(tkn_vec_rid)
    return deep_behr


def behr_level3(raw_behr):
    """
    Parameters
    ----------
    behr (dictionary)
        {keys=pid, values=list(ins_name, info, tokens)}

    Return
    ------
    dictionary
        {keys=pid, values=[[Pinfo, filtered tokens wrt level]]}
    """

    os.makedirs(os.path.join(data_path, 'level-3'),
                exist_ok=True)

    deep_behr = {}
    for pid, tkn_vec in raw_behr.items():
        for el in tkn_vec:
            if bool(re.match('wa|wi|wpp', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('::FSIQ', str(x))),
                                          el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['wechsler'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('leit', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('composite_fr|::BIQ', str(x))),
                                          el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['leiter'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('vinel', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('standard_ABC', str(x))),
                                          el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['vineland'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('ados-2modulo[1|2|3|toddler]', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('::sarrb_tot|comparison_score',
                                                                   str(x))),
                                          el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['ados'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('psi', el[0])):
                tkn_vec_rid = [el[1]] + list(filter(lambda x: bool(re.search('::raw_ts', str(x)))
                                                              and not (bool(re.search('raw_dr', str(x)))),
                                                    el))
            elif bool(re.match('srs', el[0])):
                tkn_vec_rid = [el[1]] + list(filter(lambda x: bool(re.search('::raw_tot', str(x))),
                                                    el))
            elif bool(re.match('griffiths', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('GQ', str(x))),
                                          el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['gmds'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            deep_behr.setdefault(pid, list()).append(tkn_vec_rid)
    return deep_behr


def behr_level4(raw_behr):
    """
    Parameters
    ----------
    behr (dictionary)
        {keys=pid, values=list(ins_name, info, tokens)}

    Return
    ------
    dictionary
        {keys=pid, values=[[Pinfo, filtered tokens wrt level]]}
    """

    os.makedirs(os.path.join(data_path, 'level-4'),
                exist_ok=True)

    deep_behr = {}
    for pid, tkn_vec in raw_behr.items():
        for el in tkn_vec:
            if bool(re.match('wa|wi|wpp', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('::FSIQ', str(x))),
                                          el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['wechsler'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('leit', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('composite_fr|::BIQ', str(x))),
                                          el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['leiter'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('vinel', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('standard_[MSD|DLSD|CD|SD|ABC]',
                                                                   str(x))), el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['vineland'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('ados-2modulo[1|toddler]', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('\.sa_tot|\.rrb_tot|comparison_score',
                                                                   str(x))), el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: _subadosstring_(x),
                                                 tkn_vec_rid))
            elif bool(re.match('ados-2modulo[2|3]', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('::rrb_tot|::sa_tot|comparison_score',
                                                                   str(x))), el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['ados'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('psi', el[0])):
                tkn_vec_rid = [el[1]] + list(filter(lambda x: bool(re.search('::raw_ts', str(x)))
                                                              and not (bool(re.search('raw_dr', str(x)))),
                                                    el))
            elif bool(re.match('srs', el[0])):
                tkn_vec_rid = [el[1]] + list(filter(lambda x: bool(re.search('::raw_[tot|rirb]', str(x))),
                                                    el))
            elif bool(re.match('griffiths', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('q_|GQ',
                                                                   str(x))), el))
                tkn_vec_rid = [el[1]] + list(map(lambda x: '::'.join(['gmds'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            deep_behr.setdefault(pid, list()).append(tkn_vec_rid)
    return deep_behr


def create_vocabulary(deep_behr, level=None):
    """
    Parameters
    ----------
    deep_behr (dictionary)
        {keys=pid, values=[[Pinfo, filtered tokens wrt level]]
        level for which to build vocabulary

    Return
    ------
    dictionaries
        vocabulary idx to tokens
        vocabulary tokens to idx
    """
    if level is None:
        raise NameError("Specify the depth level to consider")

    lab_to_idx = {}
    idx_to_lab = {}
    idx = 0
    for lab, seq in deep_behr.items():
        for vec in seq:
            for v in vec[1::]:
                if v not in lab_to_idx:
                    lab_to_idx[v] = idx
                    idx_to_lab[idx] = v
                    idx += 1
    print("Vocabulary size:{0}\n".format(len(lab_to_idx)))
    with open(os.path.join(data_path, 'level-{0}'.format(level),
                           'cohort-vocab.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['LABEL', 'INDEX'])
        for l, idx in lab_to_idx.items():
            wr.writerow([l, idx])
    return lab_to_idx, idx_to_lab


def create_behr(deep_behr, lab_to_idx, level=None):
    """
        Parameters
        ----------
        deep_behr (dictionary)
            {keys=pid, values=[[Pinfo, filtered tokens wrt level]]}
        level (int)
            level for which to build behr

        Return
        ------
        dictionary
            {keys=pid, values=list((DOA, list(instrument tokens))}
    """
    if level is None:
        raise NameError("Specify the depth level to consider")

    # write files
    behr = {}
    with open(os.path.join(data_path,
                           'level-{0}'.format(level),
                           'cohort-behr.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['ID_SUBJ', 'DOA', 'TERM'])
        for pid, seq in deep_behr.items():
            for s in seq:
                behr.setdefault(pid, list()).append((s[0].aoa, [lab_to_idx[s[idx]]
                                                                for idx in range(1, len(s))]))
                wr.writerow([pid, s[0].aoa] + [lab_to_idx[s[idx]]
                                               for idx in range(1, len(s))])
    return behr


def create_features_data(term_behr_level4):
    """Create quantitative feature dataframe from level 4

    Parameters
    ----------
    term_behr_level4 (dictionary)
        dictionary as output from behr_level4

    Return
    ------
    dataframes (features, scaled_features)
    """
    # elevate SettingWithCopy warning to an exception
    # pd.set_option('mode.chained_assignment', 'raise')

    # vocabulary with feature names per time period
    fn_list = _feature_names(term_behr_level4)

    # initialize empty dataframe
    feat_df = pd.DataFrame(index=term_behr_level4.keys(),
                           columns=fn_list)
    for pid, tm_list in term_behr_level4.items():
        for terms in tm_list:
            for tm in terms[1:]:
                tm_words = tm.split('::')
                feat = '::'.join([_aoa_to_tf(terms[0].aoa)] + tm_words[:-1])
                feat_df.loc[pid, feat] = int(tm_words[-1])

    # impute missing values with mean
    feat_df.loc[:, :].fillna(feat_df.mean(), inplace=True)
    # rescale data (normalize data, mean-imputed missing values end up padded with zero)
    scaler = StandardScaler()
    X = feat_df.values
    X_scaled = scaler.fit_transform(X)
    feat_scaled_df = pd.DataFrame(X_scaled,
                                  index=term_behr_level4.keys(),
                                  columns=fn_list)

    return feat_df, feat_scaled_df


"""
Private Functions
"""


def _eval_age(dob, doa):
    """
    Parameters
    ----------
    dob (datetime)
        date of birth
    doa (datetime)
        date of assessment

    Return
    ------
    float
        age at the assessment
    """
    days_in_year = 365.2425
    try:
        adate = datetime.strptime(doa, '%d/%m/%Y').date()
        bdate = datetime.strptime(dob, '%d/%m/%Y').date()
        aoa = (adate - bdate).days / days_in_year
    except TypeError:
        aoa = -1
    return aoa


def _subadosstring_(item):
    """
    Parameters
    ----------
    item (str)
            modify comparison_score token from ados

    Return
    ------
    str
        modified item
    """
    if re.search('comparison_score', item):
        item = '::'.join(['ados'] + item.split('::')[1:])
    else:
        item = '::'.join(['ados'] + item.split('.')[1:])
    return item


def _aoa_to_tf(aoa):
    """returns the time period from the age of assessment

    Parameters
    ----------
    aoa (float)
        age of assessment

    Return
    ------
    str
        time period string (F1-F5)
    """
    if 0 < float(aoa) <= 2.5:
        return 'F1'
    elif 2.5 < float(aoa) <= 6.0:
        return 'F2'
    elif 6.0 < float(aoa) <= 13.0:
        return 'F3'
    elif 13.0 < float(aoa) < 17.0:
        return 'F4'
    else:
        return 'F5'


def _feature_names(term_behr):
    """Return list of columns for the quantitative feature dataset

    Parameters
    ----------
    term_behr (dictionary)
        {pid: [[Pinfo, str of behavioral terms]]}

    Return
    ------
    list of str
    """
    feat_list = []
    for term_list in term_behr.values():
        for tl in term_list:
            feat_list.extend(list(map(lambda x:
                                      '::'.join([_aoa_to_tf(tl[0].aoa)] +
                                                x.split('::')[:-1]),
                                      tl[1:])))
    feat_list = list(set(feat_list))
    return sorted(feat_list)
