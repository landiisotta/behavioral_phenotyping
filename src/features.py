import csv
import os
import re
import numpy as np
from datetime import datetime
from dataset import data_path

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
        {keys=pid, values=doa-ordered list(tokens)}
    """
    raw_behr = {}
    for table_name, data in df_dict.items():
        for row in data.itertuples():
            if bool(re.match('ados', table_name)):
                subj_vec = [table_name] + list(row[2:4]) + [_eval_age(row[3], row[4])] + \
                           ['::'.join([table_name,
                                       df_dict[table_name].columns[idx - 1],
                                       str(row[idx])]) for idx in range(7, len(row))
                            if row[idx] != '']
            elif bool(re.match('vin|srs', table_name)):
                subj_vec = [table_name] + list(row[2:4]) + [_eval_age(row[3], row[4])] + \
                           ['::'.join([table_name,
                                       'caretaker',
                                       df_dict[table_name].columns[idx - 1],
                                       str(row[idx])]) for idx in range(7, len(row))
                            if row[idx] != '']
            elif bool(re.match('psi', table_name)):
                subj_vec = [table_name] + list(row[2:4]) + [_eval_age(row[3], row[4])] + \
                           ['::'.join([table_name,
                                       row[6],
                                       df_dict[table_name].columns[idx - 1],
                                       str(row[idx])]) for idx in range(7, len(row))
                            if row[idx] != '']
            elif not (bool(re.match('emotional', table_name))):
                subj_vec = [table_name] + list(row[2:4]) + [_eval_age(row[3], row[4])] + \
                           ['::'.join([table_name,
                                       df_dict[table_name].columns[idx - 1],
                                       str(row[idx])]) for idx in range(7, len(row))
                            if row[idx] != '']
            raw_behr.setdefault(row[1], list()).append(subj_vec)
    for string_vec in raw_behr.values():
        string_vec.sort(key=lambda x: x[3])

    seq_len = np.array([len(ins) for ins in raw_behr.values()])
    print("Average length of behavioral sequences: {0:.3f}\n".format(np.mean(seq_len)))

    return raw_behr


def behr_level1(raw_behr):
    """
    Parameters
    ----------
    behr (dictionary)
        {keys=pid, values=list(ins_name, info, tokens)}

    Return
    ------
    dictionary
        {keys=pid, values=list(filtered tokens wrt level)}
    """
    os.makedirs(os.path.join(data_path, 'level-1'),
                exist_ok=True)

    deep_behr = {}
    for pid, tkn_vec in raw_behr.items():
        for el in tkn_vec:
            if bool(re.match('leit', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('scaled',
                                                                   str(x))), el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['leiter'] +
                                                                     x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('vinel', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('scaled',
                                                                   str(x))), el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['vineland'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('wa|wi|wpp', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('scaled_(bd|si|mr|vc::|ss|oa|in|cd|co|pc::|pcn::)',
                                                                   str(x))), el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['wechsler',
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
                tkn_vec_rid = el[1:4] + tkn_vec_rid
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
                tkn_vec_rid = el[1:4] + list(filter(lambda x: bool(re.search('raw', str(x)))
                                                              and not (bool(re.search('raw_dr|raw_ts',
                                                                                      str(x)))),
                                                    el))
            elif bool(re.match('srs', el[0])):
                tkn_vec_rid = el[1:4] + list(filter(lambda x: bool(re.search('raw', str(x)))
                                                              and not (bool(re.search('raw_tot', str(x)))),
                                                    el))
            elif bool(re.match('griffiths', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('q_', str(x))),
                                          el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['gmds'] + x.split('::')[1::]),
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
        {keys=pid, values=list(filtered tokens wrt level)}
    """

    os.makedirs(os.path.join(data_path, 'level-2'),
                exist_ok=True)

    deep_behr = {}
    for pid, tkn_vec in raw_behr.items():
        for el in tkn_vec:
            if bool(re.match('wa|wi|wpp', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('sumScaled_[PR|VC|V|P|WM|PS|PO|GL]',
                                                                   str(x))), el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['wechsler', x.split('_')[1]]),
                                                 tkn_vec_rid))
            elif bool(re.match('leit', el[0])):
                tkn_vec_rid = list(
                    filter(lambda x: bool(re.search('scaled', str(x))),
                           el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['leiter'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('vinel', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('sum_',
                                                                   str(x))), el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['vineland'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('ados-2modulo[1|toddler]', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('\.sa_tot|\.rrb_tot',
                                                                   str(x))), el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['ados'] + x.split('.')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('ados-2modulo[2|3]', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('::rrb_tot|::sa_tot', str(x))),
                                          el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['ados'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('psi', el[0])):
                tkn_vec_rid = el[0:3] + list(filter(lambda x: bool(re.search('raw', str(x)))
                                                              and not (bool(re.search('raw_dr|raw_ts',
                                                                                      str(x)))),
                                                    el))
            elif bool(re.match('srs', el[0])):
                tkn_vec_rid = el[1:4] + list(filter(lambda x: bool(re.search('raw', str(x)))
                                                              and not (bool(re.search('raw_tot', str(x)))),
                                                    el))
            elif bool(re.match('griffiths', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('q_', str(x))),
                                          el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['gmds'] + x.split('::')[1:]),
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
        {keys=pid, values=list(filtered tokens wrt level)}
    """

    os.makedirs(os.path.join(data_path, 'level-3'),
                exist_ok=True)

    deep_behr = {}
    for pid, tkn_vec in raw_behr.items():
        for el in tkn_vec:
            if bool(re.match('wa|wi|wpp', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('::FSIQ', str(x))),
                                          el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['wechsler'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('leit', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('composite_fr|::BIQ', str(x))),
                                          el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['leiter'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('vinel', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('standard_ABC', str(x))),
                                          el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['vineland'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('ados-2modulo[1|2|3|toddler]', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('::sarrb_tot|comparison_score',
                                                                   str(x))),
                                          el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['ados'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('psi', el[0])):
                tkn_vec_rid = el[1:4] + list(filter(lambda x: bool(re.search('::raw_ts', str(x)))
                                                              and not (bool(re.search('raw_dr', str(x)))),
                                                    el))
            elif bool(re.match('srs', el[0])):
                tkn_vec_rid = el[1:4] + list(filter(lambda x: bool(re.search('::raw_tot', str(x))),
                                                    el))
            elif bool(re.match('griffiths', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('GQ', str(x))),
                                          el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['gmds'] + x.split('::')[1:]),
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
        {keys=pid, values=list(filtered tokens wrt level)}
    """

    os.makedirs(os.path.join(data_path, 'level-4'),
                exist_ok=True)

    deep_behr = {}
    for pid, tkn_vec in raw_behr.items():
        for el in tkn_vec:
            if bool(re.match('wa|wi|wpp', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('::FSIQ', str(x))),
                                          el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['wechsler'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('leit', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('composite_fr|::BIQ', str(x))),
                                          el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['leiter'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('vinel', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('standard_[MSD|DLSD|CD|SD|ABC]',
                                                                   str(x))), el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['vineland'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('ados-2modulo[1|toddler]', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('\.sa_tot|\.rrb_tot|comparison_score',
                                                                   str(x))), el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: _subadosstring_(x),
                                                 tkn_vec_rid))
            elif bool(re.match('ados-2modulo[2|3]', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('::rrb_tot|::sa_tot|comparison_score',
                                                                   str(x))), el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['ados'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            elif bool(re.match('psi', el[0])):
                tkn_vec_rid = el[1:4] + list(filter(lambda x: bool(re.search('::raw_ts', str(x)))
                                                              and not (bool(re.search('raw_dr', str(x)))),
                                                    el))
            elif bool(re.match('srs', el[0])):
                tkn_vec_rid = el[1:4] + list(filter(lambda x: bool(re.search('::raw_[tot|rirb]', str(x))),
                                                    el))
            elif bool(re.match('griffiths', el[0])):
                tkn_vec_rid = list(filter(lambda x: bool(re.search('q_|GQ',
                                                                   str(x))), el))
                tkn_vec_rid = el[1:4] + list(map(lambda x: '::'.join(['gmds'] + x.split('::')[1:]),
                                                 tkn_vec_rid))
            deep_behr.setdefault(pid, list()).append(tkn_vec_rid)
    return deep_behr


def create_vocabulary(deep_behr, level=None):
    """
    Parameters
    ----------
    deep_behr (dictionary)
        {keys=pid, values=list(info, tokens}
    level (int)
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
            for v in vec[3::]:
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
            {keys=pid, values=list(info, tokens}
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
                behr.setdefault(pid, list()).append((s[2], [lab_to_idx[s[idx]]
                                                            for idx in range(3, len(s))]))
                wr.writerow([pid, s[2]] + [lab_to_idx[s[idx]]
                                           for idx in range(3, len(s))])
    return behr


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
