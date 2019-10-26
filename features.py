import csv
import os
import re
import pandas as pd
from dataset import Penc, age_ass
from datamap import levels_datamap
from sklearn.preprocessing import StandardScaler
import logging
import utils as ut
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Configure the logging, logging to file.
# logging.basicConfig(level=logging.INFO)


class DataFeatures:
    """ Each instance is initialized with the desired level
    and the dictionary with the instrument tables, as dataframes,
    from the database. A dataframe stores the datamap
    for feature selection.
    """

    def __init__(self, level, df_dict):
        self.level = level
        self.df_dict = df_dict
        dm_df = levels_datamap(df_dict)  # class variable
        lev_dict = {}
        for ins, df in df_dict.items():
            lev_dict[ins] = df[df.columns[pd.Series(dm_df.loc[ins,
                                                              level],
                                                    dtype='bool')]]
        self.lev_dict = lev_dict

    def create_level_tokens(self):
        """Transforms instrument values into words joining instrument name,
        scale/subscale and score. Returns a dictionary of token dataframes
        per instrument according to level. It also returns
        the correspondent vocabulary of terms.

        Returns
        -------
        dictionary
            {key: instrument, value: list of token lists}
        dictionary
            {key: word, value: int}
        """
        logging.info(f"Building token dataframes and vocabulary for level {self.level}.")

        # Create token strings to populate behr dictionary and vocabulary
        behr_tkns = {}
        lev_vocab = set()
        for ins, df in self.lev_dict.items():
            # for _, row in df.iterrows():
            #     # The first two positions of each vector of tokens store a
            #     # Penc dataclass and the assessment age.
            #     token = [Penc(sex=row.sex,
            #                   dob=penc[row.id_subj].dob,
            #                   doa_instrument=[(correct_datetime(row.date_ass), ins)]),
            #              age_ass(penc[row.id_subj].dob,
            #                      correct_datetime(row.date_ass))]
            for _, row in df.iterrows():
                # The first two positions of each vector of tokens store a
                # Penc dataclass and the assessment age.
                token = [Penc(sex=row.sex,
                              dob=row.date_birth.strftime("%d/%m/%Y"),
                              doa_instrument=[(row.date_ass.strftime("%d/%m/%Y"),
                                               ins)]),
                         age_ass(row.date_birth, row.date_ass)]
                for c in df.columns[4:]:
                    try:
                        if row[c] != '' and pd.notna(row[c]):
                            sig = self.__create_token(row, ins, c)
                            token.append('::'.join([sig, str(int(row[c]))]))
                            lev_vocab.update(['::'.join([sig, str(int(row[c]))])])
                        else:
                            pass
                    except ValueError:
                        pass
                behr_tkns.setdefault(row['id_subj'], list()).append(token)
        bt_to_idx = {trm: idx for idx, trm in enumerate(sorted(list(lev_vocab)))}
        idx_to_bt = {idx: trm for idx, trm in enumerate(sorted(list(lev_vocab)))}
        behr = {}
        for p_id, vect in behr_tkns.items():
            vect.sort(key=lambda x: x[1])
            for v in vect:
                behr.setdefault(p_id, list()).append(v)
        logging.info(f'Vocabulary size:{len(bt_to_idx)}')
        self.__save_vocab_behr(behr, bt_to_idx)

        return behr, (bt_to_idx, idx_to_bt)

    def create_level_features(self, missing_data_plot=False):
        """ If level is not 4 it returns an Error. For level 4 it returns
        a dataframe with patient ids as index and time-ordered features as columns.
        Missing values are NaN. Dataframe and vocabulary are saved to csv file.

        Returns
        -------
        dataframe
            Table with instrument scores at level 4 (at different times F1-F5)
            per subject.
        dataframe Scaled feature set with mean imputed missing values.
        """
        if self.level != 4:
            logging.error("create_level_features() is only available for level 4.")
            raise ValueError("create_level_features() attribute is only available for level 4.")
        else:
            # Create token strings as features
            feat_set = set()
            feat_tkns = {}
            for ins, df in self.lev_dict.items():
                for _, row in df.iterrows():
                    for c in df.columns[4:]:
                        try:
                            # if row[c] != '' and pd.notna(row[c]):
                            #     sig = self.__create_token(row, ins, c)
                            #     feat_tkns.setdefault(row['id_subj'], list()).append(
                            #         '::'.join([self.__aoa_to_tf(age_ass(penc[row.id_subj].dob,
                            #                                             correct_datetime(row.date_ass))),
                            #                    sig,
                            #                    str(int(row[c]))]))
                            #     feat_set.update(['::'.join([self.__aoa_to_tf(age_ass(penc[row.id_subj].dob,
                            #                                                          correct_datetime(row.date_ass))),
                            #                                 sig])])
                            if row[c] != '' and pd.notna(row[c]):
                                sig = self.__create_token(row, ins, c)
                                feat_tkns.setdefault(row['id_subj'], list()).append(
                                    '::'.join([self.__aoa_to_tf(age_ass(row.date_birth,
                                                                        row.date_ass)),
                                               sig,
                                               str(int(row[c]))]))
                                feat_set.update(['::'.join([self.__aoa_to_tf(age_ass(row.date_birth,
                                                                                     row.date_ass)),
                                                            sig])])
                            else:
                                pass
                        except ValueError:
                            pass
            feat_df = pd.DataFrame(columns=sorted(list(feat_set)),
                                   index=sorted(list(feat_tkns.keys())))
            for p_id, vect in feat_tkns.items():
                for tkn_val in vect:
                    tkn = tkn_val.split('::')
                    feat_df.loc[p_id, ['::'.join(tkn[:-1])]] = int(tkn[-1])
            feat_df.to_csv('./data/level-4/feature_data.csv')  # dump dataframe

            scaler = StandardScaler()
            feat_df_scaled = feat_df.fillna(feat_df.mean(), inplace=False)
            feat_df_scaled = pd.DataFrame(scaler.fit_transform(feat_df_scaled),
                                          columns=feat_df.columns,
                                          index=feat_df.index)
            missing_data = feat_df.isna().mean() * 100
            logging.info(f'Percentages of missing values for columns of feature data:\n{missing_data}')

            if missing_data_plot:
                rid_list = {}
                ins = set()
                for k, v in zip(missing_data.keys(),
                                missing_data):
                    ins.add(ut.shorten_names[k.split('::')[1]])
                    rid_list.setdefault(k.split('::')[0],
                                        dict()).setdefault(ut.shorten_names[k.split('::')[1]],
                                                           list()).append(v)
                df_dict = {}
                ins = list(ins)
                for i in sorted(ins):
                    df_dict[i] = []
                    for t in rid_list.keys():
                        try:
                            df_dict[i].append(np.mean(rid_list[t][i]))
                        except KeyError:
                            df_dict[i].append(np.nan)
                df = pd.DataFrame(df_dict, index=sorted(list(rid_list.keys())))
                logging.info(f'Mean percentages over items of missing values for feature data\n{df}')
                mask = df.isnull()
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(df, mask=mask, cmap='GnBu')
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.tick_params(length=0)
                plt.xticks(rotation=90)
                plt.savefig('./data/level-4/missing_feature_data.eps', format='eps',
                            dpi=200, bbox_inches='tight')

        return feat_df, feat_df_scaled

    def __save_vocab_behr(self, behr, bt_to_idx):
        """Saves behavioral EHRs and vocabulary of terms at the level specified
        to .csv file in a new data folder according to level.

        Parameters
        ----------
        behr: dictionary
            {key:pid, value:list(list of terms for each assessment)}
        bt_to_idx: dictionary
            Dictionary with behavioral terms as keys and idx as values
        """
        os.makedirs('./data/level-{0}'.format(self.level),
                    exist_ok=True)
        with open(os.path.join('./data/level-{0}'.format(self.level),
                               'cohort-behr.csv'), 'w') as f:
            wr = csv.writer(f)
            wr.writerow(['ID_SUBJ', 'AOA', 'TERM'])
            for pid, seq in behr.items():
                for s in seq:
                    wr.writerow([pid, s[1]] + [bt_to_idx[s[idx]]
                                               for idx in range(2, len(s))])
        with open(os.path.join('./data/level-{0}'.format(self.level),
                               'bt_to_idx.csv'), 'w') as f:
            wr = csv.writer(f)
            wr.writerow(["TERM", "LABEL"])
            for bt, idx in bt_to_idx.items():
                wr.writerow([bt, idx])

    @staticmethod
    def __create_token(row, ins, c):
        """Private custom-based function to modify and uniform dataset features.
        Must be modified when changing dataset. Returns token string and value. These
        objects must be joined for NLP behavioral embedding and kept separate for
        feature dataset.

        Parameters
        ----------
        row: pandas Series
            Row corresponding to a patient assessment
        ins: str
            Instrument considered
        c: str
            Name of the instrument item considered

        Returns
        -------
        str
            String of the form instrument::item
        """
        if bool(re.match('ados', ins)):
            if bool(re.search("\.d1|\.d2|\.b1|d1|d2|b1|"
                              "comparison_score|"
                              "sa_tot|rrb_tot|sarrb_tot|"
                              "\.sa_tot|\.rrb_tot",
                              c)):
                if len(c.split('.')) > 1:
                    token = '::'.join(['ados',
                                       c.split('.')[1]])
                else:
                    token = '::'.join(['ados',
                                       c])
            else:
                if len(c.split('.')) > 1:
                    token = '::'.join([ins,
                                       c.split('.')[1]])
                else:
                    token = '::'.join([ins,
                                       c])
        elif bool(re.match('psi', ins)):
            token = '::'.join([ins,
                               row['parent'].lower(),
                               c])
        elif bool(re.match('vinel|srs', ins)):
            token = '::'.join([ins, 'caretaker', c])
        elif bool(re.match('wa|wi|wp', ins)):
            token = '::'.join(['wechsler', c])
        else:
            token = '::'.join([ins,
                               c])
        return token

    @staticmethod
    def __aoa_to_tf(aoa):
        """Returns the time period from the age of assessment

        Parameters
        ----------
        aoa: float
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
