import csv
import os
import re
import pandas as pd
from dataset import Penc, age_ass
from datamap import connect_api
import logging

# Configure the logging, logging to file.
logging.basicConfig(level=logging.INFO)


class DataFeatures:
    dm_df = connect_api()  # class variable

    def __init__(self, level, df_dict):
        self.level = level
        lev_dict = {}
        for ins, df in df_dict.items():
            if bool(re.search('wa|wi|wp', ins)):
                lev_dict['wechsler'] = df[df.columns[pd.Series(self.dm_df.loc[ins,
                                                                              [level]],
                                                               dtype=bool)]]
            else:
                lev_dict[ins] = df[df.columns[pd.Series(self.dm_df.loc[ins,
                                                                       [level]],
                                                        dtype=bool)]]
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
            for p_id, row in df.iterrows():
                # The first two positions of each vector of tokens store a
                # Penc dataclass and the assessment age.
                token = [Penc(sex=row.sex,
                              dob=row.date_birth,
                              doa_instrument=[(row.date_ass, ins)]),
                         age_ass(row.date_birth, row.date_ass)]
                for c in df.columns[2:]:
                    if row[c] != '':
                        sig = self.__create_token(row, ins, c)
                        token.append('::'.join([sig, str(row[c])]))
                        lev_vocab.update('::'.join([sig, str(row[c])]))
                    else:
                        pass
                behr_tkns.setdefault(p_id, list()).append(token)
        bt_to_idx = {trm: idx for idx, trm in enumerate(sorted(list(lev_vocab)))}
        idx_to_bt = {idx: trm for idx, trm in enumerate(sorted(list(lev_vocab)))}
        behr = {}
        for p_id, vect in behr_tkns.items():
            for v in vect.sort(lambda x: x[1]):
                behr.setdefault(p_id, list()).append(v)
        self.__save_vocab_behr(behr, bt_to_idx)

        return behr, (bt_to_idx, idx_to_bt)

    def create_level_features(self):
        """ If level is not for it returns an Error. For level 4 it returns
        a dataframe with patient ids as index and time-ordered features as columns.
        Missing values are NaN. Dataframe and vocabulary are saved to csv file.

        Returns
        -------
        dataframe
            Table with instrument scores at level 4 (at different times F1-F5)
            per subject.
        """
        if self.level != 4:
            logging.error("create_level_features() is only available for level 4.")
            return
        else:
            # Create token strings as features
            feat_set = set()
            feat_tkns = {}
            for ins, df in self.lev_dict.items():
                for p_id, row in df.iterows():
                    for c in df.columns[2:]:
                        if row[c] != '':
                            sig = self.__create_token(row, ins, c)
                            feat_set.update('::'.join([self.__aoa_to_tf(row.ass_date),
                                                       sig]))
                            feat_tkns.setdefault(p_id, list()).append('::'.join([self.__aoa_to_tf(row.ass_date),
                                                                                 sig,
                                                                                 str(row[c])]))
                        else:
                            pass
            feat_df = pd.DataFrame(columns=sorted(list(feat_set)),
                                   index=sorted(list(feat_tkns.keys())))
            for p_id, vect in feat_tkns.items():
                for tkn_val in vect:
                    tkn = tkn_val.split('::')
                    feat_df.loc[p_id, ['::'.join(tkn[:-1])]] = int(tkn[-1])
            feat_df.to_csv('./data/level-4/feature_data.csv')  # dump dataframe

        return feat_df

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
        with open(os.path.join('/data/level-{0}'.format(self.level),
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
        int
            Score of the patient for that item
        """
        if bool(re.search('ados', ins)):
            if bool(re.search("\.d1|\.d2|\.b1|d1|d2|b1|"
                              "comparison_score|"
                              "sa_tot|rrb_tot|sarrb_tot|"
                              "\.sa_tot|\.rrb_tot",
                              c)):
                if len(c.split('.')) > 1:
                    token = '::'.join(['ados',
                                       c.split('.')[1]])  # maybe private function???
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
        elif bool(re.search('psi', ins)):
            token = '::'.join([ins,
                               row['caregiver'],
                               c])
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
