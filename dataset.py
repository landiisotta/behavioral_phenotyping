from sqlalchemy import create_engine, MetaData
import datetime
from datetime import datetime
import csv
import os
import utils as ut
import pandas as pd
import logging
from dataclasses import dataclass
from basic_statistics import DataStatistics


# Dataclasses to store patient demographics,
# and patient info on encounters.
@dataclass
class Pinfo:
    sex: str
    dob: str
    n_enc: int = 0


@dataclass
class Penc:
    dob: str
    doa_instrument: list()

    def count_enc(self):
        yr_enc = list(map(lambda x: x[0].split('/')[2],
                          self.doa_instrument))
        return len(set(yr_enc))


# Configure the logging, logging to file.
logging.basicConfig(level=logging.INFO,
                    filename='/logs/dataset.log',
                    filemode='w')

# Create new directory or point to an existing one to store data.
data_dir = 'odf-data'
data_path = os.path.join(ut.DATA_FOLDER_PATH, data_dir)
os.makedirs(data_path, exist_ok=True)
runtime_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logging.info(f'{runtime_date} created ../data/odf-data folder for returned objects')


def access_db():
    """ Access the database and dump tables.

    Returns
    -------

    dictionary
        {key=table_name, value=pandas dataframe}
    """
    # connect to the database
    engine = create_engine(ut.SQLALCHEMY_CONN_STRING)
    conn = engine.connect()
    logging.info('Connection to DB established')
    # inspect the tables in the database
    metadata = MetaData(engine, reflect=True)

    logging.info('Dumping all tables')
    df_tables = {}
    for table_name in metadata.tables:
        df_tables[table_name] = pd.read_sql_table(table_name,
                                                  con=conn,
                                                  index_col='id').drop('form_info',
                                                                       axis=1)
    return df_tables


def data_wrangling(tables_dict):
    """ Drop excluded subjects and tables

    Parameters
    ----------
    tables_dict: dictionary
        dictionary with dumped tables from DB

    Returns
    -------
    dictionary
        reduced dictionary without excluded tables and subjects (rows)
    """
    adult_subj = tables_dict['ados-2modulo4'].id_subj.unique()
    logging.info(f'Dropped {len(adult_subj)} subjects')

    # names of the tables to drop from the dictionary
    tb_drop = ['ados-2modulo4',
               'emotionalavailabilityscales']

    tb_dict_rid = {}
    for tb_name, df in tables_dict.items():
        if tb_name not in tb_drop:
            row_drop = ~(df['id_subj'].isin(adult_subj))
            tb_dict_rid[tb_name] = df.loc[row_drop]

    return tb_dict_rid


def cohort_info(tables_dict):
    """Store instances of Pinfo and Penc  classes in dictionaries

    Parameters
    ----------
    tables_dict: dictionary
        dictionary with data tables

    Returns
    -------
    dictionary
        {keys=pid, values=Pinfo instances}
    dictionary
        {keys=pid, values=Penc instances}
    """
    demog_dict = {}
    enc_dict = {}
    for tn, df in tables_dict.items():
        for _, row in df.iterrows():
            ass_date = _correct_datetime(row.date_ass)
            birth_date = _correct_datetime(row.date_birth)
            if row.id_subj in enc_dict:
                enc_dict[row.id_subj].doa_instrument.append((ass_date, tn))
            else:
                enc_dict[row.id_subj] = Penc(dob=birth_date,
                                             doa_instrument=[(ass_date,
                                                              tn)])
                demog_dict.setdefault[row.id_subj] = Pinfo(sex=row.sex,
                                                           dob=birth_date)
    for pid in demog_dict:
        demog_dict[pid].n_enc = enc_dict[pid].count_enc()
    # dump info to csv files
    _dump_info(demog_dict, enc_dict)
    # save log with statistics
    DataStatistics.compute(data_dir)
    return demog_dict, enc_dict


"""
Private functions
"""


def _age(dob, doa):
    """
    Parameters
    ----------
    dob: pandas Timestamp
    doa: pandas Timestamp

    Return
    ------
    float
        age of assessment
    """
    days_in_year = 365.2425
    aoa = (dob - doa).days / days_in_year
    return aoa


def _correct_datetime(date_ts):
    """
    Parameters
    ----------
    date_ts: pandas Timestamp

    Returns
    -------
    str
        strftime %d/%m/%Y
    """
    # correct wrong dates
    today = datetime.today()
    if date_ts.year == today.year and date_ts.month >= today.month:
        corrected_date = pd.Timestamp(year=date_ts.year,
                                      month=date_ts.day,
                                      day=date_ts.month)
    else:
        corrected_date = date_ts

    return corrected_date.strftime("%d/%m/%Y")


def _dump_info(demog_info, enc_info):
    """Save csv file with demographic and encounter info

    Parameters
    ----------
    demog_info: dictionary
        {keys=pid, values=Pinfo instances}
    enc_info: dictionary
        {keys=pid, values=Penc instances}
    """
    logging.info("Saving csv files on subject info and subject encounters")
    with open(os.path.join(ut.DATA_FOLDER_PATH, data_dir,
                           'person-encounters.csv'), 'w') as f:
        wr = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        wr.writerow(['ID_SUBJ', 'DOB', 'DOA', 'AOA', 'INSTRUMENT'])
        for pid, penc in enc_info.items():
            for tup in penc.doa_instrument:
                wr.writerow([pid, penc.dob, tup[0],
                             _age(penc.dob, tup[0]),
                             tup[1]])
    with open(os.path.join(ut.DATA_FOLDER_PATH, data_dir,
                           'person-demographics.csv'), 'w') as f:
        wr = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        wr.writerow(['ID_SUBJ', 'SEX', 'DOB', 'N_ENC'])
        for pid, pinfo in demog_info.items():
            wr.writerow([pid, pinfo.sex, pinfo.dob,
                         pinfo.n_enc])
