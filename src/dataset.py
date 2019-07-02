from sqlalchemy import create_engine, MetaData, select
import datetime
from datetime import date, datetime
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import utils as ut
import pandas as pd

# create new directory or point to an existing one
data_dir = '-'.join(['odf-data', datetime.now().strftime('%Y-%m-%d-%H-%M-%S')])
os.makedirs(os.path.join(ut.DATA_FOLDER_PATH, data_dir), exist_ok=True)


# define Patient class with demographic info
class Patient:
    def __init__(self,
                 pid: str,
                 bdate: datetime,
                 sex: str):
        self.pid = pid
        self.bdate = bdate
        self.sex = sex


class Pencounters(Patient):
    def __init__(self, pid, bdate, sex,
                 instr_and_ass: list()):
        self._instr_and_ass = instr_and_ass
        super().__init__(pid, bdate, sex)

    def add_encounter(self, inst, ass_date):
        self._instr_and_ass.append((inst, ass_date))
        self._instr_and_ass.sort(key=lambda x: x[1])

    @property
    def n_enc(self):
        # one encounter corresponds to an assessment year
        ass_yr = [el[1].split('/')[2] for el in self._instr_and_ass]
        return len(set(ass_yr))

    def list_ass(self):
        return self._instr_and_ass


def access_db():
    """ Access the database, saves the tables and the subject demographics
    Returns
    -------
    list
        list of Patient class objects
    dictionary
              patient_id: Pencounters class object
    """

    # connect to the database
    engine = create_engine(ut.SQLALCHEMY_CONN_STRING)
    conn = engine.connect()
    # inspect the tables in the database
    # inspector = inspect(engine) if we want to inspect the tables (inspector.get_table_names())
    metadata = MetaData(engine, reflect=True)

    # create patient list
    subject_list = []
    # create encounter list
    encounter_dict = {}
    # create dictionary of tables
    df_dict = {}

    adult_data_table = metadata.tables['ados-2modulo4']
    q = select([adult_data_table.c.id_subj])
    adult_ids = conn.execute(q)
    adult_subj = list(set([str(aid[0]) for aid in adult_ids.fetchall()]))

    for table_name in metadata.tables:
        data_table = metadata.tables[table_name]
        q = select([data_table.c.id_subj,
                    data_table.c.date_birth,
                    data_table.c.date_ass,
                    data_table.c.sex])
        patients = conn.execute(q)
        for patient_dem in patients:
            pid, bdate, assdate, sex = patient_dem
            if not(type(bdate) == str):
                bdate = bdate.strftime('%d/%m/%Y')
            if not(type(assdate) == str):
                assdate = assdate.strftime('%d/%m/%Y')
            if pid not in adult_subj:
                subject_list.append(Patient(pid, bdate, sex))
            subject_list.append(Patient(pid, bdate, sex))
            if pid not in encounter_dict:
                encounter_dict[pid] = Pencounters(pid, bdate, sex, [(table_name, assdate)])
            else:
                encounter_dict[pid].add_encounter(table_name, assdate)
        # create dataframes from tables and save them to csv
        q_dump = select([c for c in data_table.c])
        dump = conn.execute(q_dump)
        data_ls = [row[2::] for row in dump if row[2] not in adult_subj]
        df_dict[table_name] = pd.DataFrame(data_ls,
                                           columns=dump.keys()[2::])
        df_dict[table_name].to_csv(os.path.join(ut.DATA_FOLDER_PATH, data_dir,
                                                '-'.join([table_name, 'table.csv'])),
                                   header=df_dict[table_name].columns)
    # Save demographics: (all tables, demographics + instruments)
    with open(os.path.join(ut.DATA_FOLDER_PATH, data_dir,
                           'person-instrument.csv'), 'w') as f:
        wr = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        wr.writerow(['ID_SUBJ', 'DOB', 'SEX', 'N_ENC', 'DOA', 'INSTRUMENT'])
        for pid, dem in encounter_dict.items():
            for tup in dem.list_ass():
                wr.writerow([pid, dem.bdate, dem.sex, dem.n_enc] + list(tup))
    return subject_list, encounter_dict


def demographics(subjects, encounters):
    """Print demographic statistics and save the histogram of the n_encounter distribution

    Parameters
    ----------
    list
        list of Patient class objects
    dictionary
              patient_id: Pencounters() class object
    """
    # Compute statistics
    period_span = []
    set_ins = set()
    for pid in encounters:
        period_span.extend([el[1] for el in encounters[pid].list_ass()])
        set_ins.update([el[0] for el in encounters[pid].list_ass()])

    # correct wrong dates
    today = datetime.today()
    for idx, el in enumerate(period_span):
        if el.split('/')[2] == 2019 and el.month >= today.month:
            period_span[idx] = datetime(el.split('/')[2],
                                        el.split('/')[0],
                                        el.split('/')[1]).date()

    print("Period span: %s -- %s\n" % (min(period_span), max(period_span)))

    # plot histogram with number of encounters
    plt.figure(figsize=(40, 20))
    plt.bar(list(encounters.keys()),
            [el.n_enc for el in encounters.values()])
    plt.tick_params(axis='x', rotation=90)
    plt.tick_params(axis='y', labelsize=30)
    plt.savefig(os.path.join(ut.DATA_FOLDER_PATH, data_dir, 'n-encounter.png'))

    list_n_ass = [len(encounters[pat].list_ass()) for pat in encounters]
    print("Average number of assessments: {0:.3f}".format(np.mean(list_n_ass)))
    print("Median number of assessments: {0}".format(np.median(list_n_ass)))
    print("Maximum number of assessments: {0}".format(max(list_n_ass)))
    print("Minimum number of assessments: {0}\n".format(min(list_n_ass)))

    list_n_enc = [encounters[pat].n_enc for pat in encounters]
    print("Average number of encounters: {0:.3f}".format(np.mean(list_n_enc)))
    print("Median number of assessments: {0}".format(np.median(list_n_enc)))
    print("Maximum number of assessments: {0}".format(max(list_n_enc)))
    print("Minimum number of assessments: {0}\n".format(min(list_n_enc)))

    print("Instrument list:")
    for name in set_ins:
        print("{0}".format(name))
    print('\n')

    age_ls = [age(pat.bdate) for pat in subjects]
    sex_ls = [pat.sex for pat in subjects]
    print("Mean age of the subjects: {0} -- Standard deviation: {1}".format(np.mean(age_ls),
                                                                            np.std(age_ls)))
    print("N Female: {0} -- N Male: {1}\n".format(sex_ls.count("Femmina"), sex_ls.count("Maschio")))


"""
Functions
"""


# Compute age of assessment and current age
# from birth date and assessment date
def age(birthDate):
    days_in_year = 365.2425
    bDate = datetime.strptime(birthDate, '%d/%m/%Y').date()
    currentAge = (date.today() - bDate).days / days_in_year
    return currentAge