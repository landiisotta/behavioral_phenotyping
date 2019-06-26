from sqlalchemy import create_engine, MetaData, select
import datetime
from datetime import date, datetime
from time import time
from decimal import *
import csv
import os 
import pickle
import numpy as np
import matplotlib.pyplot as plt


"""
Main: create instrument tables
"""


# create tables -- exclude adults (based on ADOS-M4 presence)
# print data statistics
def main():
    SQLALCHEMY_CONN_STRING = 'mysql+pymysql://odflab:LAB654@192.168.132.114/odflab'
    DATA_FOLDER_PATH = os.path.expanduser('~/Documents/behavioral_phenotyping/data/')

    # connect to the database
    engine = create_engine(SQLALCHEMY_CONN_STRING)
    conn = engine.connect()

    # inspect the tables in the database
    # inspector = inspect(engine) if we want to inspect the tables (inspector.get_table_names())
    metadata = MetaData(engine, reflect=True)

    subject_list = []
    for table_name in metadata.tables:
        if table_name == 'ados-2modulo4':
            table_tmp = metadata.tables[table_name]
            s = select([table_tmp.c.id_subj])
            result = conn.execute(s)
            out_subj = list(set([r[0] for r in result.fetchall()]))
    # collect demographics
    for table_name in metadata.tables:
        table_tmp = metadata.tables[table_name]
        s = select([table_tmp.c.id_subj, table_tmp.c.date_birth, 
                    table_tmp.c.date_ass, table_tmp.c.sex])
        result = conn.execute(s)
        for r in result:
            if r[0] not in out_subj:
                try:
                    subject_list.append([r[0], r[1].strftime('%d/%m/%Y'), 
                                         r[2].strftime('%d/%m/%Y'), r[3], table_name])
                except AttributeError:
                    subject_list.append([r[0], r[1], r[2].strftime('%d/%m/%Y'), r[3],
                                        table_name])  # case datetime variable, None

    table_names = []
    tables = {}
    check_out = []
    for table_name in metadata.tables:
        table = metadata.tables[table_name]
        sql = select([c for c in table.c])
        result = conn.execute(sql)
        for r in result:
            if r[2] not in out_subj:
                tables.setdefault(table_name,
                                  list()).append(list(r[2::]))
            else:
                check_out.append(r[2])
        try:
            tables[table_name].insert(0, list(result.keys()))
        except KeyError:
            pass
    for t in tables.values():
        for l in t:
            for idx, el in enumerate(l):
                if type(el) is datetime:
                    l[idx] = el.strftime('%d/%m/%Y')
                elif type(el) is Decimal:
                    l[idx] = int(el)

    subj_values = {}
    header_tables = {}
    subj_demographics = {}
    for ins, meas in tables.items():
        header_tables.setdefault(ins, list()).extend(['eval_age'] + meas[0][3::])
        for m in meas[1::]:
            current_age, eval_age = age(m[2], m[3])
            subj_values.setdefault(m[0], list()).append([ins, eval_age] + m[1::])
            subj_demographics.setdefault(m[0], list()).append([current_age, eval_age, m[2], m[3], m[1]])
    for lab in subj_values.keys():
        subj_values[lab].sort(key=lambda x: x[1])
        subj_demographics[lab].sort(key=lambda x: x[1])

    # Save objects: (all tables, demographics + instruments, header tables, scores)
    data_dir = '-'.join(['odf-data', datetime.now().strftime('%Y-%m-%d-%H-%M-%S')])
    os.makedirs(os.path.join(DATA_FOLDER_PATH, data_dir))

    save_obj(tables, os.path.join(DATA_FOLDER_PATH, data_dir, 'odf-tables'))

    with open(os.path.join(DATA_FOLDER_PATH, data_dir, 'person-instrument.csv'), 'w') as f:
        wr = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        wr.writerow(['ID_SUBJ', 'DOB', 'DOA', 'SEX', 'INSTRUMENT'])
        for sl in subject_list:
            wr.writerow(sl)

    with open(os.path.join(DATA_FOLDER_PATH, data_dir, 
                          'header-tables.csv'), 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for h in header_tables:
            wr.writerow([h] + [c for c in header_tables[h]])

    with open(os.path.join(DATA_FOLDER_PATH, data_dir,
                           'person-scores.csv'), 'w') as f:
        wr = csv.writer(f, delimiter=',')
        for l, m in subj_values.items():
            for i in range(len(m)):
                wr.writerow([l] + [v for v in m[i]])

# Compute statistics
    period_span = []
    for l in subject_list:
        try:
            period_span.append(datetime.strptime(l[2], '%d/%m/%Y').date())
        except ValueError:
            pass
# correct wrong dates
    today = datetime.today()
    for idx, el in enumerate(period_span):
        if el.year == 2019 and el.month >= today.month:
            period_span[idx] = datetime(el.year, el.day, el.month).date()

    print("Period span: %s -- %s\n" % (min(period_span), max(period_span)))

# encounters are counted per year
    n_instrument = {el[0]: 0 for el in subject_list}
    n_encounter = {el[0]: set() for el in subject_list}
    for el in subject_list:
        n_instrument[el[0]] += 1
        n_encounter[el[0]].add(el[2].split('/')[2])

    plt.figure(figsize=(40,20))
    plt.bar(list(n_encounter.keys()), [len(el) for el in n_encounter.values()])
    plt.tick_params(axis='x', rotation=90)
    plt.tick_params(axis='y', labelsize=30)
    plt.savefig(os.path.join(DATA_FOLDER_PATH, data_dir, 'n-encounter.png'))

    with open(os.path.join(DATA_FOLDER_PATH, data_dir,
                           'person-demographics.csv'), 'w') as f:
        wr = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        wr.writerow(['ID_SUBJ', 'DOB', 'DOFA', 'SEX', 'N_ENC']) # DFOA = date of first assessment
        for s, sd in subj_demographics.items():
            wr.writerow([s] + sd[0][2::] + [len(n_encounter[s])])

    print("Average number of assessments: {0}".format(np.mean(list(n_instrument.values()))))
    print("Median number of assessments: {0}".format(np.median(list(n_instrument.values()))))
    print("Maximum number of assessments: {0}".format(max(list(n_instrument.values()))))
    print("Minimum number of assessments: {0}\n".format(min(list(n_instrument.values()))))

    print("Average number of encounters: {0}".format(np.mean([len(en) for en in list(n_encounter.values())])))
    print("Median number of assessments: {0}".format(np.median([len(en) for en in list(n_encounter.values())])))
    print("Maximum number of assessments: {0}".format(max([len(el) for el in list(n_encounter.values())])))
    print("Minimum number of assessments: {0}\n".format(min([len(el) for el in list(n_encounter.values())])))

    print("Instrument list")
    ins = set([el[-1] for el in subject_list])
    for name in ins:
        print("{0}".format(name))
    print('\n')

    age = [vec[0][0] for vec in subj_demographics.values()]
    sex = [vec[0][-1] for vec in subj_demographics.values()]
    print("Mean age of the subjects: {0} -- Standard deviation: {1}".format(np.mean(np.array(age)),
                                                                        np.std(np.array(age))))
    print("N Female: {0} -- N Male: {1}\n".format(sex.count("Femmina"), sex.count("Maschio")))


if __name__ == "__main__":
    start = time()
    main()
    print("Processing time %d" % round(time() - start, 2))

"""
Private Functions
"""


# save pickle obj
def _save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


"""
Functions
"""


# Compute age of assessment and current age
# from birth date and assessment date
def age(birthDate, assessmentDate):
    days_in_year = 365.2425
    try:
        assDate = datetime.strptime(assessmentDate, '%d/%m/%Y').date()
        bDate = datetime.strptime(birthDate, '%d/%m/%Y').date()
        assAge = (assDate - bDate).days / days_in_year
    except TypeError:
        bDate = datetime.strptime(birthDate, '%d/%m/%Y').date()
        assAge = -1
    currentAge = (date.today() - bDate).days / days_in_year
    return currentAge, assAge
