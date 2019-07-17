import pandas as pd
import utils as ut
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Create a custom logger, logging to file
logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.FileHandler('/logs/descriptive_statistics.log')
c_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(message)s')
c_handler.setFormatter(c_format)

# Add handlers to the logger
logger.addHandler(c_handler)


class DataStatistics:
    """Class for data statistics computation."""

    def compute(self, data_dir):
        """Compute basic statistics and save output to log file.

        Parameter
        ---------
        data_dir: str
            directory name where to save log file
        """
        pd.set_option('float_format', '{:.3f}'.format)

        dem = pd.read_csv(os.path.join(ut.DATA_FOLDER_PATH, data_dir,
                                       'person-demographics.csv'),
                          sep=',',
                          header=0)
        enc = pd.read_csv(os.path.join(ut.DATA_FOLDER_PATH, data_dir,
                                       'person-encounters.csv'),
                          sep=',',
                          header=0)
        dem['age'] = list(map(lambda x: _age(x), dem.dob.tolist()))

        logging.info('N of subjects: %d', len(dem.id_subj.unique()))
        logging.info('%s', pd.crosstab(dem.sex, columns='count'))
        logging.info('%s\n',
                     dem.describe())

        logging.info("Instrument list:")
        for ins in sorted(enc.instrument.unique()):
            logging.info('$s', ins)
        logging.info('%s\n',
                     enc.describe())
        # Consider assessment as number of administered instruments
        ass_dict = {}
        for _, row in enc.iterrows():
            ass_dict.setdefault(row.id_subj, list()).append(row.instrument)
        count_ass = {'pid': list(ass_dict.keys()),
                     'ass_count': [len(ass_dict[pid]) for pid in ass_dict]}
        logging.info("Assessment (i.e., administered instrument counts) statistics:")
        logging.info('%s\n', pd.DataFrame(count_ass).describe())

        # return period span
        logging.info(f'Period span: {enc.doa.min()} -- {enc.doa.max()}\n')

        # plot histogram with number of encounters
        plt.figure(figsize=(40, 20))
        plt.bar(dem.id_subj, dem.n_enc)
        plt.tick_params(axis='x', rotation=90)
        plt.tick_params(axis='y', labelsize=30)
        plt.savefig(os.path.join(ut.DATA_FOLDER_PATH,
                                 data_dir,
                                 'n-encounter.png'))
        plt.close()


"""
Private functions
"""


def _age(dob):
    """
    Parameters
    ----------
    dob: str
        date of birth in format %d/%m/%Y

    Return
    ------
    float
        age from birth date
    """
    days_in_year = 365.2425
    dt_dob = datetime.strptime(dob, '%d/%m/%Y')
    current_age = (datetime.today() - dt_dob).days / days_in_year
    return current_age
