import pandas as pd
import utils as ut
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Create a custom logger, logging to file
logger = logging.getLogger('descriptive_statistics')

# Create handlers
c_handler = logging.FileHandler('./logs/descriptive_statistics.log',
                                mode='w')
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
        dem['AGE'] = list(map(lambda x: self.__age(x), dem.DOB.tolist()))

        logger.info('N of subjects: %d\n', len(dem.ID_SUBJ.unique()))
        logger.info('%s\n', pd.crosstab(dem.SEX, columns='count'))
        logger.info('%s\n',
                     dem.describe())

        logger.info("Instrument list:")
        for ins in sorted(enc.INSTRUMENT.unique()):
            logger.info('%s', ins)
        logger.info('\n%s\n',
                     enc.describe())
        # Consider assessment as number of administered instruments
        ass_dict = {}
        for _, row in enc.iterrows():
            ass_dict.setdefault(row.ID_SUBJ, list()).append(row.INSTRUMENT)
        count_ass = {'pid': list(ass_dict.keys()),
                     'ass_count': [len(ass_dict[pid]) for pid in ass_dict]}
        logger.info("Assessment (i.e., administered instrument counts) statistics:")
        logger.info('%s\n', pd.DataFrame(count_ass).describe())

        # return period span
        logger.info(f'Period span: {enc.DOA.min()} -- {enc.DOA.max()}\n')

        # plot histogram with number of encounters
        plt.figure(figsize=(40, 20))
        plt.bar(dem.ID_SUBJ, dem.N_ENC)
        plt.tick_params(axis='x', rotation=90)
        plt.tick_params(axis='y', labelsize=30)
        plt.savefig(os.path.join(ut.DATA_FOLDER_PATH,
                                 data_dir,
                                 'n-encounter.png'))
        plt.close()

    @staticmethod
    def __age(dob):
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


