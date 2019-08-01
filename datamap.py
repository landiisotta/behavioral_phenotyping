import logging
import pandas as pd
from utils import select_clm
import numpy as np

flags = None
logger = logging.getLogger('datamap')


def levels_datamap(table_dict):
    """ Returns a dataframe with a boolean vector per
    instrument per level to select the columns correspondent
    to the desired level

    Parameters
    ----------
    table_dict: dict
        dictionary of tables (df) from the database already
        filtered
    Returns
    -------
    dataframe
        instruments x levels, each element is a boolean vector
    """
    cselect_dict = {}
    insname_list = []
    for table, df in table_dict.items():
        insname_list.append(table)
        for lev in range(1, 5):
            if table in select_clm[lev]:
                clm_list = _col_select(lev, table,
                                       df.columns)
                cselect_dict.setdefault(lev, list()).append(clm_list)
            else:
                logger.info("Not considered table {0}".format(table))
    selectcol_df = pd.DataFrame(cselect_dict,
                                index=insname_list).sort_index()
    return selectcol_df


"""
Private Functions
"""


def _col_select(lev, instrument,
                clm_names):
    """ Given a table and a depth level, it returns a boolean array
    storing the columns to select.

    Parameters
    ----------
    lev: int
        Level depth
    instrument: str
        Instrument name
    clm_names: Index object

    Returns
    -------
    array
        Array of integers with the columns to select
    """
    cselect_list = np.array([1, 1, 1, 1], dtype=int)
    for col in clm_names[4:]:
        if col in select_clm[lev][instrument]:
            cselect_list = np.append(cselect_list, [int(1)])
        else:
            cselect_list = np.append(cselect_list, [int(0)])
    return cselect_list
