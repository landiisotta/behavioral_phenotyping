import os
import logging
import httplib2
import pandas as pd
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from gsheets import Sheets
from utils import home_dir, select_clm, SCOPES, APPLICATION_NAME, CLIENT_SECRET_FILE, FOLDER_ID
import numpy as np

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO)
try:
    import argparse

    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

sheets = Sheets.from_files('client_secrets.json', 'storage.json')


def connect_api():
    """ Function that connects to Spreadsheet API and select the table columns
    needed to build feature sets on 4 levels.

    Return
    ------
    dataframe
        Rows are the tables names, Columns the 4 level identifiers and elements are arrays of integers
        where 0 = drop the column, 1 = keep the column.
    """
    credentials = _get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('drive', 'v3', http=http)

    directories = service.files().list(
        q="'{}' in parents and trashed=false and mimeType = 'application/vnd.google-apps.folder'".format(FOLDER_ID),
        pageSize=100, fields="nextPageToken, files(id, name)").execute()
    subdir = directories.get('files', [])

    if subdir:
        for directory in subdir:
            results = service.files().list(
                q="'{}' in parents and trashed=false and mimeType = 'application/vnd.google-apps.spreadsheet'".format(
                    directory['id']),
                pageSize=100, fields="nextPageToken, files(id, name)").execute()
            items = results.get('files', [])
            cselect_dict = {}
            insname_list = []
            for gsheet in items:
                insname_list.append(gsheet['name'])
                for lev in range(1, 5):
                    cselect_dict[lev] = {}
                    select_clm = _col_select(_get_gspread_content(gsheet['id']))
                    cselect_dict[lev][gsheet['name']] = select_clm
    insname_list = sorted(insname_list)
    dm_dict = {}
    for lev in cselect_dict:
        dm_dict[lev] = [cselect_dict[lev][insname] for insname in insname_list]

    return pd.DataFrame(dm_dict, index=insname_list)


"""
Private Functions
"""


def _get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
    --------
        Credentials, the obtained credential.
    """
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'drive-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else:  # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        logging.info('Storing credentials to {}'.format(credential_path))
    return credentials


def _get_gspread_content(gspread_id):
    """Get the content of the spread (given the id)

    Parameters
    ----------
    gspread_id: str
        ID of the gspread.

    Returns
    -------
        Columns of Pandas Data Frame, None otherwise.
    """

    s = sheets[gspread_id]
    if s:
        try:
            sheet_df = s.sheets[0].to_frame(header=1,
                                            parse_dates=['date_birth', 'date_ass'],
                                            index_col='id',
                                            converters={
                                                'form_info': lambda t: pd.to_datetime(t,
                                                                                      format="%d/%m/%Y %H.%M.%S")},
                                            infer_datetime_format=True).drop('form_info',
                                                                             axis=1)
        except KeyError as error:
            sheet_df = None
            logging.error('An error occurred: %s' % error)
    else:
        sheet_df = None
    return sheet_df.columns


def _col_select(lev, instrument,
                clm_names):
    """ Given a table and a peth level, it returns a boolean array
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
    cselect_list = np.array([1, 1, 1], dtype=int)
    for col in clm_names[4:]:
        if col in select_clm[lev][instrument]:
            cselect_list = np.append(cselect_list, [int(1)])
        else:
            cselect_list = np.append(cselect_list, [int(0)])

    return cselect_list
