import matplotlib
import os

home_dir = os.path.expanduser('~')
# Address to connect to the database on UniTN server.
SQLALCHEMY_CONN_STRING = 'mysql+pymysql://odflab:LAB654@192.168.132.114/odflab'
# Project's folder path.
DATA_FOLDER_PATH = '/Users/ilandi/Documents/behavioral_phenotyping/data/'

# Drive folder.
FOLDER_ID = '0B98jN9m5fFxsNnVBVm1CRXBvOWM'
# Google API connection
SCOPES = 'https://www.googleapis.com/auth/drive.metadata.readonly'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Drive API Python Quickstart'

"""
Parameters
"""

# Maximum and minimum number of clusters.
min_cl = 3
max_cl = 15

# Number of clustering iterations.
n_iter = 100
# Subsampling ratio for clustering.
subsampl = 0.8

# Glove/Word2Vec parameters.
n_epoch = 50
batch_size = 205
learning_rate = 0.01

# Dimension of TruncatedSVD and Glove/Word2vec word embeddings.
n_dim = 10

# Vocabulary and behavioral ehr file names.
file_names = {'vocab': 'cohort-vocab.csv',
              'behr': 'cohort-behr.csv'}

# Models to validate per depth level.
model_vect = {'0': ['tfidf', 'glove'],
              '1': ['tfidf', 'glove'],
              '2': ['tfidf', 'glove'],
              '3': ['tfidf', 'glove'],
              '4': ['tfidf', 'glove', 'feat']}

# Data visualization colors.
col_dict = matplotlib.colors.CSS4_COLORS
c_out = ['bisque', 'mintcream', 'cornsilk', 'lavenderblush', 'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
         'beige', 'powderblue', 'floralwhite', 'ghostwhite', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
         'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue',
         'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'linen', 'palegoldenrod', 'palegreen',
         'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'mistyrose', 'lemonchiffon', 'lightblue',
         'seashell', 'white', 'blanchedalmond', 'oldlace', 'moccasin', 'snow', 'darkgray', 'ivory', 'whitesmoke']


select_clm = {
    1: {'ados-2modulo1': ['al1.a2', 'al1.a8', 'al1.b1', 'al1.b3', 'al1.b4',
                          'al1.b5', 'al1.b9', 'al1.b10', 'al1.b11', 'al1.b12',
                          'al2.a2', 'al2.a7', 'al2.a8', 'al2.b1',
                          'al2.b3', 'al2.b4', 'al2.b5', 'al2.b9', 'al2.b10',
                          'al2.b12', 'al1.a3', 'al1.d1', 'al1.d2',
                          'al1.d4', 'al2.a5', 'al2.d1', 'al2.d2',
                          'al2.d4'],
        'ados-2modulo2': ['a6', 'a7', 'b1', 'b2', 'b3',
                          'b5', 'b6', 'b8', 'b11', 'b12',
                          'a4', 'd1', 'd2', 'd4'],
        'ados-2modulo3': ['a7', 'a8', 'a9', 'b1', 'b2', 'b4', 'b7',
                          'b9', 'b10', 'b11', 'a4', 'd1',
                          'd2', 'd4'],
        'ados-2modulotoddler': ['al1.a2', 'al1.a8',
                                'al1.b1', 'al1.b4', 'al1.b5', 'al1.b6', 'al1.b12',
                                'al1.b13', 'al1.b14', 'al1.b15',
                                'al2.a7', 'al2.b1', 'al2.b4', 'al2.b5',
                                'al2.b7', 'al2.b8', 'al2.b9',
                                'al2.b13', 'al2.b15', 'al2.b16b', 'al2.b18',
                                'al1.a3', 'al1.d1', 'al1.d2',
                                'al1.d5', 'al2.d1', 'al2.d2',
                                'al2.d5'],
        'griffithsmentaldevelopmentscales': ['q_A', 'q_B', 'q_C', 'q_D', 'q_E', 'q_F'],
        'leiterinternationalperformancescale-revised': ['scaled_fg', 'scaled_fc', 'scaled_so',
                                                        'scaled_rp', 'sumScaled_fr'],
        'psi-sf': ['parent', 'raw_pd', 'raw_pcdi', 'raw_DC'],
        'srs': ['parent', 'raw_sa', 'raw_scog',
                'raw_scom', 'raw_sm', 'raw_rirb'],
        'vineland-ii': ['caregiver', 'scaled_rec', 'scaled_exp', 'scaled_wri', 'scaled_per',
                        'scaled_dom', 'scaled_community', 'scaled_ir', 'scaled_play', 'scaled_cs',
                        'scaled_gm', 'scaled_fm'],
        'wais-iv': ['scaled_bd', 'scaled_si', 'scaled_mr', 'scaled_vc', 'scaled_ss',
                    'scaled_oa', 'scaled_in', 'scaled_cd', 'scaled_co',
                    'scaled_pc'],
        'wais-r': ['scaled_bd', 'scaled_si', 'scaled_vc',
                   'scaled_oa', 'scaled_in', 'scaled_co',
                   'scaled_pc'],
        'wisc-iii': ['scaled_pc', 'scaled_in', 'scaled_cd', 'scaled_si', 'scaled_bd', 'scaled_oa',
                     'scaled_co', 'scaled_ss'],
        'wisc-iv': ['scaled_bd', 'scaled_si', 'scaled_pcn', 'scaled_cd', 'scaled_vc',
                    'scaled_pc', 'raw_ca', 'scaled_ca', 'scaled_in'],
        'wppsi': ['scaled_in', 'scaled_vc', 'scaled_si',
                  'scaled_co', 'scaled_pc', 'scaled_bd'],
        'wppsi-iiifascia26-311': ['scaled_bd', 'scaled_in', 'scaled_oa'],
        'wppsi-iiifascia40-73': ['scaled_bd', 'scaled_in', 'scaled_vc', 'scaled_pcn',
                                 'scaled_ss', 'scaled_mr', 'scaled_cd',
                                 'scaled_co', 'scaled_pc', 'scaled_si',
                                 'scaled_oa']
        },
    2: {'ados-2modulo1': ['al1.sa_tot', 'al2.sa_tot',
                          'al1.rrb_tot', 'al2.rrb_tot'],
        'ados-2modulo2': ['sa_tot', 'rrb_tot'],
        'ados-2modulo3': ['sa_tot', 'rrb_tot'],
        'ados-2modulotoddler': ['al1.sa_tot', 'al2.sa_tot',
                                'al1.rrb_tot', 'al2.rrb_tot'],
        'griffithsmentaldevelopmentscales': ['q_A', 'q_B', 'q_C', 'q_D', 'q_E', 'q_F'],
        'leiterinternationalperformancescale-revised': ['scaled_fg', 'scaled_fc', 'scaled_so',
                                                        'scaled_rp', 'sumScaled_fr'],
        'psi-sf': ['parent', 'raw_pd', 'raw_pcdi', 'raw_DC'],
        'srs': ['parent', 'raw_sa', 'raw_scog',
                'raw_scom', 'raw_sm', 'raw_rirb'],
        'vineland-ii': ['caregiver', 'sum_CD', 'sum_DLSD', 'sum_SD', 'sum_MSD'],
        'wais-iv': ['sumScaled_VC', 'sumScaled_PR', 'sumScaled_WM',
                    'sumScaled_PS', 'sumScaled_FS'],
        'wais-r': ['sumScaled_V', 'sumScaled_P'],
        'wisc-iii': ['sumScaled_V', 'sumScaled_P',
                     'sumScaled_FS', 'sumScaled_VC', 'sumScaled_PO',
                     'sumScaled_FD', 'sumScaled_PS'],
        'wisc-iv': ['sumScaled_VC', 'sumScaled_PR',
                    'sumScaled_WM', 'sumScaled_PS', 'sumScaled_FS'],
        'wppsi': ['sumScaled_V', 'sumScaled_P',
                  'sumScaled_FS'],
        'wppsi-iiifascia26-311': ['sumScaled_V', 'sumScaled_P', 'sumScaled_FS',
                                  'sumScaled_GL'],
        'wppsi-iiifascia40-73': ['sumScaled_V', 'sumScaled_P', 'sumScaled_PS',
                                 'sumScaled_FS', 'sumScaled_GL']
        },
    3: {'ados-2modulo1': ['sarrb_tot', 'comparison_score'],
        'ados-2modulo2': ['sarrb_tot', 'comparison_score'],
        'ados-2modulo3': ['sarrb_tot', 'comparison_score'],
        'ados-2modulotoddler': ['sarrb_tot'],
        'griffithsmentaldevelopmentscales': ['GQ'],
        'leiterinternationalperformancescale-revised': ['composite_fr', 'BIQ'],
        'psi-sf': ['parent', 'raw_ts'],
        'srs': ['parent', 'raw_tot'],
        'vineland-ii': ['caregiver', 'standard_ABC'],
        'wais-iv': ['FSIQ'],
        'wais-r': ['FSIQ'],
        'wisc-iii': ['FSIQ'],
        'wisc-iv': ['FSIQ'],
        'wppsi': ['FSIQ'],
        'wppsi-iiifascia26-311': ['FSIQ'],
        'wppsi-iiifascia40-73': ['FSIQ']
        },
    4: {'ados-2modulo1': ['al1.sa_tot', 'al2.sa_tot', 'al1.rrb_tot',
                          'al2.rrb_tot', 'comparison_score'],
        'ados-2modulo2': ['sa_tot', 'rrb_tot', 'comparison_score'],
        'ados-2modulo3': ['sa_tot', 'rrb_tot', 'comparison_score'],
        'ados-2modulotoddler': ['al1.sa_tot', 'al2.sa_tot',
                                'al1.rrb_tot', 'al2.rrb_tot'],
        'griffithsmentaldevelopmentscales': ['q_A', 'q_B', 'q_C', 'q_D', 'q_E', 'q_F', 'GQ'],
        'leiterinternationalperformancescale-revised': ['sumScaled_fr', 'sumScaled_BIQ'],
        'psi-sf': ['parent', 'raw_ts'],
        'srs': ['parent', 'raw_rirb', 'raw_tot'],
        'vineland-ii': ['caregiver', 'sum_CD', 'standard_CD', 'sum_DLSD', 'standard_DLSD',
                        'sum_SD', 'standard_SD', 'sum_MSD', 'standard_MSD', 'standard_ABC'],
        'wais-iv': ['sumScaled_FS'],
        'wais-r': ['sumScaled_FS'],
        'wisc-iii': ['sumScaled_FS'],
        'wisc-iv': ['sumScaled_FS'],
        'wppsi': ['sumScaled_FS'],
        'wppsi-iiifascia26-311': ['sumScaled_FS'],
        'wppsi-iiifascia40-73': ['sumScaled_FS']
        }
}
