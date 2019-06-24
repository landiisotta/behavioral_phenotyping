# Behavioral phenotyping project
Behavioral data embeddings for the stratification of individuals
with neurodevelopmenta conditions. 

Application to data of individuals with Autism Spectrum Condition (ASC)
from the laboratory of Observation, Diagnosis and Education (ODFLab).

## Code TLDR
Project folder: `~/Documents/behavioral_phenotyping`

Scripts (`~/Documents/behavioral_phenotyping/src`)

- `create-tables.py`: create data tables from database, as for now adults are excluded
based on the presence/absence of ADOS-M4 scores. Preliminary statistics are reported.

User-defined variables: DATA\_FOLDER\_PATH = path to data string.
Output: files saved in new folder `~/Documents/behavioral_phenotypes/data/odf-data-date-time`.
(odf-tables.pkl, person-instrument.csv, header-tables.csv, person-scores.csv, n-encounter.png,
person-demographics.csv)

- `create-vocabulary.py`: create vocabulary from data. Choose from 4 possible levels.

Level 1: deepest level subtests;

Level 2: aggregation of subtest scores;

Level 3: general indices;

Level 4: indices and subtest of interest for phenotype profiling.

Output: data\_folder odf-data-datetime/level-1|2|3|4 
(cohort-behr.csv, cohort-vocab.csv)

Run with `run_tokenize-embed.sh`, user defined indir, outdir, level (from terminal).

- `create-feature-data.py`: create dataset with instrument values from level 4

Output: dataframes: long, wide, wide scaled with missing values imputed with mean.

Run with `run_create-feature-data.sh`, user define datadir (data folder level 4).

- `embedding.py`: run TFIDF and GloVe embeddings and save results.

Run with `run_tokenize-embed.sh` (same user-defined variables). 
