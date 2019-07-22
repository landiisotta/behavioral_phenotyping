# Behavioral phenotyping project

Behavioral data embeddings for the stratification of individuals
with neurodevelopmental conditions.

### Abstract

> Paper abstract here 

Application to data from individuals with Autism Spectrum Condition (ASC)
from the laboratory of Observation, Diagnosis and Education (ODFLab).

### Technical Requirements

```
Python 3.6+

R 3.4+
```

The full list of required Python Packages is available in `requrirements.txt` file. It is possible
to install all the dependency by:

```bash
$ pip install -r requirements.txt 
```

## Behavioural Phenotyping Pipeline (TLDR ;))

A complete example of the _Behavioural Phenotype Stratification_ is available 
as Jupyter notebook:

```
jupyter notebook behavioral_phenotyping_pipeline.ipynb
```

### Documentation (at a glance)

The code is structured into multiple modules (`.py` files), including algorithms and methods 
for the multiple steps of the pipeline:

* `dataset.py`: Connect to the database and dump data
* `features.py`: Return vocabulary and dictionary of behavioral *EHRs* for each of the 4 possible depth level. Returns 
dataset with quantitative scores for level 4 features
* `pt_embedding.py`: Perform TFIDF for patient embeddings; Glove embeddings on words and average them out for patient embeddings
* `clustering.py`: Perform Hierachical Clustering on XXX data (embeddings, quantitative 4th level features)
* `visualization.py`: Visualize results (e.g. _scatterplot & dendrogram_ for sub-cluster visualization; 
heatmap for inspection of quantitative scores between sub-clusters
* `basic_statistics.py`: Returns basic demographic statistics for dataset description


#### TODO: Paper, Poster, Conference Reference HERE

#### TODO: Credits and Acknowledgements HERE


