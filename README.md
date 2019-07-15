# Behavioral phenotyping project

Behavioral data embeddings for the stratification of individuals
with neurodevelopmenta conditions.

### Abstract

> Paper abstract here 

Application to data of individuals with Autism Spectrum Condition (ASC)
from the laboratory of Observation, Diagnosis and Education (ODFLab).

## Requirements

```
python 3
```

## Behavioural Phenotyping Stratification Pipeline (tldr;)

A complete example of the _Behavioural Phenotyping Stratification_ is available 
as Jupyter notebook:

```
jupyter notebook phenotyping_stratification_pipeline.ipynb
```

### Documentation (at a glance)

The code is structured into multiple modules (`.py` files), including algorithms and methods 
for the multiple steps of the pipeline:

* `dataset.py`: Connect to the database and dump data
* `features.py`: Return vocabulary and dictionary of behavioral *ehars* for each of the 4 possible depth level.
* `pt_embedding.py`: Perform TFIDF for patient embeddings; Glove embeddings on words and average them out for patient embeddings
* `clustering.py`: Perform Hierachical Clustering on XXX data (embeddings?)
* `visualization.py`: Visualize results (e.g. _scatterplot & dendrogram_ for sub-cluster visualization; 
heatmap for inspection of quantitative scores between sub-clusters.


#### TODO: Paper, Poster, Conference Reference HERE

#### TODO: Credits and Acknowledgements HERE


