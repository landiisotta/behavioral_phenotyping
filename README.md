# Behavioral phenotyping project

Behavioral data embeddings for the stratification of individuals
with neurodevelopmental conditions.

Designed for observational measurements of cognition and behavior of individuals with 
Autism Spectrum Conditions (ASCs).

#### TODO: Abstract 

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

* `dataset.py`: Connects to the database and dump data
* `features.py`: Returns vocabulary and dictionary of behavioral *EHRs* for each of the 4 possible depth levels. 
It also returns a dataset with quantitative scores for level 4 features
* `pt_embedding.py`: Performs TFIDF for patient embeddings; Glove embeddings on words and average them out for 
subject embeddings; Word2vec embeddings on words, that are then averaged to output individual representations
* `clustering.py`: Performs Hierarchical Clustering/k-means on embeddings, and quantitative 4th level features
* `visualization.py`: Visualizes results (e.g. _scatterplot & dendrogram_)for sub-cluster visualization; 
_Heatmap_ for inspection of quantitative scores between sub-clusters
* `basic_statistics.py`: Returns basic demographic statistics for dataset description
* `test-demog-cl.R`: Runs multiple pairwise comparisons between subgroups 
to check for confounders and support clinical validation


#### TODO: Paper, Poster, Conference Reference

#### TODO: Credits and Acknowledgements


