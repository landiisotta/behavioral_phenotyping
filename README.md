# Behavioral phenotyping project
Behavioral data embeddings for the stratification of individuals
with neurodevelopmenta conditions. 

Application to data of individuals with Autism Spectrum Condition (ASC)
from the laboratory of Observation, Diagnosis and Education (ODFLab).

## Requirements

```
python 3
```

## Pipeline TLDR

Run jupyter notebook

```
main.ipynb
```

### Modules

Connect to the database and dump data

```
dataset.py
```

Return vocabulary and dictionary of behavioral ehars for each of the 4 possible depth level. 
Return pandas dataframe with quantitative features (instrument scores) at level 4.

```
features.py
```

Perform TFIDF for patient embeddings 
Glove embeddings on words and average them out for patient embeddings

```
pt_embedding.py
```

Run hierarchical clustering

```
clustering.py
```

Visualize results (scatterplot + dendrogram for subcluster visualization,
heatmap for inspection of quantitative scores between subclusters)

```
visualization.py
```

