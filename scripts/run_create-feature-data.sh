#!/usr/bin/env bash
  
clear
projdir=../
datadir=$projdir/data/

/Users/ilandi/anaconda3/bin/python -u $projdir/src/create-feature-dataset.py $datadir
