#!/usr/bin/env bash

clear
projdir=..
datadir=$projdir/data
indir=$datadir/odf-data-

level=$1

/Users/ilandi/anaconda3/bin/python -u $projdir/src/create-vocabulary.py $indir $level
