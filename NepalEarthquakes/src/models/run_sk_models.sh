#!/bin/bash

#Make sure to run with python enviroment activated
#to expand the search space and cross folds, the top of each script has globals
python extra_trees.py
python linear_sgd.py
python mlp_nn.py
python random_forest.py
python gradient_boosting.py
python voting.py
