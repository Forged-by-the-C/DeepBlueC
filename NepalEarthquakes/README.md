# Data Driven Nepal Earthquakes Challenge

##Organization

```
|- Data
|   |- raw                  -> all the csv's provided by the competition
|   |- split                -> the split csv's to train(80%), cross_val(10%), and test(10%)
|
|- Vizualization
|   |- local_site           -> output from Great Expectations Data Docs
|
|- src                      -> Source code for use in this project.
|   |- __init__.py          -> Makes src a Python module
|   |
|   |- data                 -> Scripts to download or generate data
|   │   |-data_summary.py
|   |   |-data_split.py
|   |
|   |- features             -> Scripts to turn raw data into features for modeling
|   |
|   |- models               -> Scripts to train models and then use trained models to make
|   |   |                        predictions
|   |   |- random_forest.py
|   |
|   |- visualization        ->  Scripts to create exploratory and results oriented visualizations
|   |
|   |- utils		    ->  General purpose scripts
|   |   |- performance_score.py
|   |   |- grab_data.py
|
|- requirements.txt
|- setup.py
```


##Setup

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Scripts

```
python data_summary.py
```
This is print to console the information (filename, shape, columns, and the first 5 rows) 
of each csv in the Data/raw directory.

```
python data_split.py
```
This is to split the provided data-set into train, dev, and test and place the result in the Data/interim folder.

## Review

We ended up sprinting with code past our upkeep of documentation, as so often happens in software development. 

Many iterations of feature engineering happened with Zac taking the charge. He created subject specific engineering (such as with a `slenderness_ratio`) and data scince techniques such as using a light extra trees classsifier to determine feature importance (which led to pruning) and categorical encoders.

On the modelling side we leaned most heavily on `sklearn` but also experimented with libraries such as `xgboost` (optimized for gradient boosting) `skopt` (a now not-maintained optimizer that sits/sat on top of sklearn, with a bit of tweaking was able to use if for Bayesian optimization for hyperparameter search on extra trees, gradien boosting, and random forests) and `autosk`, a automl package that sits on docker using baryesina optimization for hyperparameter search and some tranfer learning from backed in weights by the developers. We also developed a module for a parallelized genetic algorithm to architecture search a feed forward neural network on larger AWS machines.  Note, there is a script in here using pytorch but, for this specific purpose, we used the `sklearn.MLPClassifier` method. 

### Results

As of December 22 (competition ends on December 31 but everyone split for the Christmas holidays, and we have been working for about 3 weeks (with a break for Thanksgiving) on this competition and it is time to move on) our best model was an extra trees classifier with 320 estimators, pruning handled with  `sqrt` regularizer, and a minimum of 3 samples per leaf.  This led to the following scores

```
training_score : 0.844004221028396
val_score : 0.745088257866462
submission score: 0.7427
```

This put us at 114 (at the time, last check 131) or 1552 competitors, a finish in the top 10%, which is a great first strike we are real proud of.

### Lessons Learned

- Most of the biggest improvements came from feature engineering, in this competition that was the most important piece
- 2 good weeks is enough for a competition. As we automate tools, this will be even more managable
- Having a solid model wrapper class was huge, allowing for quick iteration on different models and architectures
- Write code and maintain git in such a way that all training tasks can be done on an aws machine that pulls down the repo. This leaves the local machines open for development and enforces discipline on the coding/repo organization