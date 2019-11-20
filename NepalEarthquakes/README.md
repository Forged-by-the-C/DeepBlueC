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

'''
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .
'''

## Scripts

'''
python data_summary.py
'''
This is print to console the information (filename, shape, columns, and the first 5 rows) 
of each csv in the Data/raw directory.

'''
python data_split.py
'''
This is to split the provided data-set into train, dev, and test and place the result in the Data/processed folder.
