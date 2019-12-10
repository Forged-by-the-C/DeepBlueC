To setup skopt, had to 
1. clone the repo, 
2. pip install -e .
3.then manually copy the last pull requests' single comitted file (searchcv.py) into the repo

Voting classifier doesnt work straight up, would have to pick the model parameters found by each optimization, train a single one in normal sklearn, then use those for the voting classifier (the pkl of the skopt models are buggy).
