# Neural Collaborative Filtering (NCF) Model
An Implementation of Neural Collaborative Filtering Models for Recommender Systems with emphasis on Implicit Data.

Dataset: Movielens (100k: default, 1M can also be used)


## Packages Required

- python3 - 3.6
- tensorflow - 1.2
    - 'pip install --upgrade tensorflow' (CPU version)
    - 'pip install --upgrade tensorflow-gpu' (if your system has a NVIDIA® GPU )
- keras - 2.1
- theano - 1.0

## Program Usage
Some example to depict the usage of the program.


1. python Models.py
    - default argument values will be used
        a. dataName: ml-100k (default)
        b. batchSize: 256 (training instamces to be included during each forwards and backward pass)
        c. num_epochs: 10 (number of training iterations)

2. Example to add the command line arguments:
    - ```python Model.py -dataName ml-1m```
        - for loading dataset with 1M ratings.
    - ```python Model.py -batchSize 128```
        - specifying the batch size for training purpose.
    - ```python Model.py -num_epochs 50```