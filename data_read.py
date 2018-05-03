from __future__ import print_function
import itertools
import os
import zipfile
import numpy as np
import requests
import scipy.sparse as sp


def _get_movielens_path():
    """
    Get path to the movielens dataset file.
    """

    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'Data.zip')

def _get_raw_movielens_data(fileName):
    """
    Return the raw lines of the train and test files.
    """

    path = _get_movielens_path()

    print("="*20, '\n\nLoading {} Movielens Dataset.....\n\n'.format(fileName), "="*20)
    
    # read the train and test data files.
    with zipfile.ZipFile(path) as datafile:
        if fileName == 'ml-100k':
            return (datafile.read('Data/ml-100k.train.rating').decode().split('\n'),
                    datafile.read('Data/ml-100k.test.rating').decode().split('\n'))
        else:
            return (datafile.read('Data/ml-1m.train.rating').decode().split('\n'),
                    datafile.read('Data/ml-1m.test.rating').decode().split('\n'))

def _parse(data):
    """
    Parse movielens dataset lines.
    Extract all the data from each tuple (userid, movieid, rating, timestamp)
    """

    for line in data:

        if not line:
            continue

        uid, iid, rating, timestamp = [int(x) for x in line.split('\t')]

        yield uid, iid, rating, timestamp


def _build_interaction_matrix(rows, cols, data):

    # Initially, create a sparse matrix
    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    # generate implicit data
    for uid, iid, rating, timestamp in data:
        # Let's assume only really good things (ratings equal to or above 4.0) are positives.
        if rating >= 1.0:
            mat[uid, iid] = 1.0
    
    # Convert this matrix to COOrdinate format
    # advantage: Once a matrix has been constructed, convert to CSR or CSC format
    # for fast arithmetic and matrix vector operations
    return mat.tocoo()

def get_triplets(mat):

    return mat.row, mat.col, np.random.randint(mat.shape[1], size=len(mat.row))


def get_movielens_data(fileName):
    """
    Return (train_interactions, test_interactions).
    """

    train_data, test_data = _get_raw_movielens_data(fileName)

    uids = set()
    iids = set()

    for uid, iid, rating, timestamp in itertools.chain(_parse(train_data),
                                                       _parse(test_data)):
        uids.add(uid)
        iids.add(iid)

    rows = max(uids) + 1
    cols = max(iids) + 1

    return (_build_interaction_matrix(rows, cols, _parse(train_data)),
            _build_interaction_matrix(rows, cols, _parse(test_data)))