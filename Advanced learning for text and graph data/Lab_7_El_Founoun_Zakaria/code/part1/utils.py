"""
Learning on Sets / Learning with Proteins - ALTEGRAD - Dec 2022
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
    # your code here #
    ##################

    X_train = np.zeros((n_train, max_train_card))
    y_train = np.zeros(n_train)

    for i in range(n_train):

        cardinal = np.random.randint(1, max_train_card + 1)
        X_train[i, -cardinal:] = np.random.randint(1, max_train_card + 1, size=cardinal)  # to respect padding
        y_train[i] = np.sum(X_train[i])

    return X_train, y_train

create_train_dataset()

def create_test_dataset():
    
    ############## Task 2
    
    ##################
    # your code here #
    ##################
    n_test = 200000

    cardinalities = range(5, 101, 5)
    n_samples_per_card = 10000
    X_test = []
    y_test = []

    for card in cardinalities:
        X = np.random.randint(1, 11, size=(n_samples_per_card, card))
        y = np.sum(X, axis=1)

        X_test.append(X)
        y_test.append(y)

    return X_test, y_test