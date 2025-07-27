# functionsCF.py
# This file contains functions for generating training and validation sets for collaborative filtering tasks.
# It includes a function to generate a training set based on given row and column indices, and
# a utility function to compute the set difference of two 2D arrays.
# SPDX-FileCopyrightText: 2023 The OpenXLA Authors.
# SPDX-License-Identifier: Apache-2.0
#$Id: functionsCF.py 1 2023-10-01 12:00:00Z user $
#
import numpy as np
#
# This is a self defined function of comparing difference of 2D arrays
# It takes two 2D arrays as input and returns the elements in the first array that are not present in the second array.
# Parameters:
# - arr1: A 2D numpy array from which elements will be selected.
# - arr2: A 2D numpy array containing elements to be excluded from arr1
# Returns:
# - A 2D numpy array containing elements from arr1 that are not in arr2
# Raises:
# - None
#
def setdiff2d_list(arr1, arr2):
    delta = set(map(tuple, arr2))
    return np.array([x for x in arr1 if tuple(x) not in delta])
# This function generates a training set and a validation set based on the provided row and column indices.
# It samples k components from each row and column, ensuring that the training set has a specified
# ratio of sampled pairs to the total number of pairs. The function returns the training and validation
# indices as tuples of numpy arrays.
# Parameters:
# - row_index: A list or array of row indices.
# - column_index: A list or array of column indices.
# - ratio: A float representing the desired ratio of sampled pairs to the total number of pairs
# Returns:
# - train_indices: A tuple of numpy arrays containing the training row and column indices.
# - validation_indices: A tuple of numpy arrays containing the validation row and column indices.
# Raises:
# - ValueError: If the lengths of row_index and column_index do not match.  
def GenerateTrainingSet(row_index, column_index, ratio):
    # set random seed
    # When row index is not the same long as the column index, it is an error
    if len(row_index) != len(column_index):
        print("Error: row length doesn't match column length")
    # convert row and column index into the numpy array type
    row_index = np.array(row_index)
    column_index = np.array(column_index)
    # create possible row and column indices
    row_set = np.unique(row_index)
    column_set = np.unique(column_index)
    # create sampled row-column pairs
    row_col_pair = []
    # create rest pairs in the dataset
    rest_row_col_pair = [[row_index[i], column_index[i]] for i in range(len(row_index))]
    #rest_row_col_pair = []
    #for i in range(len(row_index)):
    #    rest_row_col_pair.append([row_index(i),column_index(i)])
    # First, we sample k components on each row and column
    k = 3
    for i in row_set:
        possible_column_index = column_index[row_index == i]
        col = np.random.choice(list(possible_column_index), min([k,len(possible_column_index)]), replace=False)
        for s in range(len(col)):
            row_col_pair.append([i, col[s]])
    for j in column_set:
        possible_row_index = row_index[column_index == j]
        row = np.random.choice(list(possible_row_index), min([k,len(possible_row_index)]), replace=False)
        for s in range(len(row)):
            row_col_pair.append([row[s], j])
    row_col_pair = np.unique(row_col_pair, axis=0)
    rest_row_col_pair = setdiff2d_list(rest_row_col_pair,row_col_pair)
    if (len(row_col_pair) / len(row_index)) < ratio and len(rest_row_col_pair) > 0:
        sampled_indices = np.random.choice(range(len(rest_row_col_pair)), int(ratio * len(row_index) - len(row_col_pair)),replace=False)
        #for i in range(len(sampled_indices)):
        row_col_pair = np.append(row_col_pair, rest_row_col_pair[sampled_indices,], axis=0)
        row_col_pair = np.unique(row_col_pair, axis=0)
        rest_row_col_pair = setdiff2d_list(rest_row_col_pair, row_col_pair)
    train_indices = np.array(row_col_pair[:, 0] - 1).astype(int), np.array(row_col_pair[:, 1] - 1).astype(int)
    validation_indices = np.array(rest_row_col_pair[:, 0] - 1).astype(int), np.array(rest_row_col_pair[:, 1] - 1).astype(
        int)
    return train_indices, validation_indices