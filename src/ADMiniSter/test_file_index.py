#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    License
    -------
    Copyright (C) 2021  - David Fernández Castellanos
    
    This file is part of the ADMiniSter package. You can use it, redistribute
    it, and/or modify it under the terms of the Creative Commons Attribution
    4.0 International Public License.
    
"""


import numpy as np
import pandas as pd
import pytest
import os

from ADMiniSter import csv_with_metadata as csv
from ADMiniSter import file_index

# pylint: disable=W0612, E1101, E1136, W0621


def make_files(A_list, sigma_list):

    os.chdir("/tmp")

    N = 365
    t = np.arange(N)
    filenames = list()
    A_ = list()
    sigma_ = list()

    for A in A_list:
        for sigma in sigma_list:
            A = np.round(A, 2)
            sigma = np.round(sigma, 2)

            x = A * np.sin(t * 2 * 3.1415 / N) + np.random.normal(0.0, sigma, N)

            df = pd.DataFrame({"time_step": t, "value": x})

            filename = "test_file_index+A_{}+sigma_{}.csv".format(A, sigma)

            metadata = dict(
                filename=filename,
                description="this is the description",
                params={"A": A, "sigma": sigma},
            )

            csv.write(df, filename, metadata)

            filenames.append(filename)
            A_.append(A)
            sigma_.append(sigma)

    return filenames, A_, sigma_


@pytest.fixture
def get_data():

    filenames, A_list, sigma_list = make_files([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4])

    os.chdir("/tmp")

    metadata_loader = lambda filename: csv.parse_header(filename)["params"]
    original_df = file_index.build(filenames, metadata_loader)

    # create the file index, write it to a file and load it back
    file_index.write(original_df, "index.csv")
    df = file_index.load("index.csv")

    return original_df, df, filenames, A_list, sigma_list


@pytest.fixture
def get_modified_data(get_data):

    original_df, df, filenames, A_list, sigma_list = get_data

    dropped_filenames = [f for f in filenames if "A_5" in f]
    for f in dropped_filenames:
        os.system("rm -f " + f)

    updated_filenames = set(filenames) - set(dropped_filenames)
    new_filenames, _, _ = make_files([6, 7, 8], [0.1, 0.2, 0.3])
    updated_filenames.update(new_filenames)

    metadata_loader = lambda filename: csv.parse_header(filename)["params"]
    updated_df = file_index.update(df, updated_filenames, metadata_loader)

    return updated_df, updated_filenames


def test_index_contains_all_filenames(get_data):

    original_df, df, filenames, A_list, sigma_list = get_data

    assert len(df.index) == len(filenames)


def test_rows_contain_file_attributes(get_data):

    original_df, df, filenames, A_list, sigma_list = get_data

    r = list()
    for i, row in df.iterrows():
        r.append(
            row["filename"] == filenames[i]
            and row["A"] == A_list[i]
            and row["sigma"] == sigma_list[i]
        )
    assert all(r)


def test_reloaded_file_index_equals_original(get_data):

    original_df, df, filenames, A_list, sigma_list = get_data

    assert all(df == original_df)


def test_reloaded_dtypes_equal_originals(get_data):

    original_df, df, filenames, A_list, sigma_list = get_data

    assert all(df.dtypes == original_df.dtypes)


def test_reloaded_columns_equal_originals(get_data):

    original_df, df, filenames, A_list, sigma_list = get_data

    assert all(df.columns == original_df.columns)


def test_updated_index_equals_updated_filenames(get_modified_data):

    updated_df, updated_filenames = get_modified_data

    assert set(updated_df.filename.values) == updated_filenames


def test_locate_attributes(get_data):

    original_df, df, filenames, A_list, sigma_list = get_data

    subset_df = file_index.locate(original_df, dict(A=1.0))
    subset_df_ = original_df[original_df["A"] == 1.0]

    assert all(subset_df == subset_df_)


@pytest.mark.skip
def test_parallel_apply(get_data):

    # Not automated yet

    original_df, df, filenames, A_list, sigma_list = get_data

    n_cpus = 2

    # process row-wise
    target_func = lambda row: print(row["filename"])
    file_index.apply(df, target_func, n_cpus)

    # process row-wise (only a subset)
    sub_df = file_index.locate(df, dict(A=7.0))
    results = file_index.apply(sub_df, target_func, n_cpus)

    # group and process group-wise
    target_func = lambda key, group: print(group)
    results = file_index.group_and_apply(df, target_func, df["A"], n_cpus)

    print(results)
