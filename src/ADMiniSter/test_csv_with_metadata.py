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

from ADMiniSter import csv_with_metadata as csv

# pylint: disable=W0612, E1101, E1136


@pytest.fixture
def create_csv_data():

    # create sample data
    A = 1.0
    sigma = 0.5

    N = 365
    t = np.arange(N)
    x = A * np.sin(t * 2 * 3.1415 / N) + np.random.normal(0.0, sigma, N)

    df_input = pd.DataFrame({"time_step": t, "value": x})

    metadata_input = dict(
        description="this is the description",
        name="this is the name",
        params={"A": A, "sigma": sigma},
    )

    # write the data to a file and load it back
    csv.write(df_input, "/tmp/test.csv", metadata=metadata_input)
    df, metadata = csv.load("/tmp/test.csv")

    return df_input, metadata_input, df, metadata


def test_loaded_data_equals_original(create_csv_data):
    # pylint: disable=W0621,W0612
    df_input, metadata_input, df, metadata = create_csv_data
    assert all(df == df_input)


def test_loaded_types_equal_originals(create_csv_data):
    # pylint: disable=W0621,W0612
    df_input, metadata_input, df, metadata = create_csv_data
    assert all(df.dtypes == df_input.dtypes)


def test_loaded_metadata_equals_original(create_csv_data):
    # pylint: disable=W0621,W0612
    df_input, metadata_input, df, metadata = create_csv_data
    assert all([metadata[k] == metadata_input[k] for k in metadata_input.keys()])
