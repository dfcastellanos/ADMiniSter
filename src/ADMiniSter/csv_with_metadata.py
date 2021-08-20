"""
    
    License
    -------
    Copyright (C) 2021  - David Fernández Castellanos
    
    This file is part of the ADMiniSter package. You can use it, redistribute
    it, and/or modify it under the terms of the Creative Commons Attribution
    4.0 International Public License.
    
    Summary
    -------    
    Long-term storage of numerical data requires context to make sense of that data. 
    Adding metadata to the files can partially solve this problem by making the files 
    self-descriptive. While common plain-text data formats such as JSON and XML can 
    handle metadata in a natural way, the CSV format, which is specially convenient 
    for numerical data storage, does not. Thus, different applications or users resort
    to their own ways to include metadata in CSV files as a header, making this 
    metadata format non-universal and potentially laborious to be parsed and loaded
    into an application.

    This module defines a format to store data and metadata in plain text files, and
    provides the tools to create and read the data and the metadata easily.
    The format specified here is meant to be self-descriptive and straightforward enough for
    most common situations. To this end, the data is stored as CSV, and the metadata is 
    stored as a header. The header can be composed of an arbitrary number of sections, 
    and each section stores text or an arbitrary number of keys and values.
    The tools provided here allow us to write and load data stored in this format easily.
    The CSV data is handled using pandas built-in tools. The metadata in the header
    is manipulated using dictionary-like interfaces, thanks to the functions implemented
    next.
        
    Format specification
    --------------------
    1 - The header is defined by all the lines from the top of the file that start with #
    
    2 - After # and a blank space, the next single word defines a header's section name
    
    3 - After the section name, a colon, and a blank space, the section's data is specified
    
    4 - If '=' is present in the section's data:
           4.1 - The data represents dictionary-like data, with the format:
              key1=value1, key2=value2, ... => dict(key1=value1, key2=value2, ...)
           4.2 - Numerical values are automatically identified and parsed. The keys always remain strings
        else: 
           4.3 - The section's data is a single string
           
    5. - A section named dtype specifies the types of the columns following the key-value synthax:
            col_name1=col_type1, col_name2=col_type2, ...
            
    6. - After the last header line, the CSV data begins
        6.1 - The first line specifies the name of the columns
        6.2 - The rest of the lines correspond to the data
    
    7. - If no header section is specified, the file reduces to a plain CSV file
    
    Example
    -------
    An example of such data file looks as follow:
    
        # name: format_example
        # description: with this meaningful description, we will understand the data in the future
        # params: mu=1., sigma=0.5, a=3.
        # dtype: time_step=int64, value=float64
        step_number,value
        0,3.72816
        1,3.76502
        2,4.09007
        3,3.41426
        4,4.36476
        5,3.14854
        6,4.38866
        7,4.09359
        8,3.89782
        9,3.66243
        10,4.22698
        11,4.90460
        12,3.37719
        13,4.28130
        ...
    
    To create such file, we would create the data and use the write function:
            
        >>> from ADMiniSter import csv_with_metadata as csv
        >>> import numpy as np
        >>> import pandas as pd
        >>> 
        >>> mu = 1.
        >>> sigma = 0.5
        >>> a = 3.
        >>> 
        >>> n = 100
        >>> t = np.arange(n)
        >>> x = np.random.normal(mu,sigma,n)+a
        >>> 
        >>> df = pd.DataFrame({'time_step': t, 'value': x})
        >>> 
        >>> metadata = dict(name = 'format_example',
        >>>                 description = 'with this meaningful description, we will understand the data in the future',
        >>>                 params = {'mu':mu, 'sigma':sigma, 'a':a}
        >>>                 )
        >>> 
        >>> csv.write(df, '/tmp/test_file.csv', metadata)
        
        
    To load it, we would use the load function:

        >>> df, metata = csv.load('/tmp/test_file.csv')
        >>> 
        >>> df
                     time_step    value
        0           0  3.72816
        1           1  3.76502
        2           2  4.09007
        3           3  3.41426
        4           4  4.36476
        ..        ...      ...
        95         95  4.36909
        96         96  3.78041
        97         97  3.71782
        98         98  3.61544
        99         99  4.37941
                
        [100 rows x 2 columns]
        >>> 
        >>> df.dtypes
        time_step      int64
        value        float64
        dtype: object
        >>>         
        >>> metadata
        {'name': 'format_example',
        'description': 'with this meaningful description, we will understand the data in the future',
        'params': {'mu': 1.0, 'sigma': 0.5, 'a': 3.0},
        'dtype': {'time_step': 'int64', 'value': 'float64'}}
            
"""

import pandas as pd
from ast import literal_eval


def header_line_to_dict(hearder_line):
    """

    Parse a line of a text file header into a dictionary.

    Parameters
    ----------
    - hearder_line: a line of the header

    Returns
    -------
    A dictionary witih the data stored in the header line.

    For more informaiton, see format description at the top.

    """

    hearder_line = hearder_line.split(",")
    hearder_line = filter(lambda x: x != "", hearder_line)

    attrs = {}
    for S in hearder_line:
        a = S.strip(" ").split("=")
        key = a[0]
        value_str = a[1]
        try:
            attrs[key] = literal_eval(value_str)
        except ValueError:
            # in this case, the string does not represent a number, so we
            # keep it as is
            attrs[key] = value_str

    return attrs


def construct_header(metadata):
    """

    Construct a text file header in string format.
    For more informaiton, see format description at the top.

    Parameters
    ----------
    - metadata: dictionary-like object witih the metadata

    Returns
    -------
    A string with the metadata, ready to be used as header of a text file.

    """

    header_str = ""

    for name, value in metadata.items():

        if name == "filename":
            continue

        header_str += "# {}:".format(name)

        if type(value) == dict:
            for (k, v) in value.items():
                k = k.replace(" ", "_")
                if type(v) == str:
                    v = v.replace(" ", "_")
            value_str = ["{}={}".format(k, v) for k, v in value.items()]
            value_str = ("," + " ").join(value_str)
            header_str += " {}\n".format(value_str)

        else:
            header_str += " {}\n".format(value)

    return header_str.strip("\n")


def write(df, filename, metadata=None, float_format="%.5f"):
    """

    Write the input pandas DataFrame into a CSV file, with a header created from
    the input metadata.

    Parameters
    ----------
    - df: the pandas DataFrame
    - filename: the file to be written
    - metadata: the metadata, to be written as a header. It is a dictionary, where each
                key corresponds to a section of the header. Each value is a string, a number,
                or another dictionary. If it is another dictionary, it will be written as a
                header line with format key1=value1, key2=value2, ...
                (for more informaiton, see format description at the top)
    - float_format: specifies the format with which the float data is written


    Example
    -------
    See documentation at the top

    """

    with open(filename, "w") as file:
        if metadata is not None:
            metadata["dtype"] = {k: str(v) for k, v in df.dtypes.items()}
            header_str = construct_header(metadata)
            file.write(header_str + "\n")

        df.to_csv(file, index=False, sep=",", float_format=float_format)

    return


def parse_header(filename):
    """

    Parse the header of a text file.

    Parameters
    ----------
    - filename: the file to read

    Returns
    -------
    A dictionary, where each key corresponds to a line of the header. Each value is
    another dictionary with the data of that line.

    For more informaiton, see format description at the top.

    """

    header = dict()

    with open(filename, "r") as file:

        file_str = file.read().split("\n")
        hearder_lines = list()

        for line in file_str:
            if line.startswith("#"):
                hearder_lines.append(line.strip("# "))

        for line in hearder_lines:
            name = line.split(":")[0].strip(" ")
            value = line.split(":")[1].strip(" ")
            if "=" in value:
                value = header_line_to_dict(value)
            header[name] = value

    return header


def load(filename):
    """

    Load a text file with metadata as header and data in CSV format.
    This function allows to load files created with write().
    For more informaiton, see format description at the top.

    Parameters
    ----------
    - filename: the file to load

    Returns
    -------
    - df: a pandas DataFrame with the CSV data
    - metadata: a dictionary whose keys correspond to each line of metadata
                composing the header.

    Example
    -------
    See documentation at the top

    """

    metadata = parse_header(filename)
    metadata["filename"] = filename

    if "dtype" in metadata:
        dtype = metadata["dtype"]
    else:
        dtype = None

    try:
        df = pd.read_csv(filename, comment="#", engine="c", dtype=dtype)
    except FileNotFoundError:
        print("Problem reading: {}".format(filename))
        raise

    return df, metadata
