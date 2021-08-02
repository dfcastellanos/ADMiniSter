"""
    
    License
    -------
    Copyright (C) 2021  - David Fernández Castellanos
    
    This file is part of the ADMiniSter package. You can use it, redistribute
    it, and/or modify it under the terms of the Creative Commons Attribution
    4.0 International Public License.
    
    Summary
    -------
    This module aims to provide tools to manage, locate, and process
    large amounts of data files simply and efficiently. At the same time,
    it seeks to work out of the box on most systems, for which it is top of standard
    Python modules such as pandas and NumPy.
    
    Specifically, this module achieves those goals by implementing a set of 
    functions that leverage the capabilities of a so-called file index. The file 
    index is a table that relates the file paths with some attributes characteristic of
    each file. Each row contains a filename column, which is the table's primary key,
    and defines the location of the data file within the file system. The rest of
    the columns correspond to the data file attributes, which depend on the context 
    and are for the user to define (see next section). Since the file index contains
    paths to the data files but not the data itself, the file index is typically lightweight and fast.
    Thus, the file index is saved and re-loaded as a plain-text CSV file, and a
    pandas DataFrame is used to manage it.
    
    When a file index has been created, we can leverage it. Thus, with the tools
    provided next, we can efficiently locate data files with queries based on their attributes using
    the locate function. We can call the apply or
    the group_and_apply functions to launch parallel analyses on the located data files.
    
    File attributes
    ---------------   
    Too relate each file with its attributes, the user must define an attributes loader 
    function that, given a file path, it returns its attributes. The attributes
    loader is passed to the file index build (or update) function, which does
    the rest of the work. 
    
    A typical example of file attributes is, e.g., metadata contained within the 
    files, such as the model parameters used for generating the data. In this case, 
    an attributes loader function would load the data file, read the metadata, and 
    return it as a dictionary. Another typical scenario is when the file attributes
    correspond to the results of some analyses. In this case, the attributes loader
    would load the file, analyse the data, and return a dictionary with the names
    of each analysis and their results.
    
    For the specific case of the data format defined in the companion module 'csv_with_metada', 
    the file attributes might be e.g. extracted from the header. Let's consider a header with a 
    section named 'params'. In this case, a suitable attributes loader function would be:
            
        >>> attributes_loader = lambda filename : csv_with_metada.parse_header(filename)['params']
    
    which returns the header's 'params' section as a dictionary for each input filename.
    In this way, we would create a file index, relating each file path with the parameters
    stored in their headers.
        
    Example
    -------
    
    Let's consider a set of plain text files with names following the pattern 'A_*+B_*+C_*.dat', 
    where the * is the wildcard for the values of the parameters 'A', 'B', and 'C'. In this case,
    we are not interested in the contents of such files, just in how to index them.
    
    We can leverage the fact the file names follow a regular pattern to extract the values
    of 'A', 'B', and 'C' and use them to build the index. (this approach is desireable 
    since parsing the file names proves to be way faster than opening files for reading some sort
    of metadata, such as header information).
    
        >>> from ADMiniSter import file_index
        >>> import glob
        >>> 
        >>> files = glob.glob('A*+B*+C*.dat')
        >>> files
        ['A_3+B_1.7+C_-2.dat',
          'A_4+B_1.1+C_-7.dat',
          'A_1+B_1.7+C_-5.dat',
          'A_1+B_1.7+C_-2.dat',
          'A_2+B_1.1+C_-5.dat',
          'A_1+B_1.1+C_-7.dat',
        ...

    Let's define an attributes loader function what parses the filenames as '/tmp/A_4+B_1.1.dat' -> {'A': 4.0, 'B': 1.1}.
    This function does the job:
        
        >>> attrs_loader = lambda filename: {k:float(v) for k,v in [s.split('_') for s in filename.strip('.dat').split('+')]}

    Now we build the index
    
        >>> df = file_index.build(files, attrs_loader)
        >>> df
              A    B    C            filename
        0   3.0  1.7 -2.0  A_3+B_1.7+C_-2.dat
        1   4.0  1.1 -7.0  A_4+B_1.1+C_-7.dat
        2   1.0  1.7 -5.0  A_1+B_1.7+C_-5.dat
        3   1.0  1.7 -2.0  A_1+B_1.7+C_-2.dat
        4   2.0  1.1 -5.0  A_2+B_1.1+C_-5.dat
        5   1.0  1.1 -7.0  A_1+B_1.1+C_-7.dat
        ...
    
    And we can write it to a file
    
        >>> file_index.write(df, 'index.csv')
    
    From now on, we don't need to build the index anymore (maybe update it, see update function). All
    we need to do is to load it every time we need it. For that, we do
    
        >>>  df = file_index.load('index.csv')
    
    Within the index, we can locate files matching some desired attributes values. For
    example, let's look for all those files with A=2. and B=1.5
    
        >>> sub_df = file_index.locate(df, {'A':2, 'B':1.5})
        >>> sub_df
              A    B    C            filename
        17  2.0  1.5 -7.0  A_2+B_1.5+C_-7.dat
        24  2.0  1.5 -5.0  A_2+B_1.5+C_-5.dat
        39  2.0  1.5 -2.0  A_2+B_1.5+C_-2.dat

    If we were interested in analyzing those files, we could use the pandas' apply
    function over sub_df. However, let's illustrate a more demanding situation, where
    we want to analyze all the existing data files, which are potentially very numerous
    and heavy. In that case, a parallel analysis is desirable. The functions apply
    and group_and_apply defined in this module do that. Let's use the apply function.
    The first thing to do is define the target function that we want to apply
    in parallel to each different data file. In this case, for the sake of the 
    example, we don't define any complex function. Instead, we set the results
    of some hypothetical analyses using random numbers.    

        >>> def target_func(row):
        >>>     filenames = row['filename']
        >>>     # here, the file would be loaded and its data analysised
        >>>     row['analysis_1'] =  np.random.uniform()
        >>>     row['analysis_2'] =  np.random.uniform()
        >>>     row['analysis_3'] =  np.random.uniform()
        >>>     return row
    
    Now, let's run the analysis,
    
        >>> results = file_index.apply(df, target_func)
    
    The call returns a list of single-rows DataFrames. We can create a new results
    DataFrame with them as
    
        >>> results_df = pd.DataFrame(results)
        >>> results_df
              A    B    C            filename  analysis_1  analysis_2  analysis_3
        0   3.0  1.7 -2.0  A_3+B_1.7+C_-2.dat    0.193416    0.448960    0.982408
        1   4.0  1.1 -7.0  A_4+B_1.1+C_-7.dat    0.702925    0.956540    0.825651
        2   1.0  1.7 -5.0  A_1+B_1.7+C_-5.dat    0.235057    0.823497    0.334244
        3   1.0  1.7 -2.0  A_1+B_1.7+C_-2.dat    0.345587    0.632414    0.788807
        4   2.0  1.1 -5.0  A_2+B_1.1+C_-5.dat    0.408646    0.144957    0.179882
        5   1.0  1.1 -7.0  A_1+B_1.1+C_-7.dat    0.734338    0.655969    0.596402
        ...

    The function group_and_apply works similarly. However, it creates groups
    of data files and passes those groups to the target function to analyze several
    data files simultaneously. This is useful if, e.g., the data 
    files of each group correspond to different realizations of the same random process.
    
"""

# implementation of the core
import pandas as pd
import numpy as np

# for parallel apply and group_and_apply
import multiprocessing
from joblib import Parallel, delayed
from progressbar import progressbar

# for testing purposes
import os
import ADMiniSter.csv_with_metadata as csv


def load(index_filename):
    """
    
    Load an existing file index from a CSV file.
    
    Parameters
    ----------
    - index_filename: the path to the file.

    Returns
    -------
    A pandas DataFrame with the file index.
            
    """

    return pd.read_csv(index_filename, engine='c')


def write(df, index_filename):
    """
    
    Write a file index contained in a pandas DataFrame to a text CSV file.
    
    Parameters
    ----------
    - df: the DataFrame with the file index.
    - index_filename: the path to the file.
            
    """

    df.to_csv(index_filename, index=False)
    print('-> {} successfully written'.format(index_filename))
    return


def build(files, attrs_loader):
    """
    
    Build a new file index.
    
    Parameters
    ----------
    - files: the list of files to be indexed.
    - attrs_loader: a user-defined function that returns the attributes of a data file. The
                    attributes loader function must take as input the path to a datafile
                    and return a dictionary relating the names of the attributes (as keys)
                    to their respective values.                    

    Returns
    -------
    The file index as a pandas DataFrame.
    
    """

    print('-> Reading {} files...'.format(len(files)))
    if len(files) == 0: print('--> Nothing to do')

    new_data = list()
    problems = list()
    for filename in progressbar(files):
        try:
            attrs = attrs_loader(filename)
            attrs['filename'] = filename
        except Exception as e:
            print('--> Exception with ' + filename + ': \n', e)
            problems.append(filename)
            continue

        new_data.append(attrs)

    print('--> Problems reading {} files'.format(len(problems)))

    df = pd.DataFrame(new_data)

    # df.drop_duplicates(subset='filename',keep='first',inplace=True)

    return df


def update(df, files, attrs_loader):
    """
    
    updates an existing file index. If some files defined in the index are
    missing in the input file list, those missing files are removed from the
    index. If some input files are new, they are added to the index.
    
    Parameters
    ----------
    - df: a DataFrame with the existing file index.
    - files: the updated list of files.
    - attrs_loader: a user-defined function that returns the attributes of a data file. The
                attributes loader function must take as input the path to a data file
                and return a dictionary relating the names of the attributes (as keys)
                to their respective values.  

    Returns
    -------
    The updated file index as a pandas DataFrame.
            
    """

    print('-> Updating file index')

    # remove missing files    
    n0 = len(df)
    w = df['filename'].apply(os.path.isfile)  # still exisiting files
    n1 = sum(w)
    if n0 != n1:
        print('--> Missing files:')
        print(df['filename'][np.logical_not(w)])
    df = df[w]
    print('--> {} files have been cleaned from the index'.format(n0 - n1))

    # add new files
    new_files = set(files) - set(df['filename'].values)
    df_new = build(new_files, attrs_loader)
    df = pd.concat([df, df_new])

    print('--> {} new files have been added to the index'.format(len(new_files)))

    return df


def locate(df, attrs, drop=False):
    """
    
    Locate all the files with the desired attribute values in the given file index.
    
    Parameters
    ----------
    - df: a DataFrame with the file index.
    - attrs: a dictionary relating the names of the attributes (as keys) to their respective values. 
             This dictionary of attributes does not need to contain all the possible attributes,
             but a subset of them. Thus, if a file index has attributes A, B and C, it is allowed
             to query for attrs = dict(A=5.0, B=3.), which will locate all the data files that simultaneously fulfill
             that A=5. and B=3., irrespective of the value of C.
    - drop: if drop, the returned DataFrame won't contain the columns associated with the 
            input attributes (in the example above, it won't have columns A and B, but it
                              will have C).

    Returns
    -------
    A subset of the file index containing the paths to all those data files whose attributes
    match the desired values. Note: this subset is another file index to which 
    all the functions of this module are applicable.
            
    """

    # since we will perform modifications during the process, we work on a copy
    # (what is normally big is the data to which the index points, but not the index 
    # itself, so this should be ok)
    df = df.copy()

    # if the values are float64, Pandas has problems to find values in the 
    # dataframe using == operator
    for c in df.columns:
        if df[c].dtype == 'float64':
            df[c] = df[c].astype('float32')

    # list of boolean lists           
    locs = [df[k] == v for k, v in attrs.items()]

    # stack into matrix, then keep only columns where all values are true
    w = np.bitwise_and.reduce(np.vstack(locs))
    df = df[w]
    if drop:
        df.drop(list(attrs.keys()), axis=1, inplace=True)

    # if the datafile has only one row, pandas will return a series from .loc[].
    # We want to ensure the that we get always the same behavior
    if isinstance(df, pd.DataFrame):
        return df
    elif isinstance(df, pd.Series):
        out.name = None
        return out.to_frame().T
    else:
        raise TypeError


def apply(df, target_func, n_cpus=None):
    """

    Apply row-wise a user-defined target function to the file index. Each row 
    is passed as input to the function. The call for each row is done in parallel.
        
    Parameters
    ----------
    - df: a DataFrame with the file index.
    - target_func: a user-defined function to be applied, which takes as input a 
                   row from the file index. A typical target function loads 
                   the file and analyses its data. Then, it adds the results
                   as new columns of the input row.
    - n_cpus: the number of CPUs for the parallel analysis. If none, the default
             number of CPUs defined by the system will be used.

    Returns
    -------
    A list with the results. Each entry of the list contains whatever the
    target function returned when called on each file index row. If n_cpus > 1,
    the function is applied in parallel, in which case the order of the list is
    in general different from the file index.
            
    """

    return Parallel(n_cpus)(delayed(target_func)(row) for i, row in progressbar(df.iterrows()))


def group_and_apply(df, target_func, by, n_cpus=None):
    """

    Group the file index and then apply group-wise a user-defined target function. 
    The call for each group is done in parallel.
        
    Parameters
    ----------
    - df: a DataFrame with the file index.
    - target_func: a user-defined function to be applied, which takes two arguments. 
                  The first is a tuple with the grouping values. The second is the 
                  group's DataFrame. A typical target function loads the data files
                  of the group, analyses their data, and returns the
                  results from the analysis and the grouping values.
    - by: used to determine the groups. It allows us to define groups 
          of data files that share some characteristic and whose data must be 
          processed simultaneously by the target function. See pandas.DataFrame.groupby.           
    - n_cpus: the number of CPUs for the parallel analysis. If none, the default
             number of CPUs defined by the system will be used.

    Returns
    -------
    A list with the results. Each entry of the list contains whatever the
    target function returned when called on each group.
            
    """

    grouped_df = df.groupby(by)
    return Parallel(n_cpus)(delayed(target_func)(key, group) for key, group in progressbar(grouped_df))


def make_test_files(A_list, sigma_list):
    N = 365
    t = np.arange(N)
    filenames = list()
    A_ = list()
    sigma_ = list()

    for A in A_list:
        for sigma in sigma_list:
            A = np.round(A, 2)
            sigma = np.round(sigma, 2)

            x = A * np.sin(t * 2 * 3.1415 / N) + np.random.normal(0., sigma, N)

            df = pd.DataFrame({'time_step': t, 'value': x})

            filename = 'test_file_index+A_{}+sigma_{}.csv'.format(A, sigma)

            metadata = dict(filename=filename,
                            description='this is the description',
                            params={'A': A, 'sigma': sigma}
                            )

            csv.write(df, filename, metadata)

            filenames.append(filename)
            A_.append(A)
            sigma_.append(sigma)

    return filenames, A_, sigma_


def test():
    os.chdir('/tmp')

    # create files to test the file index
    filenames, A_list, sigma_list = make_test_files([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4])

    metadata_loader = lambda filename: csv.parse_header(filename)['params']
    original_df = build(filenames, metadata_loader)

    # create the file index, write it to a file and load it back
    write(original_df, 'index.csv')
    df = load('index.csv')

    test_results = {}
    test_results['len(reloaded_df.index)==len(filenames)'] = len(df.index) == len(filenames)

    r = list()
    for i, row in df.iterrows():
        r.append(row['filename'] == filenames[i] and \
                 row['A'] == A_list[i] and \
                 row['sigma'] == sigma_list[i])
    test_results['all(row of reloaded_df == (filename,A,sigma))'] = all(r)

    test_results['reloaded_df == original_df'] = all(df == original_df)
    test_results['reloaded_df.dtypes == original_df.dtypes'] = all(df.dtypes == original_df.dtypes)
    test_results['reloaded_df.columns == original_df.columns'] = all(df.columns == original_df.columns)

    dropped_filenames = [f for f in filenames if 'A_5' in f]
    for f in dropped_filenames:
        os.system('rm -f ' + f)

    updated_filenames = set(filenames) - set(dropped_filenames)
    new_filenames, _, _ = make_test_files([6, 7, 8], [0.1, 0.2, 0.3])
    updated_filenames.update(new_filenames)
    updated_df = update(df, updated_filenames, metadata_loader)

    test_results['len(updated_df)==len(updated_filenames)'] = len(updated_df.index) == len(updated_filenames)
    test_results['updated_df.filename==np.array(updated_filenames)'] = set(
        updated_df.filename.values) == updated_filenames

    subset_df = locate(df, dict(A=1.))
    subset_df_ = df[df['A'] == 1.]
    test_results['locate(df,attrs)==df[ where(attrs) ]'] = all(subset_df == subset_df_)

    # n_cpus = 1#multiprocessing.cpu_count()

    # # process row-wise
    # target_func = lambda row: print(row['filename'])
    # apply(df, target_func, n_cpus)

    # # process row-wise (only a subset)
    # sub_df = locate(df, dict(A=7.))
    # results = apply(sub_df, target_func, n_cpus)

    # # group and process group-wise
    # target_func = lambda key, group: print(group)
    # results = group_and_apply(df, target_func, df['A'], n_cpus)

    # print(results)

    return test_results
