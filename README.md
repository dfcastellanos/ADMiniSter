[![Python 3.8](https://github.com/kastellane/ADMiniSter/actions/workflows/CI.yml/badge.svg)](https://github.com/kastellane/ADMiniSter/actions/workflows/CI.yml)

[![Maintainability](https://api.codeclimate.com/v1/badges/6cdde69f7a09ee0665f9/maintainability)](https://codeclimate.com/github/kastellane/ADMiniSter/maintainability)

# ADMiniSter

ADMinister stems from "Ascii Data Minimalist management Suite." Its goal is to provide
a collection of simple yet powerful tools to manage plain text files storing numerical data.
Despite their drawbacks regarding reading speed or information density, plain text files remain 
the most widely available form of numerical data storage. The reason behind that is not only 
historical. Whenever performance or storage capabilities are not a concern, 
using plain text files for numerical data storage presents many advantages. Namely,
it is human-friendly, readable by most applications out of the box, and since the data is encoded 
into characters instead of in binary form, it is system-independent. For this latter 
reason, their compatibility with future applications or machines is also guaranteed. 
All these reasons make plain text data storage a very attractive solution whenever performance allows it. 



## Modules

### csv_with_metadata.py

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
The data is stored as CSV, and the metadata is 
stored as a well-structured header. The header can be composed of an arbitrary number 
of sections, and each section stores text or an arbitrary number of keys and values.
The tools provided here allow to write and load data and metadata stored in this
format easily. Specifically, the metadata in the header can be conveniently handled
using dictionary-like interfaces.

For further information and examples, see the documentation within the file.

### file_index.py

This module aims to provide tools to manage, locate, and process
large amounts of data files simply and efficiently. At the same time,
it seeks to work out of the box on most systems. This module achieves those goals 
by implementing a set of functions that leverage the capabilities of a so-called file index.
The file index is a table that relates the paths to many data files with some attributes 
characteristic of each file. The tools in this module help create file indices 
based on user-defined attributes loader functions, which define how attributes are to be extracted
 from the data files. Moreover, tools are provided for locating data files with queries based on 
attributes and to easily launch parallel analyses of the data files using user-defined
functions.

Since the file index contains paths to the data files but not the data itself, the file index is 
typically lightweight and fast, and can in many situations efficiently replace more
complex, heavier data management systems.


For further information and examples, see the documentation within the file.


## Installation

### Automatic installation
ADMiniSter requires `Python >= 3.4`. If your current Python environment satisfies this
condition, you can install all the dependencies and the ADMiniSter package doing

```sh
make install
```

Note: the installation requires `pip` and `setuptools`. They are likely to be
already installed. Nonetheless, if `pip` is not installed, you can follow 
[these](https://pip.pypa.io/en/stable/installation/) instructions to install it.
If `setuptools` is missing, you can install it doing `pip install setuptools` once
`pip` has been installed.

### Manual installation
Alternatively, if some problem arises during the automatic installation or you desire
more control, you can resort to a manual installation.

ADMiniSter depends on the packages `numpy`, `pandas`, `joblib` and`progressbar2`. 
You can install them doing

```sh
pip install numpy pandas joblib progressbar2
```

Once the dependencies are met, go to the ADMiniSter directory and do

```sh
python setup.py install
```

Alternatively, instead of installing it, you can add the files that you need to your project's
directory and directly import them as modules. The source files are located in 
`ADMiniSter_root/src/ADMiniSter/`. Also, you can add that
directory to your [Python path](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH)
to make all the modules available.


## Testing

For testing, the packages `pytest` and `pytest-cov` are required. If you used the
automatic installation, they should already be installed. Otherwise, you must
install them manually. Once ADMiniSter is installed in your working Python environment,
you can run the tests doing

```sh
make test
```

## Documentation
The ADMiniSter source files are fully documented and contain examples of use.

## Contact
ADMiniSter is developed and maintained by David Fernández Castellanos. You can report issues and bugs 
in the [project's repository](https://github.com/kastellane/ADMiniSter). You can contact the author 
through the methods provided on the [author's website] for longer discussions regarding, e.g., 
requests and ideas for the project.


## License
ADMiniSter is open source. You can freely use it, redistribute it, and/or modify it
under the terms of the Creative Commons Attribution 4.0 International Public 
License. The full text of the license can be found in the file LICENSE at the 
top level of the ADMiniSter distribution.
 
Copyright (C) 2021  - David Fernández Castellanos.


   [author's website]: <https://www.davidfcastellanos.com/>
   
