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


import csv_with_metadata
import file_index

import sys
import os

result = {True: 'passed', False: 'failed'}

def summary_test_results(module):

    # disable output during the test run, so it does not pollute the printed
    # info with the tests results
    old_stdout = sys.stdout
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        test_results = module.test()
    sys.stdout = old_stdout

    print('\n=== TEST RESUTLS - MODULE \"{}\" ==='.format(module.__name__))
    for k, v in test_results.items():
        print('-> ', k, ': ', result[v])
        
    return (module.__name__, all(test_results.values()))

if __name__ == '__main__':

    modules_to_test = [
                        csv_with_metadata, 
                       file_index
                       ]    

    module_result = [summary_test_results(m) for m in modules_to_test]

    print('\n=== MODULE SUMMARY ===')
    for name, r in module_result:
        print('->', name, ': ', result[r])
        
