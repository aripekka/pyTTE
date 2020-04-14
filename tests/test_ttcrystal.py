# -*- coding: utf-8 -*-

'''
Tests for the TTcrystal class
'''

import sys
import os.path

sys.path.insert(1, os.path.join(os.path.dirname(__file__),'..'))

from pyTTE import TTcrystal, Quantity

def test_init_required_inputs():
    #Tries to initialize with correct and incorrect mandatory inputs

    #Test that all the inputs are required    
    try:
        TTcrystal()
        assert False
    except KeyError as e:
        assert True

    try:
        TTcrystal(crystal='Si')
        assert False
    except KeyError as e:
        assert True

    try:
        TTcrystal(crystal='Si',hkl=[1,1,1])
        assert False
    except KeyError as e:
        assert True

    #Correct input
    try:
        TTcrystal(crystal='Si', hkl=[1,1,1], thickness=Quantity(100,'um'))
        assert True
    except KeyError as e:
        assert False

    #Check that the input types are checked
    try:
        TTcrystal(crystal=23,hkl=[1,1,1], thickness=Quantity(100,'um'))
        assert False
    except ValueError as e:
        assert True    

    #Check that the input types are checked
    try:
        TTcrystal(crystal='Si',hkl='abc', thickness=Quantity(100,'um'))
        assert False
    except ValueError as e:
        assert True    

    #Check that the input types are checked
    try:
        TTcrystal(crystal='Si',hkl=[1,1,1], thickness=100)
        assert False
    except ValueError as e:
        assert True    

