# -*- coding: utf-8 -*-

'''
Tests for the TTcrystal class. Run with pytest.
'''

import sys
import os.path

sys.path.insert(1, os.path.join(os.path.dirname(__file__),'..'))

from pyTTE import TTscan, Quantity
import numpy as np

def test_required_input():
    #Test that all the inputs are required    
    try:
        TTscan()
        assert False
    except KeyError as e:
        assert True

    try:
        TTscan(constant=Quantity(8,'keV'))
        assert False
    except KeyError as e:
        assert True

    try:
        TTscan(constant=Quantity(8,'keV'), scan=Quantity(np.linspace(-100,100,100),'urad'))
        assert False
    except KeyError as e:
        assert True

    #Correct input
    try:
        TTscan(constant=Quantity(8,'keV'), scan=Quantity(np.linspace(-100,100,100),'urad'),polarization='pi')
        assert True
    except KeyError as e:
        assert False

def test_energy_scan():    

    for p in ['s','sigma','p','pi']: 
        #These should work
        scan = [Quantity(np.linspace(-100,100,100),'meV'), Quantity(np.linspace(-1,1,10),'eV'), 100, 1]        
        for s in scan:        
            try:
                TTscan(constant=Quantity(85,'deg'), scan=s,polarization=p)
                assert True
            except ValueError as e:
                assert False
        #These should not
        scan = [Quantity(np.linspace(-100,100,100),'urad'), Quantity(np.linspace(-1,1,10),'arcsec'), [12, 23], np.array([12, 23]), 'str', -100, 0]        
        for s in scan:                
            try:
                TTscan(constant=Quantity(85,'deg'), scan=s,polarization=p)
                assert False
            except ValueError as e:
                assert True

def test_angle_scan():    

    for p in ['s','sigma','p','pi']: 
        #These should work
        scan = [Quantity(np.linspace(-100,100,100),'urad'), Quantity(np.linspace(-1,1,10),'arcsec'), 100, 1]        
        for s in scan:        
            try:
                TTscan(constant=Quantity(8.5,'keV'), scan=s,polarization=p)
                assert True
            except ValueError as e:
                assert False
        #These should not
        scan = [Quantity(np.linspace(-100,100,100),'meV'), Quantity(np.linspace(-1,1,10),'eV'), -100, 0, [12, 23], np.array([12, 23]), 'str', -100, 0]        
        for s in scan:                
            try:
                TTscan(constant=Quantity(8.5,'keV'), scan=s,polarization=p)
                assert False
            except ValueError as e:
                assert True                