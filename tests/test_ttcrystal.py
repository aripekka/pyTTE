# -*- coding: utf-8 -*-

'''
Tests for the TTcrystal class. Run with pytest.
'''

import sys
import os.path

sys.path.insert(1, os.path.join(os.path.dirname(__file__),'..'))

from pyTTE import TTcrystal, Quantity
import numpy as np


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

    try:
        TTcrystal(crystal='Si',hkl='abc', thickness=Quantity(100,'um'))
        assert False
    except ValueError as e:
        assert True    

    try:
        TTcrystal(crystal='Si',hkl=[1,1,1], thickness=100)
        assert False
    except ValueError as e:
        assert True    

def test_init_optional_inputs():

    #the first elements in the tuple are valid inputs and the second are  invalid 
    opt_inputs = {
                  'asymmetry' : ([Quantity(5,'deg')], [598, Quantity(-5,'keV')]),
                  'in_plane_rotation' : ([Quantity(-5,'deg'), [-1,1,1], (0,1,0), np.array([-1,1,1])], [598, Quantity(-5,'keV'), [1,'1',2], np.array([-1,1])]),
                  'debye_waller' : ([0,0.5,1], [-1, 598, Quantity(-5,'keV'), [1,'1',2]]),
                  'Rx' : ([Quantity(1,'m'), Quantity(-50,'cm'),'inf',None], [598, Quantity(-5,'keV'), [1,'1',2], np.array([-1,1])]),
                  'Ry' : ([Quantity(1,'m'), Quantity(-50,'cm'),'inf',None], [598, Quantity(-5,'keV'), [1,'1',2], np.array([-1,1])]),
                  'R' : ([Quantity(1,'m'), Quantity(-50,'cm'),'inf',None], [598, Quantity(-5,'keV'), [1,'1',2], np.array([-1,1])]),
                  'fix_to_axes' : (['shape', 'torques'], [Quantity(1,'m'), Quantity(-50,'cm'),'inf',None, 598, Quantity(-5,'keV'), [1,'1',2], np.array([-1,1])]),
    }
    #Elastic constants E, nu and S are tested in a separate function
    
    for k in opt_inputs:
        for i in range(1, len(opt_inputs[k][0])):
            kwargs_valid = {'crystal':'Ge', 'hkl' : [7,5,1], 'thickness' : Quantity(1,'mm')}
            kwargs_valid[k] = opt_inputs[k][0][i]

            try:
                TTcrystal(**kwargs_valid)
                assert True
            except ValueError as e:
                assert False

        for i in range(1, len(opt_inputs[k][1])):
            kwargs_invalid = {'crystal':'Ge', 'hkl' : [7,5,1], 'thickness' : Quantity(1,'mm')}
            kwargs_invalid[k] = opt_inputs[k][1][i]

            try:
                TTcrystal(**kwargs_invalid)
                assert False
            except ValueError as e:
                assert True    
        
def test_input_set_correctly():
    #TODO: all the optional inputs
    #Check that the inputs are correctly read
    xtal = TTcrystal(crystal='Si', hkl=[1,-2,3], thickness=Quantity(100,'um'))

    assert xtal.crystal_data['name'] == 'Si'
    assert xtal.hkl[0] == 1
    assert xtal.hkl[1] == -2
    assert xtal.hkl[2] == 3
    assert xtal.thickness.value == 100
    
def test_changing_parameters():
    xtal = TTcrystal(crystal='Si', hkl=[1,-2,3], thickness=Quantity(100,'um'))
    
    xtal.set_crystal('Ge')
    assert xtal.crystal_data['name'] == 'Ge'

    xtal.set_reflection([6,6,0])
    assert xtal.hkl[0] == 6
    assert xtal.hkl[1] == 6
    assert xtal.hkl[2] == 0

    xtal.set_thickness(Quantity(500,'um'))
    assert xtal.thickness.value == 500