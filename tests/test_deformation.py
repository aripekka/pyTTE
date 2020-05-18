# -*- coding: utf-8 -*-

'''
Tests for the deformation functions. Run with pytest.
'''

import sys
import os.path

sys.path.insert(1, os.path.join(os.path.dirname(__file__),'..'))

from pyTTE.deformation import *
import numpy as np

E = 165
nu = 0.27

#isotropic S
S = np.zeros((6,6))
S[0,0] = 1
S[1,1] = 1
S[2,2] = 1
S[3,3] = 2*(1+nu)
S[4,4] = 2*(1+nu)
S[5,5] = 2*(1+nu)
S[0,1] = -nu
S[0,2] = -nu
S[1,2] = -nu
S[1,0] = -nu
S[2,0] = -nu
S[2,1] = -nu
S = S/E

def test_isotropic_R_input():    
    #test R1, R2
    
    #These should work
    for R in [1.0,2,-.5]:
        J,R1,R2 = isotropic_plate(R,1,nu,1e-4)
        J,_, R2 = isotropic_plate(1,R,nu,1e-4)
        assert R1 == R
        assert R2 == R
        
    for R in ['inf', '-inf',float('inf'),np.inf,-np.inf]:
        J,R1,R2 = isotropic_plate(R,1,nu,1e-4)
        J,_, R2 = isotropic_plate(1,R,nu,1e-4)
        assert R1 == 'inf'
        assert R2 == 'inf'
        
    for R in [None]:
        J,R1,R2 = isotropic_plate(R,1,nu,1e-4)
        J,_, R2 = isotropic_plate(1,R,nu,1e-4)
        assert R1 == -1/0.27
        assert R2 == -1/0.27
    
    #These should fail
    for R in ['sdsd',0,'2']:
        all_pass = True
        try:
            J,R1,R2 = isotropic_plate(R,1,nu,1e-4)
            all_pass = False
        except:
            pass            
        try:            
            J,_, R2 = isotropic_plate(1,R,nu,1e-4)                        
            all_pass = False
        except:
            pass

def test_anisotropic_fixed_torque_R_input():    
    #test R1, R2
    
    #These should work
    for R in [1.0,2,-.5]:
        J,R1,R2 = anisotropic_plate_fixed_torques(R,1,S,1e-4)
        J,_, R2 = anisotropic_plate_fixed_torques(1,R,S,1e-4)
        assert R1 == R
        assert R2 == R
        
    for R in ['inf', '-inf',float('inf'),np.inf,-np.inf]:
        J,R1,R2 = anisotropic_plate_fixed_torques(R,1,S,1e-4)
        J,_, R2 = anisotropic_plate_fixed_torques(1,R,S,1e-4)
        assert R1 == 'inf'
        assert R2 == 'inf'
        
    for R in [None]:
        J,R1,R2 = anisotropic_plate_fixed_torques(R,1,S,1e-4)
        J,_, R2 = anisotropic_plate_fixed_torques(1,R,S,1e-4)
        assert R1 == -1/0.27
        assert R2 == -1/0.27
    
    #These should fail
    for R in ['sdsd',0,'2']:
        all_pass = True
        try:
            J,R1,R2 = anisotropic_plate_fixed_torques(R,1,S,1e-4)
            all_pass = False
        except:
            pass            
        try:            
            J,_, R2 = anisotropic_plate_fixed_torques(1,R,S,1e-4)                        
            all_pass = False
        except:
            pass

def test_anisotropic_fixed_shape_R_input():    
    #test R1, R2
    
    #These should work
    for R in [1.0,2,-.5]:
        J,R1,R2 = anisotropic_plate_fixed_shape(R,1,S,1e-4)
        J,_,R2 = anisotropic_plate_fixed_shape(1,R,S,1e-4)
        assert R1 == R
        assert R2 == R
        
    for R in ['inf', '-inf',float('inf'),np.inf,-np.inf]:
        J,R1,R2 = anisotropic_plate_fixed_shape(R,1,S,1e-4)
        J,_,R2 = anisotropic_plate_fixed_shape(1,R,S,1e-4)
        assert R1 == 'inf'
        assert R2 == 'inf'
        
    #These should fail
    for R in ['sdsd',0,'2',None]:
        all_pass = True
        try:
            J = anisotropic_plate_fixed_shape(R,1,S,1e-4)
            all_pass = False
        except:
            pass            
        try:            
            J = anisotropic_plate_fixed_shape(1,R,S,1e-4)                        
            all_pass = False
        except:
            pass

def test_consistency():
    #The models should give the same Jacobian for the same isotropic input and
    
    x = np.linspace(-1e-4,1e-4,101)
    z = np.flipud(np.linspace(-1e-4,1e-4,101))
    
    meps = np.finfo(float).eps
    
    for Rs in [(1,1),(-1,1),(2,0.5),(1,'inf'),('inf','inf')]:
        J1 = isotropic_plate(Rs[0],Rs[1],nu,1e-4)[0]
        J2 = anisotropic_plate_fixed_torques(Rs[0],Rs[1],S,1e-4)[0]
        J3 = anisotropic_plate_fixed_shape(Rs[0],Rs[1],S,1e-4)[0]
        
        for i in range(x.size):
            J1i = np.array(J1(x[i],z[i]))
            J2i = np.array(J2(x[i],z[i]))
            J3i = np.array(J3(x[i],z[i]))     
            
            assert np.all(np.abs(J1i-J2i) < meps), J1i-J2i
            assert np.all(np.abs(J1i-J3i) < meps), J1i-J3i
