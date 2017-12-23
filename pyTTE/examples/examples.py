import numpy as np
import time
import os

from .. import takagitaupin
from .. import deformation

def bragg_reflectivity_sigma():
    print ('Computing Bragg reflectivity for 300 um thick, non-bent GaAS(400)')
    print ('for sigma polarized beam at 6 keV in angle domain.')

    th = np.linspace(-10,30,150)

    t0 = time.time()
    R,T=takagitaupin('angle',th,6,'sigma','GaAs',[4,0,0],0,300,None,1,1e-10)
    print('It took ' + str(time.time()-t0) + ' seconds to solve ' + \
          str(th.size) + ' points.\n')


    A=np.loadtxt(os.path.dirname(__file__)+'/reference_curves/GaAs400_6keV_300micron_unbent_sigma.dat')

    th_ref = A[:,0]
    R_ref = A[:,1]


    return th, R, th_ref, R_ref

def bragg_reflectivity_pi_asymmetric():
    print ('Computing Bragg reflectivity for 20 um thick, non-bent Li(220) with')
    print ('asymmetry of 10 deg for pi polarized beam at 8 keV in angle domain.')

    th = np.linspace(-2,8,150)

    t0 = time.time()
    R,T=takagitaupin('angle',th,8,'pi','LiF',[2,2,0],10,20,None,1,1e-10)
    print('It took ' + str(time.time()-t0) + ' seconds to solve ' + \
          str(th.size) + ' points.\n')

    A=np.loadtxt(os.path.dirname(__file__)+'/reference_curves/LiF220_8keV_20micron_unbent_pi_asymm_10.dat')

    th_ref = A[:,0]
    R_ref = A[:,1]

    return th, R, th_ref, R_ref

def laue_symmetric():
    print ('Computing Laue reflectivity and transmission for 100 um thick')
    print ('non-bent Ge(111) for sigma polarized beam at 6 keV in angle domain.')

    th = np.linspace(-40,20,150)

    t0 = time.time()
    R,T=takagitaupin('angle',th,6,'sigma','Ge',[1,1,1],90,100,None,1,1e-11)
    print('It took ' + str(time.time()-t0) + ' seconds to solve ' + \
          str(th.size) + ' points.\n')

    A=np.loadtxt(os.path.dirname(__file__)+'/reference_curves/Ge111_6keV_100micron_unbent_laue_refl.dat')
    B=np.loadtxt(os.path.dirname(__file__)+'/reference_curves/Ge111_6keV_100micron_unbent_laue_trans.dat')

    th_ref = A[:,0]
    R_ref = A[:,1]
    T_ref = B[:,1]

    return th, R, T, th_ref, R_ref, T_ref


def bragg_bent_symmetric():
    print ('Computing Bragg reflectivity for 300 um thick, cylindrically bent')
    print ('Si(660) (1m bending radius) for sigma polarized at 88 deg in energy domain.')

    e = np.linspace(-450,250,150)

    thickness=300
    R_bend = 1
    nu = 0.27

    #Define Jacobian for cylindrical bending
    ujac= deformation.isotropic_plate(R_bend,-R_bend/nu,nu,thickness*1e-6)

    t0 = time.time()
    R,T=takagitaupin('energy',e,88,'sigma','Si',[6,6,0],0,thickness,ujac,1,1e-10)
    print('It took ' + str(time.time()-t0) + ' seconds to solve ' + \
          str(e.size) + ' points.\n')

    #TODO: ADD REFERENCE FOR BENT (multilamellar?)

    return e, R

