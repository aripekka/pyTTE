import numpy as np
import time
import os

from .. import TakagiTaupin, TTcrystal, TTscan, Quantity
from .. import deformation

def bragg_reflectivity_sigma():

    print ('Computing Bragg reflectivity for 300 um thick, non-bent GaAs(400)')
    print ('for sigma polarized beam at 6 keV in angle domain.')

    ttx = TTcrystal(crystal = 'GaAs', hkl = [4,0,0], thickness = Quantity(300,'um'))
    tts = TTscan(scan = Quantity(np.linspace(-10,30,300),'arcsec'), constant = Quantity(6,'keV'), polarization = 'sigma')
    tt = TakagiTaupin(ttx,tts)

    t0 = time.time()
    th, R , T = tt.run()

    print('It took ' + str(time.time()-t0) + ' seconds to solve ' + str(th.size) + ' points.\n')

    A=np.loadtxt(os.path.dirname(__file__)+'/reference_curves/GaAs400_6keV_300micron_unbent_sigma.dat')

    th_ref = A[:,0]
    R_ref = A[:,1]

    return th, R, th_ref, R_ref

def bragg_reflectivity_pi_asymmetric():

    print ('Computing Bragg reflectivity for 20 um thick, non-bent LiF(220) with')
    print ('asymmetry of 10 deg for pi polarized beam at 8 keV in angle domain.')

    ttx = TTcrystal(crystal = 'LiF', hkl = [2,2,0], thickness = Quantity(20,'um'), asymmetry = Quantity(10,'deg'))
    tts = TTscan(scan = Quantity(np.linspace(-2,8,300),'arcsec'), constant = Quantity(8,'keV'), polarization = 'pi')
    tt = TakagiTaupin(ttx,tts)

    t0 = time.time()
    th, R , T = tt.run()

    print('It took ' + str(time.time()-t0) + ' seconds to solve ' + str(th.size) + ' points.\n')

    A=np.loadtxt(os.path.dirname(__file__)+'/reference_curves/LiF220_8keV_20micron_unbent_pi_asymm_10.dat')

    th_ref = A[:,0]
    R_ref = A[:,1]

    return th, R, th_ref, R_ref

def laue_symmetric():

    print ('Computing Laue reflectivity and transmission for 100 um thick')
    print ('non-bent Ge(111) for sigma polarized beam at 6 keV in angle domain.')


    ttx = TTcrystal(crystal = 'Ge', hkl = [1,1,1], thickness = Quantity(100,'um'), asymmetry = Quantity(90,'deg'))
    tts = TTscan(scan = Quantity(np.linspace(-40,20,300),'arcsec'), constant = Quantity(6,'keV'), polarization = 'sigma')
    tt = TakagiTaupin(ttx,tts)

    t0 = time.time()
    th, R , T = tt.run()

    print('It took ' + str(time.time()-t0) + ' seconds to solve ' + str(th.size) + ' points.\n')

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

    nu = 0.27
    Rx = Quantity(1,'m')

    #Anticlastic bending radius
    Ry = -Rx/nu           

    ttx = TTcrystal(crystal = 'Si', hkl = [6,6,0], thickness = Quantity(300,'um'), Rx = Rx, Ry=Ry, E = Quantity(160,'GPa'), nu = 0.27)
    tts = TTscan(scan = Quantity(np.linspace(-450,250,300),'meV'), constant = Quantity(88,'deg'), polarization = 'sigma')
    tt = TakagiTaupin(ttx,tts)

    t0 = time.time()
    e, R , T = tt.run()
    print('It took ' + str(time.time()-t0) + ' seconds to solve ' + str(e.size) + ' points.\n')

    #TODO: ADD REFERENCE FOR BENT (multilamellar?)

    return e, R

