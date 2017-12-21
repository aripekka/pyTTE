from __future__ import division, print_function
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import ode
from scipy.constants.codata import physical_constants

import xraylib

def takagitaupin(scantype,scan,constant,polarization,crystal_str,hkl,asymmetry,thickness,displacement_jacobian = None,debyeWaller=1.0,min_int_step=1e-10):
    '''
    1D TT-solver.
    
    Input:
    scantype = 'energy' or 'angle'
    scan =  relative to the Bragg's energy in meV (energy scan) OR relative to the Bragg's angle in arcsec (angle scan). 
            scan is an numpy array containing these values OR an integer N generating automatic scan with N points. 
    constant = incidence angle respect to the diffracting planes in degrees (energy scan) OR photon energy in keV (angle scan) 
    polarization = 'sigma' or 'pi'
    crystal_str = supports crystals included in xraylib, e.g. 'Si', 'Ge', 'LiF'
    hkl = [h,k,l] (Miller indices)
    asymmetry = asymmetry angle (in deg, positive to clockwise direction)
    thickness = crystal thickness in microns
    displacement_jacobian = a function giving the jacobian of the displacement field as function of position (x,y). 
                            Note: y points upwards the crystal 
    debyeWaller = Debye-Waller factor
    min_int_step = minumum integration step
    '''

    if scantype == 'energy':
        is_escan = True
        scantype = 'energy'
    elif scantype == 'angle':
        is_escan = False
        scantype = 'angle'


    if scantype == 'energy':
        print('Computing elastic line for ' + str(hkl) + ' reflection of ' \
              + crystal_str + ' crystal in energy domain.' )
        is_escan = True
    elif scantype == 'angle':
        print('Computing elastic line for ' + str(hkl) + ' reflection of ' \
              + crystal_str + ' crystal in angle domain.' )
        is_escan = False

    #type conversions
    if type(scan) is int:
        print('AUTOMATIC LIMITS NOT IMPLEMENTED YET!')
        print('Function terminated.')
        return None

    scan=np.array(scan)
    
    #Unit conversions
    thickness = thickness*1e-6 #wafer thickness in meters

    #constants
    crystal = xraylib.Crystal_GetCrystal(crystal_str)

    hc = physical_constants['Planck constant in eV s'][0]*physical_constants['speed of light in vacuum'][0]*1e3 #in meV*m
    d = xraylib.Crystal_dSpacing(crystal,*hkl)*1e-10 #in m
    V = crystal['volume']*1e-30 # volume of unit cell in m^3
    r_e = physical_constants['classical electron radius'][0]

    print('')
    print('Crystal     : ', crystal_str)
    print('Reflection  : ', hkl)
    print('d_hkl       : ', d, ' m')
    print('Cell volume : ', V, ' m^3')
    print('')


    #asymmetry angle
    phi=np.radians(asymmetry)

    #Setup scan variables and constants
    if is_escan:
        escan=scan

        th0=np.radians(constant)
        th=th0

        #Conversion of incident photon energy to wavelength
        E0 = hc/(2*d*np.sin(th)) #in meV

        wavelength = hc/(E0+escan) #in m
        k = 2*np.pi/wavelength #in 1/m

    else:
        E0 = constant*1e6 #in meV

        wavelength = hc/E0 #in m
        k = 2*np.pi/wavelength #in 1/m

        if not hc/(2*d*E0) > 1:
            th0 = np.arcsin(hc/(2*d*E0))
        else:
            print('Given energy below the backscattering energy!')
            print('Setting theta to 90 deg.')
            th0 = np.pi/2

        ascan = scan*np.pi/648000 #from arcsec to rad
        th = th0+ascan

    #Incidence and exit angles
    alpha0 = th+phi
    alphah = th-phi
    
    #Direction parameters
    gamma0 = np.ones(scan.shape)/np.sin(alpha0)
    gammah = np.ones(scan.shape)/np.sin(alphah)

    if np.mean(gammah) < 0:
        print('The direction of diffraction in to the crystal -> Laue case')
        geometry = 'laue'
    else:
        print('The direction of diffraction out of the crystal -> Bragg case')
        geometry = 'bragg'

    #Polarization
    if polarization == 'sigma':
        C = 1;
        print('Solving for sigma-polarization')
    else:
        C = np.cos(2*th);
        print('Solving for pi-polarization')

    print('Asymmetry angle : ', phi,' rad, ', np.degrees(phi), ' deg')
    print('Wavelength      : ', hc/E0*1e10, ' Angstrom ')
    print('Energy          : ', E0*1e-6, ' keV ')
    
    print('Bragg angle     : ', th0,' rad, ', np.degrees(th0), ' deg')
    print('Incidence angle : ', th0+phi,' rad ', np.degrees(th0+phi), ' deg')
    print('Exit angle      : ', th0-phi,' rad ', np.degrees(th0-phi), ' deg')
    print('')

    #Compute susceptibilities
    if is_escan:
        F0 = np.zeros(escan.shape,dtype=np.complex)
        Fh = np.zeros(escan.shape,dtype=np.complex)
        Fb = np.zeros(escan.shape,dtype=np.complex)

        for ii in range(escan.size):    
            F0[ii] = xraylib.Crystal_F_H_StructureFactor(crystal, (E0+escan[ii])*1e-6, 0, 0, 0, debyeWaller, 1.0)
            Fh[ii] = xraylib.Crystal_F_H_StructureFactor(crystal, (E0+escan[ii])*1e-6, hkl[0], hkl[1], hkl[2], debyeWaller, 1.0)
            Fb[ii] = xraylib.Crystal_F_H_StructureFactor(crystal, (E0+escan[ii])*1e-6, -hkl[0], -hkl[1], -hkl[2], debyeWaller, 1.0)
    else:
        F0 = xraylib.Crystal_F_H_StructureFactor(crystal, E0*1e-6, 0, 0, 0, debyeWaller, 1.0)
        Fh = xraylib.Crystal_F_H_StructureFactor(crystal, E0*1e-6, hkl[0], hkl[1], hkl[2], debyeWaller, 1.0)
        Fb = xraylib.Crystal_F_H_StructureFactor(crystal, E0*1e-6, -hkl[0], -hkl[1], -hkl[2], debyeWaller, 1.0)

    cte = - r_e * wavelength*wavelength/(np.pi * V)
    chi0 = np.conj(cte*F0)
    chih = np.conj(cte*Fh)
    chib = np.conj(cte*Fb)

    print('F0   : ',np.mean(F0))
    print('Fh   : ',np.mean(Fh))
    print('Fb   : ',np.mean(Fb))
    print('')
    print('chi0 : ',np.mean(chi0))
    print('chih : ',np.mean(chih))
    print('chib : ',np.mean(chib))
    print('')
    print('(Mean F and chi values for energy scan)')
    print('')

    ######################
    #COEFFICIENTS FOR TTE#
    ######################

    #For solving ksi = Dh/D0 
    c0 = k*chi0/2*(gamma0+gammah)*np.ones(scan.shape)
    ch = k*C*chih*gammah/2*np.ones(scan.shape)
    cb = k*C*chib*gamma0/2*np.ones(scan.shape)

    #For solving Y = D0 
    g0 = k*chi0/2*gamma0*np.ones(scan.shape)
    gb = k*C*chib/2*gamma0*np.ones(scan.shape)

    #deviation from the kinematical Bragg condition for unstrained crystal
    beta = 2*np.pi/d*gammah*(np.sin(th)-wavelength/(2*d))

    #Bending
    if not displacement_jacobian == None:

        def strain_term(z,step):
            if is_escan:
                x = -z*np.cos(alpha0)/np.sin(alpha0)
                u_jac = displacement_jacobian(x,z)
                duh_dsh = 2*np.pi/d*(np.sin(phi)*np.cos(alphah)*u_jac[0,0] 
                                    +np.sin(phi)*np.sin(alphah)*u_jac[0,1]
                                    +np.cos(phi)*np.cos(alphah)*u_jac[1,0]
                                    +np.cos(phi)*np.sin(alphah)*u_jac[1,1]
                                    )
                return gammah[step]*duh_dsh
            else:
                x = -z*np.cos(alpha0[step])/np.sin(alpha0[step])
                u_jac = displacement_jacobian(x,z)
                duh_dsh = 2*np.pi/d*(np.sin(phi)*np.cos(alphah[step])*u_jac[0,0]
                                    +np.sin(phi)*np.sin(alphah[step])*u_jac[0,1]
                                    +np.cos(phi)*np.cos(alphah[step])*u_jac[1,0] 
                                    +np.cos(phi)*np.sin(alphah[step])*u_jac[1,1]
                                    )
                return gammah[step]*duh_dsh

    else:
        def strain_term(z,step): 
            return 0

    #INTEGRATION

    #Define ODEs and their Jacobians
    if geometry == 'bragg':
        print('Transmission in the Bragg case not implemented!')

        reflectivity = np.zeros(scan.shape)
        transmission = -np.ones(scan.shape)

        def ksiprime(z,ksi,step):
            return 1j*cb[step]*ksi**2+1j*(c0[step]+beta[step]+strain_term(z,step))*ksi+1j*ch[step]

        def ksiprime_jac(z,ksi,step):
            return 2j*cb[step]*ksi+1j*(c0[step]+beta[step]+strain_term(z,step))

    else:

        forward_diffraction = np.zeros(scan.shape)
        diffraction = np.zeros(scan.shape)

        def TTE(z,Y,step):
            return [1j*cb[step]*Y[0]**2+1j*(c0[step]+beta[step]+strain_term(z,step))*Y[0]+1j*ch[step],\
                    -1j*(g0[step]+gb[step]*Y[0])*Y[1]]

        def TTE_jac(z,Y,step):
            return [[2j*cb[step]*Y[0]+1j*(c0[step]+beta[step]+strain_term(z,step)), 0],\
                    [-1j*gb[step]*Y[1],-1j*(g0[step]+gb[step]*Y[0])]]


    #Solve the equation
    sys.stdout.write('Solving...0%')
    sys.stdout.flush()
    
    for step in range(len(scan)):
        if geometry == 'bragg':
            r=ode(ksiprime,ksiprime_jac)
        else:
            r=ode(TTE,TTE_jac)

        r.set_integrator('zvode',method='bdf',with_jacobian=True, min_step=min_int_step,max_step=1e-4,nsteps=50000)
        r.set_f_params(step)
        r.set_jac_params(step)
        
        if geometry == 'bragg':
            r.set_initial_value(0,-thickness)
            res=r.integrate(0)     
            reflectivity[step]=np.abs(res[0])**2*gamma0[step]/gammah[step] #gamma-part takes into account beam footprints 
        else:
            r.set_initial_value([0,1],0)
            res=r.integrate(-thickness)
            diffraction[step] = np.abs(res[0]*res[1])**2
            forward_diffraction[step] = np.abs(res[1])**2

        sys.stdout.write('\rSolving...%0.1f%%' % (100*(step+1)/len(scan),))  
        sys.stdout.flush()

    sys.stdout.write('\r\nDone.\n')
    sys.stdout.flush()

    if geometry == 'bragg':    
        return reflectivity, transmission
    else:    
        return diffraction, forward_diffraction

