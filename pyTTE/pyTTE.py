# -*- coding: utf-8 -*-

from __future__ import division, print_function
import sys
import os

import numpy as np

from scipy.integrate import ode
from scipy.constants.codata import physical_constants

from .quantity import Quantity
from .crystal_vectors import crystal_vectors

import xraylib

class TTcrystal:
    '''
    Contains all the information about the crystal and its depth-dependent deformation.
    '''

    def __init__(self, filepath = None, **kwargs):
        '''
        Initializes the TTcrystal instance. The instance can be initialized either by giving a path
        to a file defining the crystal parameters, or passing them to the function as keyword arguments.
        Keyword parameters are omitted if filepath given.

        Input:
            filepath   = path to the file with crystal parameters

            OR

            crystal    = string representation of the crystal in compliance with xraylib
            hkl        = list-like object of size 3 of the Miller indices of the reflection (ints or floats)
            thickness  = the thickness of the crystal wrapped in a Quantity instance e.g. Quantity(300,'um')

            (Optional)
            asymmetry  = clockwise-positive asymmetry angle wrapped in a Quantity instance.
                         0 deg for symmetric Bragg case (default), 90 deg for symmetric Laue
        '''

        params = {}

        if not filepath == None:
            #check if file path is given
            pass
        else:
            #Check the presence of the required crystal inputs
            try:
                params['crystal'] = kwargs['crystal']
                params['hkl'] = kwargs['hkl']
                params['thickness'] = kwargs['thickness']
            except:
                raise KeyError('At least one of the required keywords crystal, hkl, or thickness is missing!')

            #Optional keywords
            if 'asymmetry' in kwargs.keys():
                params['asymmetry'] = kwargs['asymmetry']
            else:
                params['asymmetry'] = Quantity(0,'deg')

        #used to prevent recalculation of the deformation in parameter set functions during init
        self._initialized = False

        self.set_crystal(params['crystal'])
        self.set_reflection(params['hkl'])
        self.set_thickness(params['thickness'])
        self.set_asymmetry(params['asymmetry'])

        self.update_rotations_and_deformation()
        self._initialized = True

    def set_crystal(self, crystal_str):
        '''
        Changes the crystal keeping other parameters the same. Recalculates
        the crystallographic parameters. The effect on the deformation depends 
        on its previous initialization:
            isotropic -> no change
            automatic anisotropic elastic matrices -> update to new crystal
            manual anisotropic elastic matrices    -> clear

        Input:
            crystal_str = string representation of the crystal in compliance with xraylib
        '''

        #Check whether the crystal_str is valid and available in xraylib
        if type(crystal_str) == type(''):
            if crystal_str in xraylib.Crystal_GetCrystalsList():
                self.crystal_data = xraylib.Crystal_GetCrystal(crystal_str)
            else:
                raise ValueError('The given crystal_str not found in xraylib!')
        else:
            raise ValueError('Input argument crystal_str is not type str!')

        #calculate the direct and reciprocal primitive vectors 
        self.direct_primitives, self.reciprocal_primitives = crystal_vectors(self.crystal_data)

        #skip this if the function is used as a part of initialization
        if self._initialized:
            self.update_rotations_and_deformation()

    def set_reflection(self, hkl):
        '''
        Set a new reflection and calculate the new crystallographic data and deformation
        for rotated crystal.

        Input:
            crystal_str = string representation of the crystal in compliance with xraylib
        '''

        #Check whether the hkl is valid
        hkl_list = list(hkl)
        if len(hkl_list) == 3:
            for i in range(3):
                if not type(hkl_list[i]) in [type(1),type(1.0)]:
                    raise ValueError('Elements of hkl have to be of type int or float!')
            self.hkl = hkl_list               
        else:
            raise ValueError('Input argument hkl does not have 3 elements!')

        #skip this if the function is used as a part of initialization
        if self._initialized:
            self.update_rotations_and_deformation()

    def set_thickness(self, thickness):
        '''
        Set crystal thickness and recalculate the deformation field.

        Input:
            thickness = the thickness of the crystal wrapped in a Quantity instance e.g. Quantity(300,'um')
        '''

        #Check that the crystal thickness is valid
        if isinstance(thickness,Quantity) and thickness.type() == 'length':
            self.thickness = Quantity(thickness.value,thickness.unit)
        else:
            raise ValueError('Thickness has to be a Quantity instance of type length!')

        #skip this if the function is used as a part of initialization
        if self._initialized:
            self.update_rotations_and_deformation()

    def set_asymmetry(self, asymmetry):
        '''
        Set crystal thickness and recalculate the deformation field.

        Input:
            asymmetry = clockwise-positive asymmetry angle wrapped in a Quantity instance 0 
                        for symmetric Bragg case (default), 90 deg for symmetric Laue
        '''

        if isinstance(asymmetry,Quantity) and asymmetry.type() == 'angle':
            self.asymmetry = Quantity(asymmetry.value,asymmetry.unit)
        else:
            raise ValueError('Asymmetry angle has to be a Quantity instance of type angle!')

        #skip this if the function is used as a part of initialization
        if self._initialized:
            self.update_rotations_and_deformation()

    def update_rotations_and_deformation(self):
        print('HELP! I am update_rotations_and_deformation() and I am not implemented yet!')

    def __str__(self):
        return 'Crystal: ' + self.crystal_data['name'] + '\n' +\
               'Crystallographic parameters:\n' +\
               '    a = ' + str(self.crystal_data['a']*0.1)[:8] + ' nm,  b = ' + str(self.crystal_data['b']*0.1)[:8] + ' nm,  c = ' + str(self.crystal_data['c']*0.1)[:8] + ' nm\n'+\
               '    alpha = ' + str(self.crystal_data['alpha']) + ' deg,  beta = ' + str(self.crystal_data['beta']) + ' nm,  gamma = ' + str(self.crystal_data['gamma']) + ' deg\n'+\
               'Direct primitive vectors (columns, in nm):\n'+ np.array2string(0.1*self.direct_primitives,precision=4,suppress_small=True)+'\n'+\
               'Reciprocal primitive vectors (columns, in 1/nm):\n'+ np.array2string(10*self.reciprocal_primitives,precision=4,suppress_small=True)+'\n'+\
               'Reflection: '+str(self.hkl)+'\n'+\
               'Asymmetry angle: ' + str(self.asymmetry)+'\n'+\
               'Thickness: ' + str(self.thickness)
class TTscan:
    
    #Class containing all the parameters for the energy or angle scan to be simulated.   

    def __init__(self, filepath = None, **kwargs):
        '''
        Initializes a TTscan instance. The instance can be initialized either by giving a path
        to a file defining the scan parameters, or passing them to the function as keyword arguments.
        Keyword parameters are omitted if filepath given.

        Input:
            filepath = path to the file with crystal parameters

            OR

            constant     = Instance of Quantity of type energy or angle. Determines value of the incident photon 
                           energy or the Bragg angle fixed during the scan
            scan         = Either a list of scan points wrapped in a Quantity e.g. Quantity(np.linspace(-100,100,250),'meV')
                           OR a non-negative integer number of scan points for automatic scan range determination. The unit 
                           of Quantity has to be angle if the unit of constant is energy and vice versa.
            polarization = 'sigma' or 's' for sigma-polarized beam OR 'pi' or 'p' for pi-polarized beam

        '''
        #Validate inputs

        params = {}

        if not filepath == None:
            #check if file path is given
            pass
        else:
            #Check the presence of the required crystal inputs
            try:
                params['constant']     = kwargs['constant']
                params['scan']         = kwargs['scan']
                params['polarization'] = kwargs['polarization']
            except:
                raise KeyError('At least one of the required keywords constant, sca, or polarization is missing!')                

        self.set_polarization(params['polarization'])
        self.set_scan(params['scan'], params['constant'])

    def set_polarization(self,polarization):
        if type(polarization) == type('') and polarization.lower() in ['sigma','s']:
            self.polarization = 'sigma'
        elif type(polarization) == type('') and polarization.lower() in ['pi','p']:
            self.polarization = 'pi'
        else:
            raise ValueError("Invalid polarization! Choose either 'sigma' or 'pi'.")       

    def set_scan(self, scan, constant):        
        if isinstance(constant, Quantity) and constant.type() in ['angle', 'energy']:
            if constant.type() == 'angle':
                self.scantype = 'energy'
            else:
                self.scantype = 'angle'
            self.constant = Quantity(constant.value,constant.unit)
        else:
            raise ValueError('constant has to be an instance of Quantity of type energy or angle!')

        if isinstance(scan, Quantity) and scan.type() == self.scantype:
            self.scan = ('manual', Quantity(scan.value,scan.unit))
        elif type(scan) == type(1) and scan > 0:
            self.scan = ('automatic',scan)
        else:
            raise ValueError('scan has to be either a Quantity of type energy (for angle constant) or angle (for energy constant) or a non-negative integer!')

    def __str__(self):
        return 'Scan type: ' + self.scantype + '\n' +\
               'Scan constant: ' + str(self.constant) +'\n' +\
               'Polarization: ' + self.polarization  +'\n' +\
               'Scan points: ' + str(self.scan[0])  +'\n'

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
    h = 2*np.pi/d

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
    beta = h*gammah*(np.sin(th)-wavelength/(2*d))

    #For deformation, the strain term function defined later stepwise 
    if displacement_jacobian == None:
        def strain_term(z): 
            return 0

    #INTEGRATION

    #Define ODEs and their Jacobians
    if geometry == 'bragg':
        print('Transmission in the Bragg case not implemented!')
        reflectivity = np.zeros(scan.shape)
        transmission = -np.ones(scan.shape)
    else:
        forward_diffraction = np.zeros(scan.shape)
        diffraction = np.zeros(scan.shape)

    #Solve the equation
    sys.stdout.write('Solving...0%')
    sys.stdout.flush()
    
    for step in range(len(scan)):
        #local variables for speedup
        c0_step   = c0[step]
        cb_step   = cb[step]
        ch_step   = ch[step]
        beta_step = beta[step]
        g0_step   = g0[step]
        gb_step   = gb[step]
        gammah_step = gammah[step]

        #Define deformation term for bent crystal
        if not displacement_jacobian == None:
            #Precomputed sines and cosines
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            if is_escan:
                cot_alpha0 = np.cos(alpha0)/np.sin(alpha0)
                sin_alphah = np.sin(alphah)
                cos_alphah = np.cos(alphah)

                def strain_term(z):
                    x = -z*cot_alpha0
                    u_jac = displacement_jacobian(x,z)
                    duh_dsh = h*(sin_phi*cos_alphah*u_jac[0][0] 
                                +sin_phi*sin_alphah*u_jac[0][1]
                                +cos_phi*cos_alphah*u_jac[1][0]
                                +cos_phi*sin_alphah*u_jac[1][1]
                                )
                    return gammah_step*duh_dsh
            else:
                cot_alpha0 = np.cos(alpha0[step])/np.sin(alpha0[step])
                sin_alphah = np.sin(alphah[step])
                cos_alphah = np.cos(alphah[step])

                def strain_term(z):
                    x = -z*cot_alpha0
                    u_jac = displacement_jacobian(x,z)
                    duh_dsh = h*(sin_phi*cos_alphah*u_jac[0][0]
                                +sin_phi*sin_alphah*u_jac[0][1]
                                +cos_phi*cos_alphah*u_jac[1][0] 
                                +cos_phi*sin_alphah*u_jac[1][1]
                                )
                    return gammah_step*duh_dsh
        
        if geometry == 'bragg':
            def ksiprime(z,ksi):
                return 1j*(cb_step*ksi*ksi+(c0_step+beta_step+strain_term(z))*ksi+ch_step)

            def ksiprime_jac(z,ksi):
                return 2j*cb_step*ksi+1j*(c0_step+beta_step+strain_term(z))

            r=ode(ksiprime,ksiprime_jac)
        else:
            def TTE(z,Y):
                return [1j*(cb_step*Y[0]*Y[0]+(c0_step+beta_step+strain_term(z))*Y[0]+ch_step),\
                        -1j*(g0_step+gb_step*Y[0])*Y[1]]

            def TTE_jac(z,Y):
                return [[2j*cb_step*Y[0]+1j*(c0_step+beta_step+strain_term(z)), 0],\
                        [-1j*gb_step*Y[1],-1j*(g0_step+gb_step*Y[0])]]

            r=ode(TTE,TTE_jac)

        r.set_integrator('zvode',method='bdf',with_jacobian=True, min_step=min_int_step,max_step=1e-4,nsteps=50000)

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

