# -*- coding: utf-8 -*-
from __future__ import division, print_function
from scipy.integrate import ode
from .ttcrystal import TTcrystal
from .quantity import Quantity
from .ttscan import TTscan
import matplotlib.pyplot as plt
import multiprocess
import numpy as np
import xraylib
import sys

class TakagiTaupin:

    ##############################
    # Methods for initialization #
    ##############################    

    def __init__(self, TTcrystal_object = None, TTscan_object = None):
        '''
        Initialize the TakagiTaupin instance.


        Input:
            TTcrystal_object = An existing TTcrystal instance for crystal 
                               parameters or a path to a file where they are 
                               defined. For any other types the crystal 
                               parameters are not set.
            
            TTscan_object    = An existing TTscan instance for scan parameters 
                               or a path to a file where they are defined.
                               For any other types the scan parameters are not set.
        '''

        self.crystal_object = None
        self.scan_object    = None
        self.solution       = None

        self.set_crystal(TTcrystal_object)
        self.set_scan(TTscan_object)



    def set_crystal(self, TTcrystal_object):
        '''
        Sets the crystal parameters for the scan instance. Any existing solutions
        are cleared.

        Input:
            TTcrystal_object = An existing TTcrystal instance for crystal 
                               parameters or a path to a file where they are 
                               defined. For any other types the crystal 
                               parameters are not set and the solution is not
                               cleared.
        '''

        if isinstance(TTcrystal_object, TTcrystal): 
            self.crystal_object = TTcrystal_object
            self.solution       = None
        elif type(TTcrystal_object) == str:
            try:
                self.crystal_object = TTcrystal(TTcrystal_object)
                self.solution       = None
            except Exception as e:
                print(e)
                print('Error initializing TTcrystal from file! Crystal not set.')
        else:
            print('ERROR! Not an instance of TTcrystal or None! Crystal not set.')



    def set_scan(self, TTscan_object):
        '''
        Sets the crystal parameters for the scan instance. Any existing solutions
        are cleared.

        Input:
            TTscan_object    = An existing TTscan instance for scan parameters 
                               or a path to a file where they are defined. For 
                               any other types the scan parameters are not set
                               and the solution is not cleared.
        '''

        if isinstance(TTscan_object, TTscan):
            self.scan_object = TTscan_object
            self.solution    = None
        elif type(TTscan_object) == str:
            try:
                self.scan_object = TTscan(TTscan_object)
                self.solution    = None
            except Exception as e:
                print(e)
                print('Error initializing TTscan from file! Scan not set.')
        else:
            print('ERROR! Not an instance of TTscan or None! Scan not set.')


    ####################################
    # Auxiliary methods for TT solving #
    ####################################    

    @staticmethod
    def calculate_structure_factors(crystal, hkl, energy, debye_waller):
        '''
        Calculates the structure factors F_0 F_h and F_bar{h}.
        
        Input:
            crystal = a dictionary returned by xraylib's Crystal_GetCrystal()
            hkl = 3 element list containing the Miller indeces of the reflection
            energy = Quantity instance of type energy. May be a single number or an array
            debye_waller = The Debye-Waller factor
        '''

        energy_in_keV = energy.in_units('keV')
        
        #preserve the original shape and reshape energy to 1d array
        orig_shape = energy_in_keV.shape
        energy_in_keV = energy_in_keV.reshape(-1)

        F0 = np.zeros(energy_in_keV.shape, dtype=np.complex)
        Fh = np.zeros(energy_in_keV.shape, dtype=np.complex)
        Fb = np.zeros(energy_in_keV.shape, dtype=np.complex)        

        for i in range(energy_in_keV.size):    
            F0[i] = xraylib.Crystal_F_H_StructureFactor(crystal, energy_in_keV[i], 0, 0, 0, 1.0, 1.0)
            Fh[i] = xraylib.Crystal_F_H_StructureFactor(crystal, energy_in_keV[i],  hkl[0],  hkl[1],  hkl[2], debye_waller, 1.0)
            Fb[i] = xraylib.Crystal_F_H_StructureFactor(crystal, energy_in_keV[i], -hkl[0], -hkl[1], -hkl[2], debye_waller, 1.0)

        return F0.reshape(orig_shape), Fh.reshape(orig_shape), Fb.reshape(orig_shape)



    def run(self):
        #Check that the required scan parameters are in place
        if self.crystal_object == None:
            print('ERROR! No crystal data found, TTcrystal object needed.')
            return
        if self.scan_object == None:
            print('ERROR! No scan data found, TTscan object needed.')
            return

        ################################################
        #Calculate the constants needed by TT-equations#
        ################################################

        hc = Quantity(1.23984193,'eV um') #Planck's constant * speed of light
      
        crystal = self.crystal_object.crystal_data
        hkl = self.crystal_object.hkl
        phi = self.crystal_object.asymmetry
        displacement_jacobian = self.crystal_object.displacement_jacobian
        debye_waller = self.crystal_object.debye_waller

        d   = Quantity(xraylib.Crystal_dSpacing(crystal,*hkl),'A')     #spacing of Bragg planes
        V   = Quantity(crystal['volume'],'A^3')                        #volume of unit cell
        r_e = Quantity(2.81794033e-15,'m')                             #classical electron radius
        h   = 2*np.pi/d                                                #reciprocal wave vector length

        #Energies and angles corresponding to the constant parameter and its counterpart given by Bragg's law
        if self.scan_object.scantype == 'energy':
            theta_bragg = self.scan_object.constant
            energy_bragg = hc/(2*d*np.sin(theta_bragg.in_units('rad')))
        else:
            energy_bragg = self.scan_object.constant
            if not (hc/(2*d*energy_bragg)).in_units('1') > 1:
                theta_bragg = Quantity(np.arcsin((hc/(2*d*energy_bragg)).in_units('1')), 'rad')
            else:
                print('Given energy below the backscattering energy!')
                print('Setting theta to 90 deg.')
                theta_bragg = Quantity(90, 'deg')

        #set the scan vector
        if self.scan_object.scan[0] == 'automatic':

            F0, Fh, Fb = TakagiTaupin.calculate_structure_factors(crystal, 
                                                                  hkl, 
                                                                  energy_bragg, 
                                                                  debye_waller)

            #conversion factor from crystal structure factor to susceptibility
            cte = -(r_e * (hc/energy_bragg)**2/(np.pi * V)).in_units('1') 

            chi0 = np.conj(cte*F0)
            chih = np.conj(cte*Fh)
            chib = np.conj(cte*Fb)

            gamma0 = 1/np.sin((theta_bragg+phi).in_units('rad'))
            gammah = 1/np.sin((theta_bragg-phi).in_units('rad'))

            #Find the (rough) maximum and minimum of the deformation term
            if not displacement_jacobian == None:
                z = np.linspace(0,-self.crystal_object.thickness.in_units('um'),1000)
                x = -z*np.cos((theta_bragg+phi).in_units('rad'))/np.sin((theta_bragg+phi).in_units('rad'))

                sin_phi = np.sin(phi.in_units('rad'))
                cos_phi = np.cos(phi.in_units('rad'))
                sin_alphah = np.sin((theta_bragg-phi).in_units('rad'))
                cos_alphah = np.cos((theta_bragg-phi).in_units('rad'))

                u_jac = displacement_jacobian(x[0],z[0])
                h_um = h.in_units('um^-1')

                def_min = h_um*(  sin_phi*cos_alphah*u_jac[0][0]
                                + sin_phi*sin_alphah*u_jac[0][1]
                                + cos_phi*cos_alphah*u_jac[1][0] 
                                + cos_phi*sin_alphah*u_jac[1][1])            
                def_max = def_min

                for i in range(1,z.size):
                    u_jac = displacement_jacobian(x[i],z[i])

                    deform = h_um*(  sin_phi*cos_alphah*u_jac[0][0]
                                   + sin_phi*sin_alphah*u_jac[0][1]
                                   + cos_phi*cos_alphah*u_jac[1][0] 
                                   + cos_phi*sin_alphah*u_jac[1][1])            
                    if deform < def_min:
                        def_min = deform
                    if deform > def_max:
                        def_max = deform
            else:
                def_min = 0.0
                def_max = 0.0

            #expand the range by the backscatter Darwin width
            if np.sin(2*theta_bragg.in_units('rad') > np.sqrt(2*np.sqrt(np.abs(chih*chib)))):
                darwin_width_term = 2*np.sqrt(np.abs(chih*chib))/np.sin(2*theta_bragg.in_units('rad'))*h*np.cos(theta_bragg.in_units('rad'))
            else:
                darwin_width_term = np.sqrt(2*np.sqrt(np.abs(chih*chib)))*h*np.cos(theta_bragg.in_units('rad'))

            k_bragg = 2*np.pi*energy_bragg/hc 
            b_const_term = -0.5*k_bragg*(1 + gamma0/gammah)*np.real(chi0)

            beta_min = b_const_term - Quantity(def_max,'um^-1') - 2*darwin_width_term
            beta_max = b_const_term - Quantity(def_min,'um^-1') + 2*darwin_width_term

            #convert beta limits to scan vectors
            if self.scan_object.scantype == 'energy':
                energy_min = beta_min*energy_bragg/(h*np.sin(theta_bragg.in_units('rad')))
                energy_max = beta_max*energy_bragg/(h*np.sin(theta_bragg.in_units('rad')))

                print('Using automatically determined scan limits:')
                print('E min:', energy_min.in_units('meV'),'meV')
                print('E max:', energy_max.in_units('meV'),'meV')
                print('')

                scan = Quantity(np.linspace(energy_min.in_units('meV'),energy_max.in_units('meV'),self.scan_object.scan[1]),'meV')
                scan_steps = scan.value.size
                scan_shape = scan.value.shape
            else:
                theta_min  = Quantity(np.arcsin(np.sin(theta_bragg.in_units('rad'))+(beta_min/h).in_units('1')),'rad')-theta_bragg
                theta_max  = Quantity(np.arcsin(np.sin(theta_bragg.in_units('rad'))+(beta_max/h).in_units('1')),'rad')-theta_bragg

                print('Using automatically determined scan limits:')
                print('Theta min:', theta_min.in_units('urad'),'urad')
                print('Theta max:', theta_max.in_units('urad'),'urad')
                print('')

                scan = Quantity(np.linspace(theta_min.in_units('urad'),theta_max.in_units('urad'),self.scan_object.scan[1]),'urad')
                scan_steps = scan.value.size
                scan_shape = scan.value.shape
        else:
            scan = self.scan_object.scan[1]
            scan_steps = scan.value.size
            scan_shape = scan.value.shape

        if self.scan_object.scantype == 'energy':
            theta  = theta_bragg
            energy  = energy_bragg + scan
        else:
            energy = energy_bragg
            theta = theta_bragg + scan

        wavelength = hc/energy
        k = 2*np.pi/wavelength

        #Incidence and exit angles
        alpha0 = theta+phi
        alphah = theta-phi
        
        #Direction parameters
        gamma0 = np.ones(scan_shape)/np.sin(alpha0.in_units('rad'))
        gammah = np.ones(scan_shape)/np.sin(alphah.in_units('rad'))

        if np.mean(gammah) < 0:
            print('The direction of diffraction in to the crystal -> Laue case')
            geometry = 'laue'
        else:
            print('The direction of diffraction out of the crystal -> Bragg case')
            geometry = 'bragg'

        #Polarization factor
        if self.scan_object.polarization == 'sigma':
            C = 1;
            print('Solving for sigma-polarization')
        else:
            C = np.cos(2*theta.in_units('rad'));
            print('Solving for pi-polarization')

        #DEBUG
        print('Asymmetry angle : ', phi)
        print('Wavelength      : ', (hc/energy_bragg).in_units('nm'),'nm')
        print('Energy          : ', energy_bragg.in_units('keV'), 'keV')       
        print('Bragg angle     : ', theta_bragg.in_units('deg'),'deg')
        print('Incidence angle : ', (theta_bragg+phi).in_units('deg'),'deg')
        print('Exit angle      : ', (theta_bragg-phi).in_units('deg'),'deg')
        print('')


        #Compute susceptibilities
        
        F0, Fh, Fb = TakagiTaupin.calculate_structure_factors(crystal, 
                                                              hkl, 
                                                              energy, 
                                                              debye_waller)

        #conversion factor from crystal structure factor to susceptibility
        cte = -(r_e * (hc/energy_bragg)**2/(np.pi * V)).in_units('1') 

        chi0 = np.conj(cte*F0)
        chih = np.conj(cte*Fh)
        chib = np.conj(cte*Fb)
        
        #DEBUG
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
        c0 = 0.5*k*chi0*(gamma0+gammah)*np.ones(scan_shape)
        ch = 0.5*k*C*chih*gammah*np.ones(scan_shape)
        cb = 0.5*k*C*chib*gamma0*np.ones(scan_shape)

        #For solving Y = D0 
        g0 = 0.5*k*chi0*gamma0*np.ones(scan_shape)
        gb = 0.5*k*C*chib*gamma0*np.ones(scan_shape)

        #Deviation from the kinematical Bragg condition for unstrained crystal
        #To avoid catastrophic cancellation, the terms in the subtraction are
        #explicitly casted to 64 bit floats.
        beta_term1 = np.sin(theta.in_units('rad')).astype(np.float64)
        beta_term2 = wavelength.in_units('nm').astype(np.float64)
        beta_term3 = (2*d.in_units('nm')).astype(np.float64)
        
        beta = h*gammah*(beta_term1 - beta_term2/beta_term3).astype(np.float)

        #############
        #INTEGRATION#
        #############

        #Define ODEs and their Jacobians
        if geometry == 'bragg':
            print('Transmission in the Bragg case not implemented!')
            reflectivity = np.zeros(scan_shape)
            transmission = -np.ones(scan_shape)
        else:
            forward_diffraction = np.zeros(scan_shape)
            diffraction = np.zeros(scan_shape)

        #Fix the length scale to microns for solving
        c0   =   c0.in_units('um^-1'); ch = ch.in_units('um^-1'); cb = cb.in_units('um^-1')
        g0   =   g0.in_units('um^-1'); gb = gb.in_units('um^-1')
        beta = beta.in_units('um^-1'); h  =  h.in_units('um^-1')
        thickness = self.crystal_object.thickness.in_units('um')

        def integrate_single_scan_step(step):
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
                sin_phi = np.sin(phi.in_units('rad'))
                cos_phi = np.cos(phi.in_units('rad'))

                if self.scan_object.scantype == 'energy':
                    cot_alpha0 = np.cos(alpha0.in_units('rad'))/np.sin(alpha0.in_units('rad'))
                    sin_alphah = np.sin(alphah.in_units('rad'))
                    cos_alphah = np.cos(alphah.in_units('rad'))

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
                    cot_alpha0 = np.cos(alpha0.in_units('rad')[step])/np.sin(alpha0.in_units('rad')[step])
                    sin_alphah = np.sin(alphah.in_units('rad')[step])
                    cos_alphah = np.cos(alphah.in_units('rad')[step])

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
                #Non-bent crystal 
                def strain_term(z): 
                    return 0
            
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

            r.set_integrator('zvode',method='bdf',with_jacobian=True, 
                             min_step=self.scan_object.integration_step.in_units('um'),
                             max_step=thickness,nsteps=50000)
        
            #Update the solving process
            lock.acquire()
            steps_calculated.value = steps_calculated.value + 1
            sys.stdout.write('\rSolving...%0.1f%%' % (100*(steps_calculated.value)/scan_steps,))  
            sys.stdout.flush()
            lock.release()            

            if geometry == 'bragg':
                if not self.scan_object.start_depth == None:
                    start_depth = self.scan_object.start_depth.in_units('um')
                    if start_depth > 0 or start_depth < -thickness:
                        print('Warning! The given starting depth ' + str(start_depth) + 'um is outside the crystal!')                    
                    r.set_initial_value(0,start_depth)
                else:
                    r.set_initial_value(0,-thickness)
                res=r.integrate(0)     
                reflectivity = np.abs(res[0])**2*gamma0[step]/gammah[step] #gamma-part takes into account beam footprints
                transmission = -1 #Not implemented yet
                return reflectivity, transmission
            else:
                if not self.scan_object.start_depth == None:
                    print('Warning! The alternative start depth is negleted in the Laue case.')
                r.set_initial_value([0,1],0)
                res=r.integrate(-thickness)
                diffraction = np.abs(res[0]*res[1])**2
                forward_diffraction = np.abs(res[1])**2
                return diffraction, forward_diffraction

        n_cores = multiprocess.cpu_count()
    
        print('\nCalculating the TT-curve using ' + str(n_cores) + ' cores.')

        #Solve the equation
        sys.stdout.write('Solving...0%')
        sys.stdout.flush()
       
        def mp_init(l,v):
            global lock
            global steps_calculated
            lock = l
            steps_calculated = v

        pool = multiprocess.Pool(n_cores,initializer=mp_init, initargs=(multiprocess.Lock(), multiprocess.Value('i', 0)))
        output = np.array(pool.map(integrate_single_scan_step,range(scan_steps)))
        pool.close()
        pool.join()

        sys.stdout.write('\r\nDone.\n')
        sys.stdout.flush()

        if geometry == 'bragg':    
            reflectivity = output[:,0]
            transmission = output[:,1]

            self.solution = {'scan' : scan, 
                             'geometry' : 'bragg', 
                             'reflectivity' : reflectivity, 
                             'transmission': transmission
                            }

            return scan.value, reflectivity, transmission
        else:    
            diffraction = output[:,0]
            forward_diffraction = output[:,1]      

            self.solution = {'scan' : scan, 
                             'geometry' : 'laue', 
                             'diffraction' : diffraction, 
                             'forward_diffraction': forward_diffraction
                            }

            return scan.value, diffraction, forward_diffraction

    def plot(self):
        '''
        Plots the calculated solution
        '''

        if self.solution == None:
            print('No calculated Takagi-Taupin curves found! Call run() first!')
            return

        if self.solution['geometry'] == 'bragg':
            plt.plot(self.solution['scan'].value, self.solution['reflectivity'])
            if Quantity._type2str(self.solution['scan'].unit) == 'energy':
                plt.xlabel('Energy (' + Quantity._unit2str(self.solution['scan'].unit) + ')')
            else:
                plt.xlabel('Angle (' + Quantity._unit2str(self.solution['scan'].unit) + ')')
            plt.ylabel('Reflectivity')
        else:
            plt.plot(self.solution['scan'].value, self.solution['forward_diffraction'],label = 'Forward-diffraction')
            plt.plot(self.solution['scan'].value, self.solution['diffraction'],label = 'Diffraction')

            if Quantity._type2str(self.solution['scan'].unit) == 'energy':
                plt.xlabel('Energy (' + Quantity._unit2str(self.solution['scan'].unit) + ')')
            else:
                plt.xlabel('Angle (' + Quantity._unit2str(self.solution['scan'].unit) + ')')
            plt.ylabel('Intensity w.r.t incident')
            plt.legend()

        plt.show()

    def __str__(self):
        #TODO: Improve output presentation
        return   'CRYSTAL PARAMETERS\n'\
               + '------------------\n\n'\
               + str(self.crystal_object) + '\n\n'\
               + 'SCAN PARAMETERS\n'\
               + '---------------\n\n'\
               + str(self.scan_object)
