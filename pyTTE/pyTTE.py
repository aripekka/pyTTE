# -*- coding: utf-8 -*-

from __future__ import division, print_function
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import ode
from scipy.constants.codata import physical_constants

from .quantity import Quantity
from .crystal_vectors import crystal_vectors
from .elastic_tensors import elastic_matrices, rotate_elastic_matrix
from .deformation import isotropic_plate, anisotropic_plate
from .rotation_matrix import rotate_asymmetry, align_vector_with_z_axis

import xraylib

import multiprocess

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
            filepath     = path to the file with crystal parameters

            OR

            crystal      = string representation of the crystal in compliance with xraylib
            hkl          = list-like object of size 3 of the Miller indices of the reflection (ints or floats)
            thickness    = the thickness of the crystal wrapped in a Quantity instance e.g. Quantity(300,'um')

            (Optional)
            asymmetry         = clockwise-positive asymmetry angle wrapped in a Quantity instance.
                                0 deg for symmetric Bragg case (default), 90 deg for symmetric Laue
            in_plane_rotation = counterclockwise-positive rotation of the crystal directions about hkl-vector 
                                wrapped in a Quantity instance of type angle
                                OR
                                a crystal direction [q,r,s] corresponding to a direct space vector
                                R = q*a1 + r*a2 + s*a3 which will be rotated about the hkl vector so that its
                                component perpendicular to hkl (and the crystal as a whole with it) will be 
                                aligned with the y-axis. Will raise an error if R || hkl.
            debye_waller      = The Debye-Waller factor to account for thermal motion. Defaults to 1 (0 K).

            S                 = 6x6 compliance matrix wrapped in a Quantity instance. Overrides the default 
                                compliance matrix given by elastic_tensors and given E and nu, if given.

                                If an input file is used, the non-zero elements of the compliance matrix
                                in the upper triangle and on the diagonal should be given in the units GPa^-1 
                                (order doesn't matter). Any lower triangle inputs will be omitted as they are 
                                obtained symmetrically from the upper triangle. 

                                Example input: 
                                    S11  0.00723
                                    S22  0.00723
                                    S33  0.00723
                                    S12 -0.00214
                                    etc.

            E                 = Young's modulus for isotropic material in a Quantity instance. Overrides the 
                                default compliance matrix. Neglected if S is given
            nu                = Poisson's ratio for isotropic material. Overrides the default compliance matrix. 
                                Neglected if S is given.

            Rx, Ry            = Meridional and sagittal bending radii for toroidal bending wrapped in 
                                Quantity instances e.g. Quantity(1,'m'). If omitted, defaults to inf (no bending). 
                                Overridden by R.
            R                 = Bending radius for spherical bending wrapped in Quantity instance. Overrides Rx and Ry. 
        '''

        params = {}

        #set the default values for the optional parameters
        params['asymmetry'] = Quantity(0,'deg')
        params['in_plane_rotation'] = Quantity(0,'deg')
        params['debye_waller'] = 1.0

        params['S']  = None
        params['E']  = None
        params['nu'] = None

        params['Rx'] = None
        params['Ry'] = None

        if not filepath == None:
            #read file contents
            try:
                f = open(filepath,'r')    
                lines = f.readlines()
            except Exception as e:
                raise e
            finally:
                f.close()

            #Boolean to check if elements of the compliance matrix are given
            is_S_given = False
            S_matrix = np.zeros((6,6))

            #check and parse parameters
            for line in lines:
                if not line[0] == '#':  #skip comments
                    ls = line.split() 
                    if ls[0] == 'crystal' and len(ls) == 2:
                        params['crystal'] = ls[1]
                    elif ls[0] == 'hkl' and len(ls) == 4:
                        params['hkl'] = [int(ls[1]),int(ls[2]),int(ls[3])]
                    elif ls[0] == 'in_plane_rotation' and len(ls) == 4:
                        params['in_plane_rotation'] = [float(ls[1]),float(ls[2]),float(ls[3])]
                    elif ls[0] == 'in_plane_rotation' and len(ls) == 3:
                        params['in_plane_rotation'] = Quantity(float(ls[1]),ls[2])
                    elif ls[0] == 'thickness' and len(ls) == 3:
                        params['thickness'] = Quantity(float(ls[1]),ls[2])
                    elif ls[0] == 'asymmetry' and len(ls) == 3:
                        params['asymmetry'] = Quantity(float(ls[1]),ls[2])
                    elif ls[0] == 'debye_waller' and len(ls) == 2:
                        params['debye_waller'] = float(ls[1])
                    elif ls[0] == 'E' and len(ls) == 3:
                        params['E'] = Quantity(float(ls[1]),ls[2])
                    elif ls[0] == 'nu' and len(ls) == 2:
                        params['nu'] = float(ls[1])
                    elif ls[0][0] == 'S' and len(ls[0]) == 3 and len(ls) == 2:
                        is_S_given = True
                        i = int(ls[0][1])-1
                        j = int(ls[0][2])-1
                        if i > j:
                            print('Omitted the lower triangle element ' + ls[0] + '.')
                        else:
                            S_matrix[i,j] = float(ls[1])
                            S_matrix[j,i] = float(ls[1])
                    elif ls[0] == 'Rx' and len(ls) == 3:
                        params['Rx'] = Quantity(float(ls[1]),ls[2])
                    elif ls[0] == 'Ry' and len(ls) == 3:
                        params['Ry'] = Quantity(float(ls[1]),ls[2])
                    elif ls[0] == 'R' and len(ls) == 3:
                        params['Rx'] = Quantity(float(ls[1]),ls[2])
                        params['Ry'] = Quantity(float(ls[1]),ls[2])
                    else:
                        print('Skipped an invalid line in the file: ' + line)

            #Finalize the S matrix
            if is_S_given:
                params['S'] = Quantity(S_matrix,'GPa^-1') 

            #Check the presence of the required crystal inputs
            try:
                params['crystal']; params['hkl']; params['thickness']
            except:
                raise KeyError('At least one of the required keywords crystal, hkl, or thickness is missing!')           
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
            if 'in_plane_rotation' in kwargs.keys():
                params['in_plane_rotation'] = kwargs['in_plane_rotation']
            if 'debye_waller' in kwargs.keys():
                params['debye_waller'] = kwargs['debye_waller']
            if 'S' in kwargs.keys():
                params['S'] = kwargs['S']
            if 'E' in kwargs.keys():
                if 'nu' in kwargs.keys():
                    params['E'] = kwargs['E']
                    params['nu'] = kwargs['nu']
                else:
                    raise KeyError('Both E and nu required for isotropic material!')
            elif 'nu' in kwargs.keys():            
                raise KeyError('Both E and nu required for isotropic material!')

            if 'Rx' in kwargs.keys():
                params['Rx'] = kwargs['Rx']
            if 'Ry' in kwargs.keys():
                params['Ry'] = kwargs['Ry']
            if 'R' in kwargs.keys():  
                if 'Rx' in kwargs.keys() or 'Rx' in kwargs.keys():
                    print('Warning! Rx and/or Ry given but overridden by R.')
                params['Rx'] = kwargs['R']
                params['Ry'] = kwargs['R']

        #used to prevent recalculation of the deformation in parameter set functions during init
        self._initialized = False

        #determines the length scale in which the position coordinate to the jacobian are given 
        self._jacobian_length_unit = 'um'

        self.set_crystal(params['crystal'])
        self.set_reflection(params['hkl'])
        self.set_thickness(params['thickness'])
        self.set_asymmetry(params['asymmetry'])
        self.set_in_plane_rotation(params['in_plane_rotation'])
        self.set_debye_waller(params['debye_waller'])

        if not params['S'] == None:
            self.set_elastic_constants(S = params['S'])
            if 'E' in params.keys() or 'nu' in params.keys():
                print('Warning! Isotropic E and/or nu given but overridden by the compliance matrix S.')
        elif (not params['E'] == None) and (not params['nu'] == None):
            self.set_elastic_constants(E = params['E'], nu = params['nu'])
        else:
            self.set_elastic_constants()

        self.set_bending_radii(params['Rx'], params['Ry'])

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
            self.thickness = thickness.copy()
        else:
            raise ValueError('Thickness has to be a Quantity instance of type length!')

        #skip this if the function is used as a part of initialization
        if self._initialized:
            self.update_rotations_and_deformation()

    def set_asymmetry(self, asymmetry):
        '''
        Set the asymmetry angle.

        Input:
            asymmetry = clockwise-positive asymmetry angle wrapped in a Quantity instance 0 
                        for symmetric Bragg case (default), 90 deg for symmetric Laue
        '''

        if isinstance(asymmetry,Quantity) and asymmetry.type() == 'angle':
            self.asymmetry = asymmetry.copy()
        else:
            raise ValueError('Asymmetry angle has to be a Quantity instance of type angle!')

        #skip this if the function is used as a part of initialization
        if self._initialized:
            self.update_rotations_and_deformation()

    def set_in_plane_rotation(self, in_plane_rotation):
        '''
        Set the in-plane rotation angle.

        Input:
            in_plane_rotation = counterclockwise-positive rotation of the crystal directions about hkl-vector 
                                wrapped in a Quantity instance of type angle
                                OR
                                a crystal direction [q,r,s] corresponding to a direct space vector
                                R = q*a1 + r*a2 + s*a3 which will be rotated about the hkl vector so that its
                                component perpendicular to hkl (and the crystal as a whole with it) will be 
                                aligned with the y-axis. Will raise an error if R || hkl.
        '''

        if isinstance(in_plane_rotation, Quantity) and in_plane_rotation.type() == 'angle':
            self.in_plane_rotation = in_plane_rotation.copy()
        elif len(in_plane_rotation) == 3:
            #Check the list entry types
            for i in in_plane_rotation:
                if not type(i) in [type(1),type(1.0)]:
                    raise ValueError('In-plane rotation angle has to be a Quantity instance of type angle OR a list of floats size 3!')

            #calculate the given crystal direction in the direct space
            r = in_plane_rotation[0]*self.direct_primitives[:,0] +\
                in_plane_rotation[1]*self.direct_primitives[:,1] +\
                in_plane_rotation[2]*self.direct_primitives[:,2]

            #calculate reciprocal vector of the diffraction hkl
            h = self.hkl[0]*self.reciprocal_primitives[:,0] +\
                self.hkl[1]*self.reciprocal_primitives[:,1] +\
                self.hkl[2]*self.reciprocal_primitives[:,2]

            #check the relative angle of r and h
            if abs(np.dot(r,h) - np.sqrt(np.dot(r,r)*np.dot(h,h))) < np.finfo(type(1.0)).eps:
                raise ValueError('in_plane_rotation can not be parallel to the reciprocal diffraction vector!')

            #hkl||z alignment
            R = align_vector_with_z_axis(h)

            #rotate r to a coordinate system where z||hkl            
            r_rot = np.dot(R,r)

            #Calculate the inclination between r_rot and the xy-plane
            incl = np.arctan2(r_rot[2],np.sqrt(r_rot[0]**2 + r_rot[1]**2))

            print('Deviation of the given in_plane_rotation direction from the rotation plane: ' + str(np.degrees(incl)) + ' deg.')
            
            #The angle between the direction vector projected to the new xy-plane and the y-axis
            rotation_angle = np.arctan2(r_rot[0],r_rot[1])

            self.in_plane_rotation = Quantity(np.degrees(rotation_angle),'deg')
        else:
            raise ValueError('In-plane rotation angle has to be a Quantity instance of type angle OR a list of floats size 3!')

        #skip this if the function is used as a part of initialization
        if self._initialized:
            self.update_rotations_and_deformation()

    def set_debye_waller(self, debye_waller):
        '''
        Set the Debye-Waller factor.

        Input:
            debye_waller = A float or integer in the range [0,1]
        '''

        if debye_waller >= 0 and debye_waller <= 1:
            self.debye_waller = debye_waller
        else:
            raise ValueError('Debye-Waller factor has to be a float or integer in range [0,1]!')

        #skip this if the function is used as a part of initialization
        if self._initialized:
            self.update_rotations_and_deformation()

    def set_elastic_constants(self, **kwargs):
        '''
        Set either the compliance matrix (fully anisotropic) or Young's modulus and Poisson ratio (isotropic).

        Input:
            None for the compliance matrix in the internal database

            OR

            S = 6x6 compliance matrix wrapped in a instance of Quantity of type pressure^-1
            
            OR
            
            E  = Young's modulus in a Quantity instance of type pressure
            nu = Poisson's ratio (float or int) 
        '''

        if 'S' in kwargs.keys():
            if isinstance(kwargs['S'], Quantity) and kwargs['S'].type() == 'pressure^-1':
                if kwargs['S'].value.shape == (6,6):
                    self.isotropy = 'anisotropic'
                    self.S0 = kwargs['S'].copy()
                else:
                    raise ValueError('Shape of S has to be (6,6)!')
            else:
                raise ValueError('S has to be an instance of Quantity of type pressure^-1!')
        elif 'E' in kwargs.keys() and 'nu' in kwargs.keys():
            if isinstance(kwargs['E'], Quantity) and kwargs['E'].type() == 'pressure':
                if type(kwargs['nu']) in [int, float]:
                    self.isotropy = 'isotropic'
                    self.E  = kwargs['E'].copy()
                    self.nu = kwargs['nu']
                else:
                    raise ValueError('nu has to be float or int!')
            else:
                raise ValueError('E has to be an instance of Quantity of type pressure!')
        else:
            self.isotropy = 'anisotropic'
            self.S0 = Quantity(0.01*elastic_matrices(self.crystal_data['name'])[1],'GPa^-1')
            
        #skip this if the function is used as a part of initialization
        if self._initialized:
            self.update_rotations_and_deformation()

    def set_bending_radii(self, Rx, Ry):
        '''
        Sets the meridional and sagittal bending radii.

        Input:
            Rx, Ry = Meridional and sagittal bending radii wrapped in Quantity instances of type length.
                     Alternatively can be float('inf'), 'inf' or None
        '''

        if isinstance(Rx, Quantity) and Rx.type() == 'length':
            self.Rx = Rx.copy()
        elif Rx == None or Rx == 'inf' or Rx == float('inf'):
            self.Rx = Quantity(float('inf'),'m')
        else:
            raise ValueError('Rx has to be an instance of Quantity of type length, inf, or None!')
        if isinstance(Ry, Quantity) and Ry.type() == 'length':
            self.Ry = Ry.copy()
        elif Ry == None or Ry == 'inf' or Ry == float('inf'):
            self.Ry = Quantity(float('inf'),'m')
        else:
            raise ValueError('Ry has to be an instance of Quantity of type length, inf, or None!')
            
        #skip this if the function is used as a part of initialization
        if self._initialized:
            self.update_rotations_and_deformation()

    def update_rotations_and_deformation(self):
        '''
        Applies the in-plane and asymmetry rotations to the elastic matrix (for anisotropic crystal) 
        and calculates the Jacobian of the deformation field based on the elastic parameters and the
        bending radii.
        '''

        #Apply rotations of the crystal to the elastic matrix
        if self.isotropy == 'anisotropic':
            #TODO: inplane_rotation

            #calculate reciprocal vector of the diffraction hkl
            hkl = self.hkl[0]*self.reciprocal_primitives[:,0] +\
                  self.hkl[1]*self.reciprocal_primitives[:,1] +\
                  self.hkl[2]*self.reciprocal_primitives[:,2]

            #hkl||z alignment
            R1 = align_vector_with_z_axis(hkl)

            #asymmetry alignment
            R2 = rotate_asymmetry(self.asymmetry.in_units('deg'))

            Rmatrix = np.dot(R2,R1)            

            self.S = Quantity(rotate_elastic_matrix(self.S0.value, 'S', Rmatrix), Quantity._unit2str(self.S0.unit))
            self.crystal_directions = np.dot(Rmatrix.T,self.direct_primitives)

            print(self.crystal_directions)
        
        #calculate the depth-dependent deformation jacobian
        if self.Rx.value == float('inf') and self.Ry.value == float('inf'):
            self.displacement_jacobian = None
        elif self.isotropy == 'anisotropic':
            self.displacement_jacobian = anisotropic_plate(self.Rx.in_units(self._jacobian_length_unit),
                                                           self.Ry.in_units(self._jacobian_length_unit),
                                                           self.S.in_units('GPa^-1'),
                                                           self.thickness.in_units(self._jacobian_length_unit))

        else:
            self.displacement_jacobian = isotropic_plate(self.Rx.in_units(self._jacobian_length_unit),
                                                         self.Ry.in_units(self._jacobian_length_unit),
                                                         self.nu,
                                                         self.thickness.in_units(self._jacobian_length_unit))

    def __str__(self):
        #TODO: Improve output presentation
        if self.isotropy == 'anisotropic':
            elastic_str = 'S (with rotations applied):\n' + str(self.S)
        else:
            elastic_str = 'E: ' + str(self.E) + '\nnu: '+ str(self.nu) 

        return 'Crystal: ' + self.crystal_data['name'] + '\n' +\
               'Crystallographic parameters:\n' +\
               '    a = ' + str(self.crystal_data['a']*0.1)[:8] + ' nm,  b = ' + str(self.crystal_data['b']*0.1)[:8] + ' nm,  c = ' + str(self.crystal_data['c']*0.1)[:8] + ' nm\n'+\
               '    alpha = ' + str(self.crystal_data['alpha']) + ' deg,  beta = ' + str(self.crystal_data['beta']) + ' nm,  gamma = ' + str(self.crystal_data['gamma']) + ' deg\n'+\
               'Direct primitive vectors (columns, in nm):\n'+ np.array2string(0.1*self.direct_primitives,precision=4,suppress_small=True)+'\n'+\
               'Reciprocal primitive vectors (columns, in 1/nm):\n'+ np.array2string(10*self.reciprocal_primitives,precision=4,suppress_small=True)+'\n'+\
               'Reflection: '+str(self.hkl)+'\n'+\
               'Asymmetry angle: ' + str(self.asymmetry)+'\n'+\
               'In-plane rotation angle: ' + str(self.in_plane_rotation)+'\n'+\
               'Thickness: ' + str(self.thickness)+'\n'+\
               'Meridional bending radius: ' + str(self.Rx) +'\n'+\
               'Sagittal bending radius: ' + str(self.Ry) +'\n'+\
               'Elastic material: ' + str(self.isotropy) +'\n' + elastic_str        
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
            #read file contents
            try:
                f = open(filepath,'r')    
                lines = f.readlines()
            except Exception as e:
                raise e
            finally:
                f.close()

            #check and parse parameters
            for line in lines:
                if not line[0] == '#':  #skip comments
                    ls = line.split() 
                    if ls[0] == 'constant' and len(ls) == 3:
                        params['constant'] = Quantity(float(ls[1]),ls[2])
                    elif ls[0] == 'scan' and len(ls) == 2:
                        params['scan'] = int(ls[1])
                    elif ls[0] == 'scan' and len(ls) == 5:
                        params['scan'] = Quantity(np.linspace(float(ls[1]),float(ls[2]),int(ls[3])),ls[4])
                    elif ls[0] == 'polarization' and len(ls) == 2:
                        params['polarization'] = ls[1]
                    else:
                        print('Skipped an invalid line in the file: ' + line)
 
            #Check the presence of the required crystal inputs
            try:
                params['constant']; params['scan']; params['polarization']
            except:
                raise KeyError('At least one of the required keywords constant, scan, or polarization is missing!')

        else:
            #Check the presence of the required crystal inputs
            try:
                params['constant']     = kwargs['constant']
                params['scan']         = kwargs['scan']
                params['polarization'] = kwargs['polarization']
            except:
                raise KeyError('At least one of the required keywords constant, scan, or polarization is missing!')                

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
            self.constant = constant.copy()
        else:
            raise ValueError('constant has to be an instance of Quantity of type energy or angle!')

        if isinstance(scan, Quantity) and scan.type() == self.scantype:
            self.scan = ('manual', scan.copy())
        elif type(scan) == type(1) and scan > 0:
            self.scan = ('automatic',scan)
        else:
            raise ValueError('scan has to be either a Quantity of type energy (for angle constant) or angle (for energy constant) or a non-negative integer!')

    def __str__(self):
        #TODO: Improve output presentation
        return 'Scan type: ' + self.scantype + '\n' +\
               'Scan constant: ' + str(self.constant) +'\n' +\
               'Polarization: ' + self.polarization  +'\n' +\
               'Scan points: ' + str(self.scan[0])  +'\n'

class TakagiTaupin:

    def __init__(self, TTcrystal_object = None, TTscan_object = None):
        self.crystal_object = None
        self.scan_object = None
        self.solution = None

        self.set_crystal(TTcrystal_object)
        self.set_scan(TTscan_object)

    def set_crystal(self, TTcrystal_object):
        if isinstance(TTcrystal_object, TTcrystal): 
            self.crystal_object = TTcrystal_object
        elif type(TTcrystal_object) == str:
            try:
                self.crystal_object = TTcrystal(TTcrystal_object)
            except Exception as e:
                print(e)
                print('Error initializing TTcrystal from file! Crystal not set.')
        else:
            print('ERROR! Not an instance of TTcrystal or None! Crystal not set.')

    def set_scan(self, TTscan_object):
        if isinstance(TTscan_object, TTscan):
            self.scan_object = TTscan_object
        elif type(TTscan_object) == str:
            try:
                self.scan_object = TTscan(TTscan_object)
            except Exception as e:
                print(e)
                print('Error initializing TTscan from file! Scan not set.')
        else:
            print('ERROR! Not an instance of TTscan or None! Scan not set.')

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

        d   = Quantity(xraylib.Crystal_dSpacing(crystal,*hkl),'A')    #spacing of Bragg planes
        V   = Quantity(crystal['volume'],'A^3')                       #volume of unit cell
        r_e = Quantity(2.81794033e-15,'m')                            #classical electron radius
        h   = 2*np.pi/d                                               #reciprocal wave vector length

        if self.scan_object.scan[0] == 'automatic':
            print('AUTOMATIC LIMITS NOT IMPLEMENTED YET!')
            print('Function terminated.')
            return None
        else:
            scan = self.scan_object.scan[1]
            scan_steps = scan.value.size
            scan_shape = scan.value.shape


        if self.scan_object.scantype == 'energy':
            theta0 = self.scan_object.constant
            theta  = theta0

            energy0 = hc/(2*d*np.sin(theta.in_units('rad')))
            energy  = energy0 + scan

        else:
            energy0 = self.scan_object.constant
            energy = energy0

            if not hc/(2*d*energy) > 1:
                theta0 = Quantity(np.arcsin((hc/(2*d*energy)).in_units('1')), 'rad')
            else:
                print('Given energy below the backscattering energy!')
                print('Setting theta to 90 deg.')
                theta0 = Quantity(90, 'deg')

            theta = theta0 + scan

        wavelength = hc/energy
        k = 2*np.pi/wavelength

        #DEBUG:
        print('hc',hc)
        print('d',d)
        print('V',V)
        print('r_e',r_e)
        print('h',h)

        print('theta',theta)
        print('energy',energy.in_units('keV'))

        print('wavelength',wavelength)
        print('k',k)

        #asymmetry angle
        phi = self.crystal_object.asymmetry

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
        print('Wavelength      : ', hc/energy0)
        print('Energy          : ', energy0.in_units('keV'), ' keV ')       
        print('Bragg angle     : ', theta0)
        print('Incidence angle : ', theta0+phi)
        print('Exit angle      : ', theta0-phi)
        print('')

        #Compute susceptibilities
        debye_waller = self.crystal_object.debye_waller

        if self.scan_object.scantype == 'energy':
            F0 = np.zeros(scan_shape,dtype=np.complex)
            Fh = np.zeros(scan_shape,dtype=np.complex)
            Fb = np.zeros(scan_shape,dtype=np.complex)

            for ii in range(scan_steps):    
                F0[ii] = xraylib.Crystal_F_H_StructureFactor(crystal, energy.in_units('keV')[ii], 0, 0, 0, debye_waller, 1.0)
                Fh[ii] = xraylib.Crystal_F_H_StructureFactor(crystal, energy.in_units('keV')[ii],  hkl[0],  hkl[1],  hkl[2], debye_waller, 1.0)
                Fb[ii] = xraylib.Crystal_F_H_StructureFactor(crystal, energy.in_units('keV')[ii], -hkl[0], -hkl[1], -hkl[2], debye_waller, 1.0)
        else:
            F0 = xraylib.Crystal_F_H_StructureFactor(crystal, energy0.in_units('keV'), 0, 0, 0, debye_waller, 1.0)
            Fh = xraylib.Crystal_F_H_StructureFactor(crystal, energy0.in_units('keV'),  hkl[0],  hkl[1],  hkl[2], debye_waller, 1.0)
            Fb = xraylib.Crystal_F_H_StructureFactor(crystal, energy0.in_units('keV'), -hkl[0], -hkl[1], -hkl[2], debye_waller, 1.0)

        cte = -(r_e * wavelength*wavelength/(np.pi * V)).in_units('1')
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

        #deviation from the kinematical Bragg condition for unstrained crystal
        beta = h*gammah*(np.sin(theta.in_units('rad'))-(wavelength/(2*d)).in_units('1'))

        print(c0)
        print(beta)

        displacement_jacobian = self.crystal_object.displacement_jacobian

        print(displacement_jacobian)
        print(displacement_jacobian(0,0))
        print(displacement_jacobian(0,-self.crystal_object.thickness.in_units('um')))


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
        c0 = c0.in_units('um^-1'); ch = ch.in_units('um^-1'); cb = cb.in_units('um^-1')
        g0 = g0.in_units('um^-1'); gb = gb.in_units('um^-1')
        beta = beta.in_units('um^-1'); h = h.in_units('um^-1')
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

            r.set_integrator('zvode',method='bdf',with_jacobian=True, min_step=1e-11,max_step=thickness,nsteps=50000)
        
            lock.acquire()
            steps_calculated.value = steps_calculated.value + 1
            sys.stdout.write('\rSolving...%0.1f%%' % (100*(steps_calculated.value)/scan_steps,))  
            sys.stdout.flush()
            lock.release()            


            if geometry == 'bragg':
                r.set_initial_value(0,-thickness)
                res=r.integrate(0)     
                reflectivity = np.abs(res[0])**2*gamma0[step]/gammah[step] #gamma-part takes into account beam footprints
                transmission = -1 #Not implemented yet
                return reflectivity, transmission
            else:
                r.set_initial_value([0,1],0)
                res=r.integrate(-thickness)
                diffraction = np.abs(res[0]*res[1])**2
                forward_diffraction = np.abs(res[1])**2
                return diffraction, forward_diffraction

        n_cores = multiprocess.cpu_count()
        print('Using ' + str(n_cores) + ' cores.')

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
        return str(self.crystal_object) +'\n'+ str(self.scan_object)


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

