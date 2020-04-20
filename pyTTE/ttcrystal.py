# -*- coding: utf-8 -*-

from __future__ import division, print_function
from .quantity import Quantity
from .crystal_vectors import crystal_vectors
from .elastic_tensors import elastic_matrices, rotate_elastic_matrix
from .deformation import isotropic_plate, anisotropic_plate_fixed_shape, anisotropic_plate_fixed_torques
from .rotation_matrix import rotate_asymmetry, align_vector_with_z_axis, inplane_rotation
import numpy as np
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
                                
                                Note that S is supposed to be in the Cartesian coordinate system aligned
                                with the conventional unit vectors before any rotations i.e. x || a_1 
                                and a_2 is in the xy-plane. For rectagular systems this means that the 
                                Cartesian basis is aligned with the unit vectors. 

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
                                Overridden by R. The other one can be set to None if isotropic model is used or 
                                if fix_to_axes = 'torque' when it is determined by the anticlastic bending.
            R                 = Bending radius for spherical bending wrapped in Quantity instance. Overrides Rx and Ry.
            fix_to_axes       = Used to determine the anisotropic bending model used. If 'torques' then the plate is
                                bent by two orthogonal torques acting about x- and y-axes, if 'shape' then the main 
                                axes of curvature are assumed to be along x and y (and given by Rx and Ry).
        '''

        params = {}

        if filepath is not None:
            
            #####################################
            #Read crystal parameters from a file#
            #####################################

            #Overwrite possible kwargs 
            kwargs = {}

            with open(filepath,'r') as f:
                lines = f.readlines()

            #Boolean to check if elements of the compliance matrix are given
            is_S_given = False
            S_matrix = np.zeros((6,6))

            #check and parse parameters
            for line in lines:
                if not line[0] == '#':  #skip comment lines
                    ls = line.split() 
                    if ls[0] == 'crystal' and len(ls) == 2:
                        kwargs['crystal'] = ls[1]
                    elif ls[0] == 'hkl' and len(ls) == 4:
                        kwargs['hkl'] = [int(ls[1]),int(ls[2]),int(ls[3])]
                    elif ls[0] == 'in_plane_rotation':
                        if len(ls) == 4:
                            kwargs['in_plane_rotation'] = [float(ls[1]),float(ls[2]),float(ls[3])]
                        elif  len(ls) == 3:
                            kwargs['in_plane_rotation'] = Quantity(float(ls[1]),ls[2])
                        else:
                            print('Skipped an invalid line in the file: ' + line)
                    elif ls[0] == 'debye_waller' and len(ls) == 2:
                        kwargs['debye_waller'] = float(ls[1])
                    elif ls[0] in ['thickness', 'asymmetry', 'E'] and len(ls) == 3:
                        kwargs[ls[0]] = Quantity(float(ls[1]),ls[2])
                    elif ls[0] == 'nu' and len(ls) == 2:
                        kwargs['nu'] = float(ls[1])
                    elif ls[0][0] == 'S' and len(ls[0]) == 3 and len(ls) == 2:
                        is_S_given = True
                        i = int(ls[0][1])-1
                        j = int(ls[0][2])-1
                        if i > j:
                            print('Omitted the lower triangle element ' + ls[0] + '.')
                        else:
                            S_matrix[i,j] = float(ls[1])
                            S_matrix[j,i] = float(ls[1])
                    elif ls[0] in ['Rx', 'Ry']:
                        if len(ls) == 3:
                            kwargs[ls[0]] = Quantity(float(ls[1]),ls[2])
                        elif len(ls) == 2:
                            kwargs[ls[0]] = ls[1]
                        else:
                            print('Skipped an invalid line in the file: ' + line)                           
                    elif ls[0] == 'R' and len(ls) == 3:                        
                        kwargs['Rx'] = Quantity(float(ls[1]),ls[2])
                        kwargs['Ry'] = Quantity(float(ls[1]),ls[2])
                    elif ls[0] == 'fix_to_axes' and len(ls) == 2:
                        kwargs['fix_to_axes'] = ls[1]                        
                    else:
                        print('Skipped an invalid line in the file: ' + line)

            if is_S_given:
                #Finalize the S matrix
                kwargs['S'] = Quantity(S_matrix,'GPa^-1') 

        ###################################################
        #Check the presence of the required crystal inputs#
        ###################################################
        
        try:
            params['crystal']   = kwargs['crystal']
            params['hkl']       = kwargs['hkl']
            params['thickness'] = kwargs['thickness']
        except:
            raise KeyError('At least one of the required keywords crystal, hkl, or thickness is missing!')

        #Optional keywords       
        for k in ['asymmetry','in_plane_rotation']:
            params[k] = kwargs.get(k, Quantity(0,'deg'))

        params['debye_waller'] = kwargs.get('debye_waller', 1.0)

        for k in ['S','E','nu']:
            params[k] = kwargs.get(k, None)

        #Check that if either E or nu is given, then the other one is also
        if (params['E'] is not None) ^ (params['nu'] is not None):
            raise KeyError('Both E and nu required for isotropic material!')

        params['Rx'] = kwargs.get('Rx', 'inf')
        params['Ry'] = kwargs.get('Ry', 'inf')

        if 'R' in kwargs.keys():  
            if 'Rx' in kwargs.keys() or 'Rx' in kwargs.keys():
                print('Warning! Rx and/or Ry given but overridden by R.')
            params['Rx'] = kwargs['R']
            params['Ry'] = kwargs['R']

        params['fix_to_axes'] = kwargs.get('fix_to_axes', 'shape')

        ###########################################
        #Initialize with the read/given parameters#
        ###########################################

        #determines the length scale in which the position coordinate to the jacobian are given
        self._jacobian_length_unit = 'um'

        self.set_crystal(params['crystal'], skip_update = True)
        self.set_reflection(params['hkl'], skip_update = True)
        self.set_thickness(params['thickness'], skip_update = True)
        self.set_asymmetry(params['asymmetry'], skip_update = True)
        self.set_in_plane_rotation(params['in_plane_rotation'], skip_update = True)
        self.set_debye_waller(params['debye_waller'], skip_update = True)

        if params['S'] is not None:
            self.set_elastic_constants(S = params['S'], skip_update = True)
            if 'E' in params.keys() or 'nu' in params.keys():
                print('Warning! Isotropic E and/or nu given but overridden by the compliance matrix S.')
        elif (params['E'] is not None) and (params['nu'] is not None):
            self.set_elastic_constants(E = params['E'], nu = params['nu'], skip_update = True)
        else:
            self.set_elastic_constants(skip_update = True)

        self.set_fix_to_axes(params['fix_to_axes'], skip_update = True)
        self.set_deformation(jacobian = None, skip_update = True)
        self.set_bending_radii(params['Rx'], params['Ry'], skip_update = True)

        self.update_rotations_and_deformation()

    def set_crystal(self, crystal_str, skip_update = False):
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
        if not skip_update:
            self.update_rotations_and_deformation()

    def set_reflection(self, hkl, skip_update = False):
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
        if not skip_update:
            self.update_rotations_and_deformation()

    def set_thickness(self, thickness, skip_update = False):
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
        if not skip_update:
            self.update_rotations_and_deformation()

    def set_asymmetry(self, asymmetry, skip_update = False):
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
        if not skip_update:
            self.update_rotations_and_deformation()

    def set_in_plane_rotation(self, in_plane_rotation, skip_update = False):
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
        elif type(in_plane_rotation) in [type([]),type((1,)),type(np.array([]))] and len(in_plane_rotation) == 3:
                     
            #Check the list entry types
            for i in in_plane_rotation:
                if not np.isreal(i):
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
        if not skip_update:
            self.update_rotations_and_deformation()

    def set_debye_waller(self, debye_waller, skip_update = False):
        '''
        Set the Debye-Waller factor.

        Input:
            debye_waller = A float or integer in the range [0,1]
        '''

        if not isinstance(debye_waller,Quantity) and np.array(debye_waller).size == 1 \
        and np.real(debye_waller) and debye_waller >= 0 and debye_waller <= 1:
            self.debye_waller = debye_waller
        else:
            raise ValueError('Debye-Waller factor has to be a float or integer in range [0,1]!')

        #skip this if the function is used as a part of initialization
        if not skip_update:
            self.update_rotations_and_deformation()

    def set_elastic_constants(self, S = None, E = None, nu = None, skip_update = False):
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

        if (E is not None) and (nu is not None):
            if isinstance(E, Quantity) and E.type() == 'pressure':
                if type(nu) in [int, float]:
                    self.isotropy = 'isotropic'
                    self.E  = E.copy()
                    self.nu = nu
                else:
                    raise ValueError('nu has to be float or int!')
            else:
                raise ValueError('E has to be an instance of Quantity of type pressure!')
        elif S is not None:
            if isinstance(S, Quantity) and S.type() == 'pressure^-1':
                if S.value.shape == (6,6):
                    self.isotropy = 'anisotropic'
                    self.S0 = S.copy()
                else:
                    raise ValueError('Shape of S has to be (6,6)!')
            else:
                raise ValueError('S has to be an instance of Quantity of type pressure^-1!')
        else:
            self.isotropy = 'anisotropic'
            self.S0 = Quantity(0.01*elastic_matrices(self.crystal_data['name'])[1],'GPa^-1')
            
        #skip this if the function is used as a part of initialization
        if not skip_update:
            self.update_rotations_and_deformation()

    def set_fix_to_axes(self, fix_to_axes, skip_update = False):
        '''
        Sets the deformation field.

        Input:
            fix_to_axes = Determines the anisotropic bending model used. If 
                          'torques' then the plate is bent by two orthogonal 
                          torques acting about x- and y-axes, if 'shape' then 
                          the main axes of curvature are assumed to be along 
                          x and y (and given by Rx and Ry).
        '''

        if fix_to_axes in ['shape', 'torques']:
            self.fix_to_axes = fix_to_axes
        else:
            raise ValueError("The allowed values for fix_to_axes are 'torques' and 'shape'!" )
                        
        #skip this if the function is used as a part of initialization
        if not skip_update:
            if self.deformation_model[0] == 'custom':                
                self.set_deformation(jacobian = self.deformation_model[1], skip_update = True)
            else:
                self.set_deformation(jacobian = None, skip_update = True)
            self.set_bending_radii(self.Rx, self.Ry, skip_update = True)
            self.update_rotations_and_deformation()

    def set_bending_radii(self, Rx, Ry, skip_update = False):
        '''
        Sets the meridional and sagittal bending radii.

        Input:
            Rx, Ry = Meridional and sagittal bending radii wrapped in Quantity 
                     instances of type length. Alternatively can be float('inf'), 
                     'inf', or None. If self.deformation_model == ['anisotropic',
                     'fixed_shape'], None is interpreted as 'inf'.
        '''

        if isinstance(Rx, Quantity) and Rx.type() == 'length':
            self.Rx = Rx.copy()
        elif Rx == 'inf' or Rx == float('inf'):
            self.Rx = Quantity(float('inf'),'m')
        elif Rx is None:
            if self.deformation_model == ['anisotropic','fixed_shape']:
                self.Rx = Quantity(float('inf'),'m')
            else:
                self.Rx = None
        else:
            raise ValueError('Rx has to be an instance of Quantity of type length, inf, or None!')
        if isinstance(Ry, Quantity) and Ry.type() == 'length':
            self.Ry = Ry.copy()
        elif Ry == 'inf' or Ry == float('inf'):
            self.Ry = Quantity(float('inf'),'m')
        elif Ry is None:
            if self.deformation_model == ['anisotropic','fixed_shape']:
                self.Ry = Quantity(float('inf'),'m')
            else:
                self.Ry = None
        else:
            raise ValueError('Ry has to be an instance of Quantity of type length, inf, or None!')
            
        #skip this if the function is used as a part of initialization
        if not skip_update:
            self.update_rotations_and_deformation()

    def set_deformation(self, jacobian = None, skip_update = False):
        '''
        Sets the deformation field.

        Input:
            jacobian = function returning the partial derivatives of the 
                       displacement vector u as a function of (x,z). If None,
                       the deformation model will be determined from 
                       self.isotropy, and self.fix_to_axes
        '''

        if jacobian is not None:
            #test the given jacobian
            ujac = jacobian(0,0)

            if len(ujac) == 2:
               if len(ujac[0]) == 2 and len(ujac[1]) == 2:
                   self.deformation_model = ['custom', jacobian]
               else:
                   raise ValueError('Output of jacobian has to be 2x2 list or array!')
            else:
                raise ValueError('Output of jacobian has to be 2x2 list or array!')
        else:
            if self.isotropy == 'isotropic':
                self.deformation_model = ['isotropic']
            else:
                if self.fix_to_axes == 'shape':
                    self.deformation_model = ['anisotropic', 'fixed_shape']                
                else:
                    self.deformation_model = ['anisotropic', 'fixed_torques']                            
            
        #skip this if the function is used as a part of initialization
        if not skip_update:
            self.set_bending_radii(self.Rx, self.Ry, skip_update = True)
            self.update_rotations_and_deformation()

    def update_rotations_and_deformation(self):
        '''
        Applies the in-plane and asymmetry rotations to the elastic matrix (for anisotropic crystal) 
        and calculates the Jacobian of the deformation field based on the elastic parameters and the
        bending radii.
        '''

        #calculate reciprocal vector of the diffraction hkl
        hkl = self.hkl[0]*self.reciprocal_primitives[:,0] +\
              self.hkl[1]*self.reciprocal_primitives[:,1] +\
              self.hkl[2]*self.reciprocal_primitives[:,2]

        #hkl||z alignment
        R1 = align_vector_with_z_axis(hkl)
        
        R2 = inplane_rotation(self.in_plane_rotation.in_units('deg'))

        #asymmetry alignment
        R3 = rotate_asymmetry(self.asymmetry.in_units('deg'))

        Rmatrix = np.dot(R3,np.dot(R2,R1))

        #rotate the primitive vectors
        dir_prim_rot = np.dot(Rmatrix,self.direct_primitives)
        
        #calculate the basis transform matrix from cartesian to crystal direction 
        #indices, whose columns are equal to crystal directions along main axes 
        self.crystal_directions = np.linalg.inv(dir_prim_rot)

        #Apply rotations of the crystal to the elastic matrix
        if self.deformation_model[0] == 'anisotropic':
            self.S = Quantity(rotate_elastic_matrix(self.S0.value, 'S', Rmatrix), 
                              Quantity._unit2str(self.S0.unit))
        
        #calculate the depth-dependent deformation jacobian
        if self.deformation_model[0] == 'custom':
            self.displacement_jacobian = self.deformation_model[1]
        elif self.Rx.value == float('inf') and self.Ry.value == float('inf'):
            self.displacement_jacobian = None
        else:
            if self.Rx is not None:
                Rx = self.Rx.in_units(self._jacobian_length_unit)
            else:
                Rx = None
            if self.Ry is not None:
                Ry = self.Ry.in_units(self._jacobian_length_unit)
            else:
                Ry = None

            if self.deformation_model[0] == 'anisotropic':
                if self.deformation_model[1] == 'fixed_shape': 
                    self.displacement_jacobian = anisotropic_plate_fixed_shape(Rx, Ry, self.S.in_units('GPa^-1'),
                                                                               self.thickness.in_units(self._jacobian_length_unit))
                else:
                    self.displacement_jacobian = anisotropic_plate_fixed_torques(Rx, Ry, self.S.in_units('GPa^-1'),
                                                                                 self.thickness.in_units(self._jacobian_length_unit))[0]
            else: 
                self.displacement_jacobian = isotropic_plate(Rx, Ry, self.nu,
                                                             self.thickness.in_units(self._jacobian_length_unit))[0]

    def __str__(self):
        #TODO: Improve output presentation
        if self.isotropy == 'anisotropic':
            elastic_str = 'Compliance matrix S (with rotations applied):\n' + np.array2string(self.S.in_units('GPa^-1'),precision=4, suppress_small =True) + ' GPa^-1'
        else:
            elastic_str = "Young's modulus E: " + str(self.E) + "\nPoisson's ratio nu: "+ str(self.nu) 

        return 'Crystal: ' + self.crystal_data['name'] + '\n' +\
               'Crystallographic parameters:\n' +\
               '    a = ' + str(self.crystal_data['a']*0.1)[:8] + ' nm,  b = ' + str(self.crystal_data['b']*0.1)[:8] + ' nm,  c = ' + str(self.crystal_data['c']*0.1)[:8] + ' nm\n'+\
               '    alpha = ' + str(self.crystal_data['alpha']) + ' deg,  beta = ' + str(self.crystal_data['beta']) + ' nm,  gamma = ' + str(self.crystal_data['gamma']) + ' deg\n'+\
               'Direct primitive vectors (before rotations, in nm):\n'+\
               '    a1 = '+np.array2string(0.1*self.direct_primitives[:,0],precision=4,suppress_small=True)+'\n'+\
               '    a2 = '+np.array2string(0.1*self.direct_primitives[:,1],precision=4,suppress_small=True)+'\n'+\
               '    a3 = '+np.array2string(0.1*self.direct_primitives[:,2],precision=4,suppress_small=True)+'\n'+\
               'Reciprocal primitive vectors (before rotations, in 1/nm):\n' +\
               '    b1 = ' + np.array2string(10*self.reciprocal_primitives[:,0],precision=4,suppress_small=True)+'\n'+\
               '    b2 = ' + np.array2string(10*self.reciprocal_primitives[:,1],precision=4,suppress_small=True)+'\n'+\
               '    b3 = ' + np.array2string(10*self.reciprocal_primitives[:,2],precision=4,suppress_small=True)+'\n\n'+\
               'Reflection: '+str(self.hkl)+'\n'+\
               'Asymmetry angle: ' + str(self.asymmetry)+'\n'+\
               'In-plane rotation angle: ' + str(self.in_plane_rotation)+'\n'+\
               'Crystal directions parallel to the Cartesian axes (after rotations):\n'+\
               '    x || ' + np.array2string(self.crystal_directions[:,0]/np.abs(self.crystal_directions[:,0]).max(),precision=4,suppress_small=True)+'\n'+\
               '    y || ' + np.array2string(self.crystal_directions[:,1]/np.abs(self.crystal_directions[:,1]).max(),precision=4,suppress_small=True)+'\n'+\
               '    z || ' + np.array2string(self.crystal_directions[:,2]/np.abs(self.crystal_directions[:,2]).max(),precision=4,suppress_small=True)+'\n'+\
               'Debye-Waller factor: ' + str(self.debye_waller)+'\n\n'+\
               'Crystal thickness: ' + str(self.thickness)+'\n'+\
               'Meridional bending radius: ' + str(self.Rx) +'\n'+\
               'Sagittal bending radius: ' + str(self.Ry) +'\n\n'+\
               'Material elastic isotropy: ' + str(self.isotropy) +'\n' + elastic_str        