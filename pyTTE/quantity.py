# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np

UNITS = {'m'   : ('length', 1e0),
         'cm'  : ('length', 1e-2),
         'mm'  : ('length', 1e-3),
         'µm'  : ('length', 1e-6),
         'um'  : ('length', 1e-6),
         'nm'  : ('length', 1e-9),
         'Å'   : ('length', 1e-10),
         'A'   : ('length', 1e-10),
         'AA'  : ('length', 1e-10),

         'J'   : ('energy', 1e0),
         'MeV' : ('energy', 1.602176634e-13),
         'keV' : ('energy', 1.602176634e-16),
         'eV'  : ('energy', 1.602176634e-19),
         'meV' : ('energy', 1.602176634e-22),
         'µeV' : ('energy', 1.602176634e-25),
         'ueV' : ('energy', 1.602176634e-25),
         'neV' : ('energy', 1.602176634e-28),

         'rad'     : ('angle', 1e0),
         'mrad'    : ('angle', 1e-3),
         'urad'    : ('angle', 1e-6),
         'µrad'    : ('angle', 1e-6),
         'nrad'    : ('angle', 1e-9),
         'deg'     : ('angle', np.pi/180),
         'arcmin'  : ('angle', np.pi/10800),
         'arc min' : ('angle', np.pi/10800),
         'arcsec'  : ('angle', np.pi/648000),
         'arc sec' : ('angle', np.pi/648000)
}


class Quantity:
    '''
    A class used to deal with various different physical units used in the diffraction calculations.
    '''

    #The available units in the class. The unit is given as a key to the dictionary whose values are
    #tuples whose first field is the type of unit and the second one is the conversion factor to the
    #base unit (often in SI)

    def __init__(self,value,unit):
        if type(value) in [type(1),type(1.0),type(1j)]:
            self.value = value
        else:
            raise ValueError('Value of the quantity has to be one of following types: int, float, complex') 

        if type(unit) == type(''):
            if not unit in UNITS.keys():
                raise ValueError('Unit not found in the list of available units.')
            self.unit = unit
        else:
            raise ValueError('Unit of the quantity has to be of str type.') 


    def in_units(self,unit):
        '''
        Converts the quantity to given unit. If the conversion is not valid, will raise a ValueError.

        Input:
            unit = str of the unit to be converted to.
        Output:
            value = converted value in the units required
        '''

        if not unit in UNITS.keys():
            raise ValueError('Unit not found in the list of available units.')

        #check that the old unit and the new unit are of the same type
        if not UNITS[self.unit][0] == UNITS[unit][0]:
            raise ValueError('Can not convert '+ UNITS[self.unit][0] + ' to '+ UNITS[unit][0] +'.' )

        value = self.value*UNITS[self.unit][1]/UNITS[unit][1]
        return value

    def __str__(self):
        return str(self.value)+' '+self.unit
