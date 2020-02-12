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
        '''
        Intitialize a Quantity instance.

        Input:
            value = numeric data (literal, list, numpy array)
            unit  = string of the unit e.g. mm, deg, eV. Can also be a compound unit such as 'm s^-1'
                    or 'eV s'. Note the proper formatting (e.g. m/s or eV*s are not valid).
                    
        '''

        value = np.array(value)
        try:
            value + 1.0; value * 1.0    #Check if the input is of numeric type 
            self.value = value
        except:
            raise ValueError('Value(s) of the quantity has to be numerical!') 

        if type(unit) == type(''):
            self.unit = Quantity._parse_units(unit)
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

        unit_parsed = Quantity._parse_units(unit)

        unit_p_type = {}
        unit_self_type = {}

        #calculate the unit dimensions
        for k in unit_parsed:
            utype = UNITS[k][0]

            #first encounter, add unit in the dictionary
            if not utype in unit_p_type:
                unit_p_type[utype] = unit_parsed[k]
            else:
                unit_p_type[utype] = unit_p_type[utype] + unit_parsed[k]

        for k in self.unit:
            utype = UNITS[k][0]

            #first encounter, add unit in the dictionary
            if not utype in unit_self_type:
                unit_self_type[utype] = self.unit[k]
            else:
                unit_self_type[utype] = unit_self_type[utype] + self.unit[k]

        #compare units
        for k1 in unit_p_type:
            match_found = False
            for k2 in unit_self_type:
                if k1 == k2 and unit_p_type[k1] == unit_self_type[k2]:
                    match_found = True                       
            if not match_found:
                raise ValueError('Can not convert '+ Quantity._unit2str(unit_self_type) + ' to '+ Quantity._unit2str(unit_p_type) +'.')
        for k1 in unit_self_type:
            match_found = False
            for k2 in unit_p_type:
                if k1 == k2 and unit_self_type[k1] == unit_p_type[k2]:
                    match_found = True                       
            if not match_found:
                raise ValueError('Can not convert '+ Quantity._unit2str(unit_self_type) + ' to '+ Quantity._unit2str(unit_p_type) +'.')

        #calculate the conversion factor
        convf = 1
        for k in self.unit:
            convf = convf*UNITS[k][1]**self.unit[k]
        for k in unit_parsed:
            convf = convf/UNITS[k][1]**unit_parsed[k]

        value = self.value*convf
        return value

    def type(self):
        '''
        Returns the str of the type of the quantity. 
        '''
        return 'temp' #UNITS[self.unit][0]



    def _parse_units(unit_string):
        '''
        Parses the compounded unit string into basic units.
        '''
        
        unit_str_split = unit_string.split()

        #parse units exponents
        units = {}
        for u in unit_str_split:
            u2 = u.split('^')

            if not u2[0] in UNITS.keys():
                raise ValueError(str(u2[0]) + ' not found in the list of available units.')

            #first encounter, add unit in the dictionary
            if not u2[0] in units.keys():
                units[u2[0]] = 0

            if len(u2) == 1:
                #no exponent given explicitely -> exponent = 1
                units[u2[0]] = units[u2[0]] + 1
            elif len(u2) == 2:
                exponent = int(u2[1]) 
                if not exponent == 0:
                    units[u2[0]] = units[u2[0]] + exponent
            else:
                raise ValueError('Invalid unit: '+ str(u) + '.')

        #remove exponents zero from the unit list
        for k in units:
            if units[k] == 0:
                del units[k]

        return units
    
    def _unit2str(unit):

        '''
        Makes a pretty string representation of unit dictionary
        '''
        
        unit_str = ''

        for k in unit:
            unit_str = unit_str + k
            if unit[k] == 1:
                unit_str = unit_str + ' '
            else:
                unit_str = unit_str +'^'+str(unit[k])+' '

        return unit_str[:-1]

    def _type2str(unit):

        '''
        Makes a pretty string representation of unit dictionary's type
        '''
        unit_type = {}

        for k in unit:
            utype = UNITS[k][0]

            #first encounter, add unit in the dictionary
            if not utype in unit_type:
                unit_type[utype] = unit[k]
            else:
                unit_type[utype] = unit_type[utype] + unit[k]

        return Quantity._unit2str(unit_type)

    def __str__(self):
        unit_str = ''

        for k in self.unit:
            unit_str = unit_str + k
            if self.unit[k] == 1:
                unit_str = unit_str + ' '
            else:
                unit_str = unit_str +'^'+str(self.unit[k])+' '

        return str(self.value)+' '+ unit_str[:-1]
