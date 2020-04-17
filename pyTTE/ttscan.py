# -*- coding: utf-8 -*-

from numpy import linspace

from .quantity import Quantity

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
                        params['scan'] = Quantity(linspace(float(ls[1]),float(ls[2]),int(ls[3])),ls[4])
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
        
        if self.scan[0] == 'manual':
            N_points = self.scan[1].value.size
            limit_str = 'manual from ' + str(self.scan[1].value.min()) \
                        + ' to ' + str(self.scan[1].value.max()) + ' ' \
                        + Quantity._unit2str(self.scan[1].unit)
        else:
            N_points = self.scan[1]
            limit_str = 'automatic'
            
        return 'Scan type     : ' + self.scantype + '\n' +\
               'Scan constant : ' + str(self.constant) +'\n' +\
               'Polarization  : ' + self.polarization  +'\n' +\
               'Scan points   : ' + str(N_points)  +'\n' +\
               'Scan range    : ' + limit_str   +'\n'


