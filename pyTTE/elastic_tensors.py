from __future__ import division, print_function
import numpy as np

#Elastic constants for single crystals in units 10^11 Pa
#Source: CRC Handbook of Chemistry and Physics, 82nd edition
CRYSTALS = {
            'Si' : {'system' : 'cubic', 'C11' : 1.6578, 'C12' : 0.6394, 'C44' : 0.7962},
            'Ge' : {'system' : 'cubic', 'C11' : 1.2835, 'C12' : 0.4823, 'C44' : 0.6666},
            'CaMoO4' : {'system' : 'tetragonal', 'C11' : 1.447, 'C12' : 0.664,\
                        'C13' : 0.455, 'C16' : 0.134, 'C33':1.265,'C44':0.369,'C66':0.451},
            'CaCO3'  : {'system' : 'orthorhombic', 'C11' : 1.5958, 'C12' : 0.3663, 'C13' : 0.0197,\
                        'C22' : 0.8697,'C23':0.1597, 'C33':0.8503,'C44' : 0.4132,'C55':0.2564,'C66':0.4274},
            'aegirine' : {'system' : 'monoclinic', 'C11' : 1.858, 'C12' : 0.685, 'C13' : 0.707, 'C15':0.098,'C22':1.813,\
                         'C23':0.626, 'C25':0.094,'C33':2.344,'C35':0.214,'C44':0.629,'C46':0.077,'C55':0.510,'C66':0.474},
            'Be' : {'system' : 'hexagonal', 'C11': 2.923, 'C12':0.267, 'C13':0.140, 'C33':3.364,'C55':1.625},
            'quartz' : {'system': 'trigonal', 'C11': 0.8670, 'C12':0.0704, 'C13':0.1191, 'C14':-0.1804,'C33':1.0575,'C44':0.5820},

            'test_cubic' : {'system' : 'cubic', 'C11' : 11, 'C12' : 12, 'C44' : 44},
            'test_tetragonal' : {'system' : 'tetragonal', 'C11' : 11, 'C12' : 12,'C13' : 13, 'C16' : 16, 'C33':33,'C44':44,'C66':66},
            'test_orthorhombic'  : {'system' : 'orthorhombic', 'C11':11,'C12':12,'C13':13,'C22':22,'C23':23,'C33':33,'C44' : 44,'C55':55,'C66':66},
            'test_monoclinic' : {'system' : 'monoclinic', 'C11':11,'C12' :12,'C13':13,'C15':15,'C22':22,'C23':23,'C25':25,'C33':33,'C35':35,'C44':44,'C46':46,'C55':55,'C66':66},
            'test_hexagonal' : {'system' : 'hexagonal', 'C11':11, 'C12':12, 'C13':13, 'C33':33,'C55':55},
            'test_trigonal' : {'system': 'trigonal', 'C11':11,'C12':12,'C13':13,'C14':14,'C33':33,'C44':44},
            'test_triclinic': {'system':'triclinic', 'C11':11,'C12':12,'C13':13,'C14':14,'C15':15,'C16':16,\
                                                     'C22':22,'C23':23,'C24':24,'C25':25,'C26':26,\
                                                     'C33':33,'C34':34,'C35':35,'C36':36,\
                                                     'C44':44,'C45':45,'C46':46,\
                                                     'C55':55,'C56':56,\
                                                     'C66':66}
           }

def matrix2tensor(matrix):
    '''
        Converts the stiffness matrix using Voigt notation to elastic tensor.
    '''
    Cc = np.zeros((3,3,3,3))

    C11 = matrix[0,0]; C12 = matrix[0,1]; C13 = matrix[0,2]
    C14 = matrix[0,3]; C15 = matrix[0,4]; C16 = matrix[0,5]

    C22 = matrix[1,1]; C23 = matrix[1,2]; C24 = matrix[1,3]
    C25 = matrix[1,4]; C26 = matrix[1,5]

    C33 = matrix[2,2]; C34 = matrix[2,3]
    C35 = matrix[2,4]; C36 = matrix[2,5]

    C44 = matrix[3,3]; C45 = matrix[3,4]; C46 = matrix[3,5]

    C55 = matrix[4,4]; C56 = matrix[4,5]

    C66 = matrix[5,5]

    Cc[0,0,0,0] = C11
    Cc[0,0,1,1], Cc[1,1,0,0] = C12, C12
    Cc[0,0,2,2], Cc[2,2,0,0] = C13, C13
    Cc[0,0,1,2], Cc[0,0,2,1], Cc[1,2,0,0], Cc[2,1,0,0] = C14, C14, C14, C14
    Cc[0,0,2,0], Cc[0,0,0,2], Cc[0,2,0,0], Cc[2,0,0,0] = C15, C15, C15, C15
    Cc[0,0,0,1], Cc[0,0,1,0], Cc[0,1,0,0], Cc[1,0,0,0] = C16, C16, C16, C16

    Cc[1,1,1,1] = C22
    Cc[1,1,2,2], Cc[2,2,1,1] = C23, C23
    Cc[1,1,1,2], Cc[1,1,2,1], Cc[1,2,1,1], Cc[2,1,1,1] = C24, C24, C24, C24
    Cc[1,1,2,0], Cc[1,1,0,2], Cc[0,2,1,1], Cc[2,0,1,1] = C25, C25, C25, C25
    Cc[1,1,0,1], Cc[1,1,1,0], Cc[0,1,1,1], Cc[1,0,1,1] = C26, C26, C26, C26

    Cc[2,2,2,2] = C33
    Cc[2,2,1,2], Cc[2,2,2,1], Cc[1,2,2,2], Cc[2,1,2,2] = C34, C34, C34, C34
    Cc[2,2,2,0], Cc[2,2,0,2], Cc[0,2,2,2], Cc[2,0,2,2] = C35, C35, C35, C35
    Cc[2,2,0,1], Cc[2,2,1,0], Cc[0,1,2,2], Cc[1,0,2,2] = C36, C36, C36, C36

    Cc[1,2,1,2], Cc[1,2,2,1], Cc[2,1,1,2], Cc[2,1,2,1] = C44, C44, C44, C44
    Cc[1,2,2,0], Cc[1,2,0,2], Cc[2,1,2,0], Cc[2,1,0,2] = C45, C45, C45, C45
    Cc[2,0,1,2], Cc[0,2,1,2], Cc[2,0,2,1], Cc[0,2,2,1] = C45, C45, C45, C45
    Cc[1,2,0,1], Cc[1,2,1,0], Cc[2,1,0,1], Cc[2,1,1,0] = C46, C46, C46, C46
    Cc[0,1,1,2], Cc[1,0,1,2], Cc[0,1,2,1], Cc[1,0,2,1] = C46, C46, C46, C46

    Cc[2,0,2,0], Cc[2,0,0,2], Cc[0,2,2,0], Cc[0,2,0,2] = C55, C55, C55, C55
    Cc[2,0,0,1], Cc[2,0,1,0], Cc[0,2,0,1], Cc[0,2,1,0] = C56, C56, C56, C56
    Cc[0,1,2,0], Cc[1,0,2,0], Cc[0,1,0,2], Cc[1,0,0,2] = C56, C56, C56, C56

    Cc[0,1,0,1], Cc[0,1,1,0], Cc[1,0,0,1], Cc[1,0,1,0] = C66, C66, C66, C66

    return Cc

def tensor2matrix(tensor):
    '''
        Converts the stiffness tensor to matrix using Voigt notation.
    '''

    Crot = tensor

    C = np.array([
        [Crot[0,0,0,0], Crot[0,0,1,1], Crot[0,0,2,2], Crot[0,0,1,2], Crot[0,0,0,2], Crot[0,0,0,1]],
        [Crot[1,1,0,0], Crot[1,1,1,1], Crot[1,1,2,2], Crot[1,1,1,2], Crot[1,1,0,2], Crot[1,1,0,1]],
        [Crot[2,2,0,0], Crot[2,2,1,1], Crot[2,2,2,2], Crot[2,2,1,2], Crot[2,2,0,2], Crot[2,2,0,1]],
        [Crot[2,1,0,0], Crot[2,1,1,1], Crot[2,1,2,2], Crot[1,2,1,2], Crot[1,2,0,2], Crot[1,2,0,1]],
        [Crot[2,0,0,0], Crot[2,0,1,1], Crot[2,0,2,2], Crot[2,0,1,2], Crot[0,2,0,2], Crot[2,0,0,1]],
        [Crot[1,0,0,0], Crot[1,0,1,1], Crot[1,0,2,2], Crot[1,0,1,2], Crot[1,0,0,2], Crot[0,1,0,1]]
    ]).reshape((6,6))
    
    return C
    
def rotation_matrix(hkl,system='cubic'):
    '''
        Computes the rotation matrix which aligns the given hkl along z-axis.
        NOTE: works currently only for the cubic systems
        TODO: figure out how to generalize to other systems, preferably by
        obtaining the crystal directions from the elastic tensors
    '''
    if not system == 'cubic':
        raise NotImplementedError('Rotation of non-cubic systems not implemented!')

    if hkl[0] or hkl[1]:
        #rotation axis
        u = -np.array([hkl[1],-hkl[0]])/np.sqrt(hkl[0]**2+hkl[1]**2)
        #rotation angle
        th = np.arccos(hkl[2]/np.sqrt(hkl[0]**2+hkl[1]**2+hkl[2]**2))

        #rotation matrix
        R=np.array([[np.cos(th)+u[0]**2*(1-np.cos(th)), u[0]*u[1]*(1-np.cos(th)), u[1]*np.sin(th)],
           [u[0]*u[1]*(1-np.cos(th)), np.cos(th)+u[1]**2*(1-np.cos(th)), -u[0]*np.sin(th)],
           [-u[1]*np.sin(th), u[0]*np.sin(th), np.cos(th)]])
    else:
        R=np.array([[1,0,0],[0,1,0],[0,0,1]])

    return R.transpose()

def rotation_matrix_axis_angle(u,theta):
    '''
        Computes a matrix which performs a rotation of theta degrees about axis u.
    '''    
    #normalize
    u = np.array(u)
    u = u/np.sqrt(u[0]**2+u[1]**2+u[2]**2)
    #rotation angle
    th = np.radians(theta)
    print('u',u)
    #rotation matrix
    R=np.array([[np.cos(th)+u[0]**2*(1-np.cos(th)), u[0]*u[1]*(1-np.cos(th)) - u[2]*np.sin(th), u[0]*u[2]*(1-np.cos(th)) + u[1]*np.sin(th)],
       [u[0]*u[1]*(1-np.cos(th)) + u[2]*np.sin(th), np.cos(th)+u[1]**2*(1-np.cos(th)), u[1]*u[2]*(1-np.cos(th)) -u[0]*np.sin(th)],
       [u[0]*u[2]*(1-np.cos(th))-u[1]*np.sin(th), u[1]*u[2]*(1-np.cos(th)) + u[0]*np.sin(th), np.cos(th) + u[2]**2*(1-np.cos(th))]])

    return R.transpose()

def compute_elastic_matrices(zdir, xtal):
    '''
        Computes the compliance and stiffness matrices S and C a given z-direction.
        The x- and y-directions are determined automatically
        returns: S, C, x_dir, y_dir
    '''

    try:
        xtal_data=CRYSTALS[xtal]
    except KeyError:
        raise KeyError("Elastic parameters for '"+str(xtal)+"' not found!")
       
    if xtal_data['system'] == 'cubic':
        C11, C12, C44 = xtal_data['C11'], xtal_data['C12'], xtal_data['C44']
        C13, C14, C15, C16 = C12, 0, 0, 0
        C22, C23, C24, C25, C26 = C11, C12, 0, 0, 0
        C33, C34, C35, C36 = C11, 0, 0, 0
        C45, C46 = 0, 0
        C55, C56 = C44, 0
        C66 = C44
    elif xtal_data['system'] == 'tetragonal':
        C11, C12, C13 = xtal_data['C11'], xtal_data['C12'], xtal_data['C13']
        C16 = xtal_data['C16']
        C33  = xtal_data['C33']
        C44 = xtal_data['C44']
        C66 = xtal_data['C66']
        C14, C15 = 0, 0
        C22, C23, C24, C25, C26 = C11, C13, 0, 0, -C16
        C34, C35, C36 = 0, 0, 0
        C45, C46 = 0, 0
        C55, C56 = C44, 0
    elif xtal_data['system'] == 'orthorhombic':
        C11, C12, C13 = xtal_data['C11'], xtal_data['C12'], xtal_data['C13']
        C22, C23 = xtal_data['C22'], xtal_data['C23']
        C33  = xtal_data['C33']
        C44 = xtal_data['C44']
        C55 = xtal_data['C55']
        C66 = xtal_data['C66']
        C14, C15, C16 = 0,0,0
        C24, C25, C26 = 0,0,0
        C34, C35, C36 = 0,0,0
        C45, C46 = 0,0
        C56 = 0
    elif xtal_data['system'] == 'monoclinic':
        C11, C12, C13 = xtal_data['C11'], xtal_data['C12'], xtal_data['C13']
        C15 = xtal_data['C15']
        C22, C23, C25 = xtal_data['C22'], xtal_data['C23'], xtal_data['C25']
        C33, C35 = xtal_data['C33'], xtal_data['C35']
        C44, C46 = xtal_data['C44'], xtal_data['C46']
        C55 = xtal_data['C55']
        C66 = xtal_data['C66']
        C14, C16, C24, C26, C34, C36, C45, C56 = 0,0,0,0,0,0,0,0
    elif xtal_data['system'] == 'hexagonal':
        C11, C12, C13 = xtal_data['C11'], xtal_data['C12'], xtal_data['C13']
        C33  = xtal_data['C33']
        C55 = xtal_data['C55']
        C14, C15, C16 = 0, 0, 0
        C22, C23, C24, C25, C26 = C11, C13, 0, 0, 0
        C34, C35, C36 = 0, 0, 0
        C44, C45, C46 = C55, 0, 0
        C56 =  0
        C66 = (C11-C12)/2
    elif xtal_data['system'] == 'trigonal':
        C11, C12, C13, C14 = xtal_data['C11'], xtal_data['C12'], xtal_data['C13'], xtal_data['C14']
        C33  = xtal_data['C33']
        C44 = xtal_data['C44']
        C15, C16 = 0, 0
        C22, C23, C24, C25, C26 = C11, C13, -C14, 0, 0
        C34, C35, C36 = 0, 0, 0
        C45, C46 =  0, 0
        C55, C56 =  C44, C14
        C66 = (C11-C12)/2
    elif xtal_data['system'] == 'triclinic':
        C11, C12, C13 = xtal_data['C11'], xtal_data['C12'], xtal_data['C13']
        C14, C15, C16 = xtal_data['C14'], xtal_data['C15'], xtal_data['C16']
        C22, C23, C24 = xtal_data['C22'], xtal_data['C23'], xtal_data['C24']
        C25, C26 = xtal_data['C25'], xtal_data['C26']
        C33, C34, C35 = xtal_data['C33'], xtal_data['C34'], xtal_data['C35']
        C36 =  xtal_data['C36']
        C44, C45, C46 = xtal_data['C44'], xtal_data['C45'], xtal_data['C46']
        C55, C56 = xtal_data['C55'], xtal_data['C56']
        C66 = xtal_data['C66']
    else:
        ValueError('Not a valid crystal system!')

    Cc = matrix2tensor(np.array([[C11, C12, C13, C14, C15, C16],
                                 [C12, C22, C23, C24, C25, C26],
                                 [C13, C23, C33, C34, C35, C36],
                                 [C14, C24, C34, C44, C45, C46],
                                 [C15, C25, C35, C45, C55, C56],
                                 [C16, C26, C36, C46, C56, C66]]))

    Q = rotation_matrix(zdir,xtal_data['system'])

    #Rotate the tensor
    #New faster version according to
    #http://stackoverflow.com/questions/4962606/fast-tensor-rotation-with-numpy

    QQ = np.outer(Q,Q)
    QQQQ = np.outer(QQ,QQ).reshape(4*Q.shape)
    axes = ((0, 2, 4, 6), (0, 1, 2, 3))
    Crot = np.tensordot(QQQQ, Cc, axes)

    #Assemble the stiffness matrix
    C = tensor2matrix(Crot)

    C=C*1e11 #in pascal
    S = np.linalg.inv(C)

    #calculate x and y directions
    #TODO generalize to non-cubic systems
    x_dir = np.dot(Q.T,np.array([[1,0,0]]).T)
    y_dir = np.dot(Q.T,np.array([[0,1,0]]).T)

    return S, C, x_dir, y_dir

def rotate_inplane_and_apply_asymmetry(tensor, phi, asymmetry):
    
    #In-plane rotation
    Q = rotation_matrix_axis_angle([0,0,1],phi)
    QQ = np.outer(Q,Q)
    QQQQ = np.outer(QQ,QQ).reshape(4*Q.shape)
    axes = ((0, 2, 4, 6), (0, 1, 2, 3))
    tensor = np.tensordot(QQQQ, tensor, axes)

    x_dir = np.dot(Q.T,np.array([[1,0,0]]).T)
    y_dir = np.dot(Q.T,np.array([[0,1,0]]).T)

    #asymmetry rotation (NB for asymmetry the rotation axis is -y)
    Q = rotation_matrix_axis_angle([0,-1,0],asymmetry)
    QQ = np.outer(Q,Q)
    QQQQ = np.outer(QQ,QQ).reshape(4*Q.shape)
    axes = ((0, 2, 4, 6), (0, 1, 2, 3))
    tensor = np.tensordot(QQQQ, tensor, axes)

    x_dir = np.dot(Q.T,x_dir)
    y_dir = np.dot(Q.T,y_dir)

    #return tensor, x_dir, y_dir
    return tensor

if __name__=='__main__':
    print('Cubic:\n',np.array2string(compute_elastic_matrices([1,0,0],'test_cubic')[1]/1e11,precision=4,suppress_small=True))


    C,x,y = rotate_inplane_and_apply_asymmetry(matrix2tensor(compute_elastic_matrices([1,1,1],'test_cubic')[1]/1e11),0,90)

    print(np.array2string(tensor2matrix(C),precision=4,suppress_small=True))
    print('x',x)
    print('y',y)
       
    '''
    print('Tetragonal:\n',np.array2string(compute_elastic_matrices([0,0,1],'test_tetragonal')[1]/1e11,precision=4,suppress_small=True))
    print('Orthorhombic:\n',np.array2string(compute_elastic_matrices([0,0,1],'test_orthorhombic')[1]/1e11,precision=4,suppress_small=True))
    print('Monoclinic:\n',np.array2string(compute_elastic_matrices([0,0,1],'test_monoclinic')[1]/1e11,precision=4,suppress_small=True))
    print('Hexagonal:\n',np.array2string(compute_elastic_matrices([0,0,1],'test_hexagonal')[1]/1e11,precision=4,suppress_small=True))
    print('Trigonal:\n',np.array2string(compute_elastic_matrices([0,0,1],'test_trigonal')[1]/1e11,precision=4,suppress_small=True))
    print('Triclinic:\n',np.array2string(compute_elastic_matrices([0,0,1],'test_triclinic')[1]/1e11,precision=4,suppress_small=True))
    '''