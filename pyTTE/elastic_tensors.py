from __future__ import division, print_function
import numpy as np

#Elastic constants for single crystals in units 10^11 Pa
#Source: CRC Handbook of Chemistry and Physics, 82nd edition
CRYSTALS = {
            'Si' : {'system' : 'cubic', 'C11' : 1.6578, 'C12' : 0.6394, 'C44' : 0.7962},
            'Ge' : {'system' : 'cubic', 'C11' : 1.2835, 'C12' : 0.4823, 'C44' : 0.6666},
           }

def rotation_matrix(hkl):
    '''
        Computes the rotation matrix which aligns the given hkl along z-axis.
        NOTE: works currently only for the cubic systems
    '''

    if hkl[0] or hkl[1]:
        #rotation axis
        u = -np.array([[hkl[1]],[-hkl[0]]])/np.sqrt(hkl[0]**2+hkl[1]**2)
        #rotation angle
        th = np.arccos(hkl[2]/np.sqrt(hkl[0]**2+hkl[1]**2+hkl[2]**2))

        #rotation matrix
        R=np.array([[np.cos(th)+u[0]**2*(1-np.cos(th)), u[0]*u[1]*(1-np.cos(th)), u[1]*np.sin(th)],
           [u[0]*u[1]*(1-np.cos(th)), np.cos(th)+u[1]**2*(1-np.cos(th)), -u[0]*np.sin(th)],
           [-u[1]*np.sin(th), u[0]*np.sin(th), np.cos(th)]])
    else:
        R=np.array([[1,0,0],[0,1,0],[0,0,1]])

    return R.transpose()

def compute_elastic_matrices(zdir, xtal):
    '''
        Computes the compliance and stiffness matrices S and C a given z-direction.
        The x- and y-directions are determined automatically
    '''

    try:
        xtal_data=CRYSTALS[xtal]
    except KeyError:
        raise KeyError("Elastic parameters for '"+str(xtal)+"' not found!")

    if xtal_data['system'] == 'cubic':
        C11, C12, C44 = xtal_data['C11'], xtal_data['C12'], xtal_data['C44']
        C22, C33 = C11, C11
        C13, C23 = C12, C12
        C55, C66 = C44, C44
        C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 = 0,0,0,0,0,0,0,0,0,0,0,0
    else:
        ValueError('Non-cubic systems not implemented yet!')

    Cc = np.zeros((3,3,3,3))

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

    '''
    #TODO: generalize to other systems alongside the cubic as well
    Cc = np.zeros((3,3,3,3))

    Cc[0,0,0,0], Cc[1,1,1,1], Cc[2,2,2,2] = c1111, c1111, c1111
    Cc[0,0,1,1], Cc[0,0,2,2], Cc[1,1,0,0] = c1122, c1122, c1122
    Cc[1,1,2,2], Cc[2,2,0,0], Cc[2,2,1,1] = c1122, c1122, c1122

    Cc[0,2,0,2], Cc[2,0,0,2], Cc[0,2,2,0], Cc[2,0,2,0] = c2323, c2323, c2323, c2323
    Cc[1,2,1,2], Cc[2,1,1,2], Cc[1,2,2,1], Cc[2,1,2,1] = c2323, c2323, c2323, c2323
    Cc[0,1,0,1], Cc[1,0,0,1], Cc[0,1,1,0], Cc[1,0,1,0] = c2323, c2323, c2323, c2323
    '''

    Q = rotation_matrix(zdir)

    #Rotate the tensor
    #New faster version according to
    #http://stackoverflow.com/questions/4962606/fast-tensor-rotation-with-numpy

    QQ = np.outer(Q,Q)
    QQQQ = np.outer(QQ,QQ).reshape(4*Q.shape)
    axes = ((0, 2, 4, 6), (0, 1, 2, 3))
    Crot = np.tensordot(QQQQ, Cc, axes)

    #Assemble the stiffness matrix
    C = np.array([
        [Crot[0,0,0,0], Crot[0,0,1,1], Crot[0,0,2,2], Crot[0,0,1,2], Crot[0,0,0,2], Crot[0,0,0,1]],
        [Crot[1,1,0,0], Crot[1,1,1,1], Crot[1,1,2,2], Crot[1,1,1,2], Crot[1,1,0,2], Crot[1,1,0,1]],
        [Crot[2,2,0,0], Crot[2,2,1,1], Crot[2,2,2,2], Crot[2,2,1,2], Crot[2,2,0,2], Crot[2,2,0,1]],
        [Crot[2,1,0,0], Crot[2,1,1,1], Crot[2,1,2,2], Crot[1,2,1,2], Crot[1,2,0,2], Crot[1,2,0,1]],
        [Crot[2,0,0,0], Crot[2,0,1,1], Crot[2,0,2,2], Crot[2,0,1,2], Crot[0,2,0,2], Crot[2,0,0,1]],
        [Crot[1,0,0,0], Crot[1,0,1,1], Crot[1,0,2,2], Crot[1,0,1,2], Crot[1,0,0,2], Crot[0,1,0,1]]
    ]).reshape((6,6))


    C=C*1e11 #in pascal
    S = np.linalg.inv(C)

    return S, C

if __name__=='__main__':
    print(compute_elastic_matrices([1,0,0],'Si'))
