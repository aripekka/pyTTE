from __future__ import division, print_function
import numpy as np

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
    xtal = xtal.lower()

    #TODO: read the elastic constants from a file
    if xtal == 'ge':
    	c1111, c1122, c2323 = 1.292, 0.479, 0.670
    elif xtal == 'si':
    	c1111, c1122, c2323 = 1.657, 0.639, 0.796
    else:
        raise ValueError('Elastic parameters for the crystal not found!')

    #TODO: generalize to other systems alongside the cubic as well
    Cc = np.zeros((3,3,3,3))

    Cc[0,0,0,0], Cc[1,1,1,1], Cc[2,2,2,2] = c1111, c1111, c1111
    Cc[0,0,1,1], Cc[0,0,2,2], Cc[1,1,0,0] = c1122, c1122, c1122
    Cc[1,1,2,2], Cc[2,2,0,0], Cc[2,2,1,1] = c1122, c1122, c1122

    Cc[0,2,0,2], Cc[2,0,0,2], Cc[0,2,2,0], Cc[2,0,2,0] = c2323, c2323, c2323, c2323
    Cc[1,2,1,2], Cc[2,1,1,2], Cc[1,2,2,1], Cc[2,1,2,1] = c2323, c2323, c2323, c2323
    Cc[0,1,0,1], Cc[1,0,0,1], Cc[0,1,1,0], Cc[1,0,1,0] = c2323, c2323, c2323, c2323

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
    print(compute_elastic_matrices([1,1,1],'si'))
