from __future__ import division, print_function
from numpy import inf
from .elastic_tensors import rotate_elastic_matrix
from .rotation_matrix import inplane_rotation

def isotropic_plate(Rx,Ry,nu,thickness):
    '''
    Creates a function for computing the Jacobian of
    the displacement field for an isotropic plate.
    HINT: For cylindrical bending with anticlastic
    curvature, set Ry = -Rx/nu
    '''

    if Rx == 'inf' or Rx == 'Inf' or Rx == inf:
        invRx = 0
    else:
        invRx = 1/Rx

    if Ry == 'inf' or Ry == 'Inf' or Ry == inf:
        invRy = 0
    else:
        invRy = 1/Ry

    def jacobian(x,z):
        ux_x = -(z+0.5*thickness)*invRx
        ux_z = -x*invRx

        uz_x = x*invRx
        uz_z = nu/(1-nu)*(invRx+invRy)*(z+0.5*thickness)

        return [[ux_x,ux_z],[uz_x,uz_z]]

    return jacobian

def anisotropic_plate(Rx,Ry,S,thickness):
    '''
    Creates a function for computing the Jacobian of
    the displacement field for an isotropic plate.
    '''
    if Rx == 'inf' or Rx == 'Inf' or Rx == inf:
        invRx = 0
    else:
        invRx = 1/Rx

    if Ry == 'inf' or Ry == 'Inf' or Ry == inf:
        invRy = 0
    else:
        invRy = 1/Ry

    #In the general case, the torques are not necessarely aligned with
    #x- and y-axes but have to be rotated.

    S = np.array(S)

    meps = np.finfo(type(S[0][0])).eps
    if abs(S[5,0]) < meps and abs(S[5,1]) < meps and abs(S[1,1] - S[0,0]) < meps and abs(S[0,0] + S[1,1] - 2*S[0,1] - S[5,5]) < meps:
        alpha = 0
    else:
        Aa = S[5,5]*(S[0,0] + S[1,1] + 2*S[0,1]) - (S[5,0] + S[5,1])**2
        Ba = 2*(S[5,1]*(S[0,1] + S[0,0]) - S[5,0]*(S[0,1] + S[1,1])) 
        Ca = S[5,5]*(S[1,1]-S[0,0]) + S[5,0]**2 - S[5,1]**2
        Da = 2*(S[5,1]*(S[0,1] - S[0,0]) + S[5,0]*(S[0,1] - S[1,1]))

        alpha = 0.5*np.arctan(Da*(invRy+invRx) - Ba*(invRy-invRx), Aa*(invRy-invRx) - Ca*(invRy+invRx))

    #rotate S by alpha
    Sp = rotate_elastic_matrix(S, 'S', inplane_rotation(alpha))

    #Precomputed coefficients
    mx = 0.5*((Sp[0,1]-Sp[1,1])*(invRy + invRx) + (Sp[0,1]+Sp[1,1])*(invRy - invRx)*np.cos(2*alpha))/(Sp[0,0]*Sp[1,1] - Sp[0,1]*Sp[0,1])
    my = 0.5*((Sp[0,1]-Sp[0,0])*(invRy + invRx) + (Sp[0,1]+Sp[0,0])*(invRy - invRx)*np.cos(2*alpha))/(Sp[0,0]*Sp[1,1] - Sp[0,1]*Sp[0,1])

    coef1 = Sp[2,0]*mx + Sp[2,1]*my
    coef2 = (Sp[4,0]*mx + Sp[4,1]*my)*np.cos(alpha) - (Sp[3,0]*mx + Sp[3,1]*my)*np.sin(alpha)

    def jacobian(x,z):
        ux_x = -invRx*(z+0.5*thickness)
        ux_z = -invRx*x + coef2*(z+0.5*thickness)

        uz_x = invRx*x
        uz_z = coef1*(z+0.5*thickness)

        return [[ux_x,ux_z],[uz_x,uz_z]]

    return jacobian
