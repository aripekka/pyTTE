from __future__ import division, print_function
from numpy import inf

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
    HINT: For cylindrical bending with anticlastic
    curvature, set Ry = S_11/S_12*Rx
    '''
    if Rx == 'inf' or Rx == 'Inf' or Rx == inf:
        invRx = 0
    else:
        invRx = 1/Rx

    if Ry == 'inf' or Ry == 'Inf' or Ry == inf:
        invRy = 0
    else:
        invRy = 1/Ry

    #Precomputed coefficients
    mx = (S[0][1]*invRy-S[1][1]*invRx)/(S[0][0]*S[1][1]-S[0][1]*S[1][0])
    my = (S[1][0]*invRx-S[0][0]*invRy)/(S[0][0]*S[1][1]-S[0][1]*S[1][0])

    coef1 = S[0][0]*mx + S[0][1]*my
    coef2 = S[4][0]*mx + S[4][1]*my
    coef3 = S[2][0]*mx + S[2][1]*my

    def jacobian(x,z):
        ux_x = coef1*(z+0.5*thickness)
        ux_z = coef1*x+coef2*(z+0.5*thickness)

        uz_x = -coef1*x
        uz_z = coef3*(z+0.5*thickness)

        return [[ux_x,ux_z],[uz_x,uz_z]]

    return jacobian
