from __future__ import division, print_function

def isotropic_plate(Rx,Ry,nu,thickness):
    '''
    Creates a function for computing the Jacobian of
    the displacement field for an isotropic plate.
    HINT: For cylindrical bending with anticlastic
    curvature, set Ry = -Rx/nu
    '''

    if Rx == 'inf' or Rx == 'Inf' or Rx == np.inf:
        invRx = 0
    else:
        invRx = 1/Rx

    if Ry == 'inf' or Ry == 'Inf' or Ry == np.inf:
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

