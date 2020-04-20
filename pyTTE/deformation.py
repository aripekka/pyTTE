from __future__ import division, print_function
from numpy import inf, array, finfo, cos, sin, arctan2, isinf
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

def anisotropic_plate_fixed_shape(R1,R2,S,thickness):
    '''
    Creates a function for computing the Jacobian of the displacement field 
    for an anisotropic plate with fixed shape.
    
    Parameters
    ----------
    
    R1 : float
        The meridional bending radius (in the same units as R2 and thickness).
        Use 'inf' if the direction is unbent.
        
    R2 : float
        The sagittal bending radius (in the same units as R1 and thickness)
        Use 'inf' if the direction is unbent.
        
    S : 6x6 Numpy array of floats
        The compliance matrix in the Voigt notation. Units are not important.
        
    thickness : float
        The thickness of the crystal (in the same units as R1 and R2)


    Returns
    -------    
    
    jacobian : function
        Returns the partial derivatives of the displacement vector u as a 
        function of coordinates (x,z). The length scale is determined by the
        units of R1, R2 and thickness.
        
    '''
    
    
    #Convert the bending radii to their inverses:    
    if isinf(float(R1)):
        invR1 = 0
    else:
        invR1 = 1.0/R1

    if isinf(float(R2)):
        invR2 = 0
    else:
        invR2 = 1.0/R2

    #calculate the rotation angle alpha
    S = array(S)
    meps = finfo(type(S[0][0])).eps
    
    if abs(S[5,0]) < meps and abs(S[5,1]) < meps and abs(S[1,1] - S[0,0]) < meps\
    and abs(S[0,0] + S[1,1] - 2*S[0,1] - S[5,5]) < meps:
        alpha = 0  
    else:
        Aa = S[5,5]*(S[0,0] + S[1,1] + 2*S[0,1]) - (S[5,0] + S[5,1])**2
        Ba = 2*(S[5,1]*(S[0,1] + S[0,0]) - S[5,0]*(S[0,1] + S[1,1])) 
        Ca = S[5,5]*(S[1,1]-S[0,0]) + S[5,0]**2 - S[5,1]**2
        Da = 2*(S[5,1]*(S[0,1] - S[0,0]) + S[5,0]*(S[0,1] - S[1,1]))

        alpha = 0.5*arctan2(Da*(invR2+invR1) - Ba*(invR2-invR1), 
                            Aa*(invR2-invR1) - Ca*(invR2+invR1))

    #rotate S by alpha
    Sp = rotate_elastic_matrix(S, 'S', inplane_rotation(alpha))

    #Compute torques
    m_divider = 2*(Sp[0,0]*Sp[1,1] - Sp[0,1]*Sp[0,1])
    
    mx = (Sp[0,1]-Sp[1,1])*(invR2 + invR1)
       + (Sp[0,1]+Sp[1,1])*(invR2 - invR1)*cos(2*alpha)
    mx = mx / m_divider

    my = (Sp[0,1]-Sp[0,0])*(invR2 + invR1)
       - (Sp[0,1]+Sp[0,0])*(invR2 - invR1)*cos(2*alpha)
    my = my / m_divider  

    #Coeffiecients for the Jacobian
    coef1 = Sp[2,0]*mx + Sp[2,1]*my
    coef2 = (Sp[4,0]*mx + Sp[4,1]*my)*cos(alpha) 
          - (Sp[3,0]*mx + Sp[3,1]*my)*sin(alpha)

    def jacobian(x,z):
        ux_x = -invR1*(z+0.5*thickness)
        ux_z = -invR1*x + coef2*(z+0.5*thickness)

        uz_x = invR1*x
        uz_z = coef1*(z+0.5*thickness)

        return [[ux_x,ux_z],[uz_x,uz_z]]

    return jacobian
