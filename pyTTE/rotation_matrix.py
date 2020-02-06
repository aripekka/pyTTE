from __future__ import division, print_function
import numpy as np

def align_vector_with_z_axis(h):
    '''
    Computes the rotation matrix which aligns the given vector along z-axis.
    For example, for reflection (hkl), h = h*b1 + k*b2 + l*b3, where bi
    are the primitive reciprocal vectors.
    '''

    if h[0] or h[1]:
        #rotation axis
        u = np.array([h[1],-h[0]])/np.sqrt(h[0]**2+h[1]**2)
        #rotation angle
        th = np.arccos(h[2]/np.sqrt(h[0]**2+h[1]**2+h[2]**2))
    else:
        if h[2] > 0:
            #zero deg rotation about -y
            u = np.array([0,-1])
            th = 0
        else:
            #180 deg rotation about -y
            u = np.array([0,-1])
            th = np.pi

    #rotation matrix
    R=np.array([[ np.cos(th) + u[0]**2*(1-np.cos(th)),          u[0]*u[1]*(1-np.cos(th)),  u[1]*np.sin(th)],
                [            u[0]*u[1]*(1-np.cos(th)), np.cos(th)+u[1]**2*(1-np.cos(th)), -u[0]*np.sin(th)],
                [                    -u[1]*np.sin(th),                   u[0]*np.sin(th),       np.cos(th)]])

    return R

def rotation_matrix_axis_angle(u,theta):
    '''
        Computes a matrix which performs a rotation of theta degrees counterclockwise about axis u.
    '''    
    #normalize
    u = np.array(u)
    u = u/np.sqrt(u[0]**2+u[1]**2+u[2]**2)
    #rotation angle
    th = np.radians(theta)

    #rotation matrix
    R=np.array([[        np.cos(th) + u[0]**2*(1-np.cos(th)), u[0]*u[1]*(1-np.cos(th)) - u[2]*np.sin(th), u[0]*u[2]*(1-np.cos(th)) + u[1]*np.sin(th)],
                [ u[0]*u[1]*(1-np.cos(th)) + u[2]*np.sin(th),        np.cos(th) + u[1]**2*(1-np.cos(th)), u[1]*u[2]*(1-np.cos(th)) - u[0]*np.sin(th)],
                [ u[0]*u[2]*(1-np.cos(th)) - u[1]*np.sin(th), u[1]*u[2]*(1-np.cos(th)) + u[0]*np.sin(th),        np.cos(th) + u[2]**2*(1-np.cos(th))]])

    return R

def rotate_inplane(tensor, phi, x_dir = np.array([[1,0,0]]).T, y_dir = np.array([[0,1,0]]).T):
    '''
        Rotates the given tensor around the z-axis by phi degrees counterclockwise.
        x_dir and y_dir are the crystal directions (normalized hkl) along the x- and 
        y-axes.
    '''

    #In-plane rotation
    Q = rotation_matrix_axis_angle([0,0,1],phi)
    for i in range(4):
        tensor = np.tensordot(Q,tensor,axes=((1,),(i,)))

    #calculate the crystal directions along the in-plane axes  

    #this computes what were the directions of post-rotation x- and y axes 
    #in terms of the pre-rotated x- and y- coordinates
    prerot_x_coor = np.dot(Q.T, np.array([[1,0,0]]).T)
    prerot_y_coor = np.dot(Q.T, np.array([[0,1,0]]).T)
    
    #prerotated x(y)-axis aligns with the crystal direction x(y)_dir
    #and assuming that they are properly normalized, they form an orthonormal basis.
    #Thus the postrotation x_dir and y_dir are linear combinations of
    #prerot x_dir and y_dir based on the calculated coordinate transform:
    new_x_dir = prerot_x_coor[0]*x_dir + prerot_x_coor[1]*y_dir
    new_y_dir = prerot_y_coor[0]*x_dir + prerot_y_coor[1]*y_dir

    return tensor, new_x_dir, new_y_dir

def apply_asymmetry(tensor, phi, x_dir = np.array([[1,0,0]]).T, y_dir = np.array([[0,1,0]]).T):
    '''
        Rotates the given tensor around the y-axis by phi degrees counterclockwise.
        This corresponds to the definition of clockwise-positive asymmetry angle in
        xz-plane as defined in the documentation. x_dir and y_dir are the crystal 
        directions (normalized hkl) along the x- and y-axes.
    '''

    #Asymmetric rotation (note that in pyTTE documentation
    #rotation angle is positive in clockwise direction in
    #right-handed xz-plane = counterclockwise rotation around y-axis)
    Q = rotation_matrix_axis_angle([0,1,0],phi)
    for i in range(4):
        tensor = np.tensordot(Q,tensor,axes=((1,),(i,)))

    #calculate the crystal directions along the in-plane axes  

    #this computes what were the directions of post-rotation x- and y axes 
    #in terms of the pre-rotated x- and y- coordinates
    prerot_x_coor = np.dot(Q.T, np.array([[1,0,0]]).T)
    prerot_y_coor = np.dot(Q.T, np.array([[0,1,0]]).T)
    
    #prerotated x(y)-axis aligns with the crystal direction x(y)_dir
    #and assuming that they are properly normalized, they form an orthonormal basis.
    #Thus the postrotation x_dir and y_dir are linear combinations of
    #prerot x_dir and y_dir based on the calculated coordinate transform:
    new_x_dir = prerot_x_coor[0]*x_dir + prerot_x_coor[1]*y_dir
    new_y_dir = prerot_y_coor[0]*x_dir + prerot_y_coor[1]*y_dir

    return tensor, new_x_dir, new_y_dir

