import numpy as np

def rotx( q ):
    Rx = np.array( [ [ 1,            0,            0 ], 
                     [ 0,  np.cos( q ), -np.sin( q ) ],
                     [ 0,  np.sin( q ),  np.cos( q ) ]  ] )

    return Rx

def roty( q ):
    Ry = np.array( [ [  np.cos( q ),  0,  np.sin( q ) ], 
                     [            0,  1,            0 ],
                     [ -np.sin( q ),  0,  np.cos( q ) ]  ] )

    return Ry

def rotz( q ):
    Rz = np.array( [ [ np.cos( q ), -np.sin( q ), 0 ], 
                     [ np.sin( q ),  np.cos( q ), 0 ],
                     [           0,            0, 1 ]  ] )

    return Rz

def is_skew_symmetric( matrix ):
    """
    Check if a matrix is skew-symmetric.
    
    Parameters:
    matrix (numpy array): The matrix to check.
    
    Returns:
    is_skew (bool): True if the matrix is skew-symmetric, False otherwise.
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check if the matrix is equal to the negation of its transpose
    return np.allclose(matrix, -matrix.T)


def R3_to_so3( w ):
    """
    Convert a 3-element vector 'w' to a skew-symmetric matrix.
    """

    if w.shape != (3,):
        raise ValueError("Input 'w' must be a 3-element vector")


    return np.array( [ [   0, -w[2],  w[1] ],
                       [ w[2],    0, -w[0] ],
                       [-w[1], w[0],     0 ] ])

def so3_to_R3( mat ):
    """
    Convert a 3-element vector 'w' to a skew-symmetric matrix.
    """

    assert( is_skew_symmetric( mat ) )

    return np.array( [ mat[ 2,1  ], mat[ 0, 2], mat[ 1, 0 ] ] )


def SO3_to_R3(rotation_matrix):
    """
    Logarithmic map from SO(3) to R3.

    :param rotation_matrix: A 3x3 rotation matrix.
    :return: A 3D vector representing the axis-angle representation.
    """

    # Ensure the matrix is close to a valid rotation matrix
    if not np.allclose(np.dot(rotation_matrix.T, rotation_matrix), np.eye(3)) or not np.allclose(np.linalg.det(rotation_matrix), 1):
        raise ValueError("The input matrix is not a valid rotation matrix.")
 
    angle = np.arccos((np.trace(rotation_matrix) - 1) / 2)
    
    # Check for the singularity (angle close to 0)
    if np.isclose(angle, 0) or np.isnan( angle ):
        return np.zeros(3)

    # Compute the skew-symmetric matrix
    skew_symmetric = (rotation_matrix - rotation_matrix.T) / (2 * np.sin(angle))
    
    # Extract the rotation axis
    axis = np.array([skew_symmetric[2, 1], skew_symmetric[0, 2], skew_symmetric[1, 0]])

    return axis * angle

def R3_to_SO3( r3_vector ):
    angle = np.linalg.norm(r3_vector)
    if np.isclose(angle, 0):
        return np.eye(3)
    
    axis = r3_vector / angle
    skew_symmetric = R3_to_so3( axis )
    rotation_matrix = np.eye(3) + np.sin(angle) * skew_symmetric + (1 - np.cos(angle)) * np.dot(skew_symmetric, skew_symmetric)

    return rotation_matrix