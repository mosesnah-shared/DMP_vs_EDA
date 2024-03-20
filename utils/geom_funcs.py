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

def is_SO3(R):
    """
        Quick check whether R is a rotation matrix.
    """
    # Check if the matrix is square
    if R.shape[0] != R.shape[1]:
        return False
    # Check if the determinant is close to 1
    if not np.isclose(np.linalg.det(R), 1):
        return False
    # Check if R * R^T is close to the identity matrix
    if not np.allclose(np.dot(R, R.T), np.eye(R.shape[0])):
        return False
    return True

def quat_real(quat):
    """
    Get the scalar part of a quaternion.

    Parameters:
    quat (numpy.ndarray): 1x4 or 4x1 quaternion vector.

    Returns:
    float: Scalar part of the quaternion.
    """
    assert isinstance(quat, np.ndarray), "quat must be a numpy array"
    assert quat.shape == (4,) or quat.shape == (1, 4) or quat.shape == (4, 1), "quat must be 1x4 or 4x1"

    # If quat is a 2D array (1, 4) or (4, 1), convert it to a 1D array
    if quat.ndim == 2:
        quat = quat.flatten()

    qw = quat[0]
    return qw

def quat_imag(quat):
    """
    Get the imaginary part of a quaternion.

    Parameters:
    quat (numpy.ndarray): 1x4 or 4x1 quaternion vector.

    Returns:
    3D vector: imaginary part of the quaternion.
    """
    assert isinstance(quat, np.ndarray), "quat must be a numpy array"
    assert quat.shape == (4,) or quat.shape == (1, 4) or quat.shape == (4, 1), "quat must be 1x4 or 4x1"

    # If quat is a 2D array (1, 4) or (4, 1), convert it to a 1D array
    if quat.ndim == 2:
        quat = quat.flatten()

    qxyz = quat[ -3: ]
    return qxyz


def ExpQuat(quat):
    """
    Exponential Map of a quaternion using np.isclose for threshold comparison.
    
    Parameters:
    quat (numpy.ndarray): 1x4 or 4x1 quaternion vector.
    
    Returns:
    numpy.ndarray: The exponential map of the quaternion.
    """
    assert quat.shape in [(4,), (1, 4), (4, 1)], "Quaternion must have 4 elements."
    
    # Ensure quat is a 1D array for convenience
    quat = quat.flatten()
    
    # Extract the scalar and vector parts of the quaternion
    qw = quat[0]
    qv = quat[1:]
    
    # Calculate the norm of the vector part
    tmp = np.linalg.norm(qv)
    
    # Compute the exponential map
    quat_new = np.zeros_like(quat)
    if np.isclose(tmp, 0):
        quat_new[0] = 1
    else:
        quat_new[0] = np.exp(qw) * np.cos(tmp)
        quat_new[1:] = np.exp(qw) * np.sin(tmp) * qv / tmp
    
    return quat_new


def get_quat_error( quat1, quat2 ):
    """Get the error 3D vector between two unit quaternions."""
    # Assuming is_unit_quat checks if a quaternion is a unit quaternion
    # assert is_unit_quat(quat1) and is_unit_quat(quat2)

    assert np.isclose( np.linalg.norm( quat1 ), 1.0), "Quaternion 1 must be a unit quaternion."
    assert np.isclose( np.linalg.norm( quat2 ), 1.0), "Quaternion 2 must be a unit quaternion."

    vec = 2 * quat_imag( LogQuat(  quat_mul( quat_conj( quat1 ), quat2 )  )  );
    return vec

def LogQuat(quat):
    """
    Logarithmic Map of a unit quaternion, without the 1e-6 threshold check.
    
    Parameters:
    quat (numpy.ndarray): 1x4 or 4x1 unit quaternion vector.
    
    Returns:
    numpy.ndarray: The logarithm of the quaternion.
    """
    assert np.isclose( np.linalg.norm(quat), 1.0), "Quaternion must be a unit quaternion."
    
    # Initialize the result quaternion
    quat_new = np.zeros_like(quat)
    
    # Extract the real (scalar) part and the imaginary (vector) part of the quaternion
    qw = quat_real(quat)
    qv = quat_imag(quat)
    
    qv_norm = np.linalg.norm(qv)
    
    # Compute the logarithm directly
    if qv_norm != 0:  # To handle division by zero
        quat_new[1:] = np.arccos(qw) * qv / qv_norm
    # No else branch needed as quat_new is initialized with zeros
    
    return quat_new



def quat_mul(quat1, quat2):
    """
    Perform quaternion multiplication.
    
    Parameters:
    quat1 (numpy.ndarray): 1x4 or 4x1 quaternion vector.
    quat2 (numpy.ndarray): 1x4 or 4x1 quaternion vector.
    
    Returns:
    numpy.ndarray: The result of quaternion multiplication, quat1 x quat2.
    """
    assert isinstance(quat1, np.ndarray) and isinstance(quat2, np.ndarray), "Inputs must be numpy arrays."
    assert quat1.shape in [(4,), (1, 4), (4, 1)] and quat2.shape in [(4,), (1, 4), (4, 1)], "Both quaternions must be 1x4 or 4x1"
    assert quat1.size == 4 and quat2.size == 4, "Both quaternions must have 4 elements."

    # Ensure quaternions are 1D arrays for the calculation
    quat1 = quat1.flatten()
    quat2 = quat2.flatten()

    # Quaternion multiplication
    quat_new = np.zeros(4)
    quat_new[0] = quat1[0] * quat2[0] - np.dot(quat1[-3:], quat2[-3:])
    quat_new[-3:] = (quat1[0] * quat2[-3:] + quat2[0] * quat1[-3:] + np.cross(quat1[-3:], quat2[-3:]))

    return quat_new


def quat_conj(quat):
    """
    Perform quaternion conjugation.
    
    Parameters:
    quat (numpy.ndarray): 1x4 or 4x1 quaternion vector.
    
    Returns:
    numpy.ndarray: The conjugated version of the input quaternion.
    """
    assert isinstance(quat, np.ndarray), "quat must be a numpy array."
    assert quat.shape in [(4,), (1, 4), (4, 1)] and quat.size == 4, "quat must be 1x4 or 4x1 and have 4 elements."

    # Ensure quat is a 1D array for the calculation
    quat = quat.flatten()

    # Create a new quaternion for the conjugate
    quat_new = np.zeros_like(quat)
    quat_new[0] = quat[0]
    quat_new[1:4] = -quat[1:4]

    return quat_new

def SO3_to_quat(R):
    """
    Conversion from SO3 to unit quaternion.

    Parameters:
    R (numpy.ndarray): An SO3 matrix

    Returns:
    numpy.ndarray: unit quaternion of R
    """
    assert is_SO3(R), "R must be a valid SO3 matrix."

    # Calculate the Quaternion Matrix
    R00 = np.trace(R)

    values = np.abs([R00, R[0, 0], R[1, 1], R[2, 2]])
    k = np.argmax(values)

    if k == 0:
        ek = 0.5 * np.sqrt(1 + R00)
    else:
        ek = 0.5 * np.sqrt(1 + 2 * R[k-1, k-1] - R00)

    if k == 0:
        e0 = ek
        e1 = (R[2, 1] - R[1, 2]) / (4 * ek)
        e2 = (R[0, 2] - R[2, 0]) / (4 * ek)
        e3 = (R[1, 0] - R[0, 1]) / (4 * ek)
    elif k == 1:
        e0 = (R[2, 1] - R[1, 2]) / (4 * ek)
        e1 = ek
        e2 = (R[1, 0] + R[0, 1]) / (4 * ek)
        e3 = (R[0, 2] + R[2, 0]) / (4 * ek)
    elif k == 2:
        e0 = (R[0, 2] - R[2, 0]) / (4 * ek)
        e1 = (R[1, 0] + R[0, 1]) / (4 * ek)
        e2 = ek
        e3 = (R[2, 1] + R[1, 2]) / (4 * ek)
    elif k == 3:
        e0 = (R[1, 0] - R[0, 1]) / (4 * ek)
        e1 = (R[0, 2] + R[2, 0]) / (4 * ek)
        e2 = (R[2, 1] + R[1, 2]) / (4 * ek)
        e3 = ek

    quat = np.array([e0, e1, e2, e3])

    if e0 < 0:
        quat = -quat

    return quat

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


def dLogQuat(quat, dquat):
    """
    Derivative of the Logarithmic Map of a unit quaternion.
    
    Parameters:
    quat (numpy.ndarray): 1x4 or 4x1 unit quaternion vector.
    dquat (numpy.ndarray): Time derivative of the quaternion, 4x1 vector.
    
    Returns:
    numpy.ndarray: 1x3 result vector.
    """
    assert np.isclose( np.linalg.norm( quat ), 1.0), "quat must be a unit quaternion."
    assert dquat.shape == (4,) or dquat.shape == (4, 1), "dquat must be a 4-element vector."
    
    # Ensure dquat is a 1D array for convenience
    dquat = dquat.flatten()
    
    # Getting out the parts
    eta = quat_real(quat)
    eps = quat_imag(quat)
    eps_norm = np.linalg.norm(eps)
    
    # Check to prevent division by zero
    if np.isclose( np.linalg.norm( eps_norm ), 0):
        raise ValueError("Norm of the imaginary part is too small for stable computation.")
    
    # The Jacobian matrix
    a1 = (-eps_norm + np.arccos(eta) * eta) / eps_norm**3
    a2 = np.arccos(eta) / eps_norm
    JQ = np.hstack([a1 * eps.reshape(-1, 1), np.eye(3) * a2])  # Reshape eps for correct broadcasting
    

    vec_new = JQ @ dquat  # Use only the vector part of dquat
    
    return vec_new

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