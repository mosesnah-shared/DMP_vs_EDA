import numpy as np

# Derived from EXPLICIT
# [REF] https://github.com/explicit-robotics/Explicit-MATLAB
# For people who are interested about the details of deriving these values, please feel free to reach out to me:
# [Email] mosesnah@mit.edu

def get2DOF_M( q_arr ):
    q1, q2 = q_arr

    M = np.zeros( ( 2, 2 ) )
    M[0, 0] = np.cos(q2) + 7/2
    M[0, 1] = np.cos(q2)/2 + 5/4
    M[1, 0] = np.cos(q2)/2 + 5/4
    M[1, 1] = 5/4

    return M

def get2DOF_C( q_arr, dq_arr ):
    q1,   q2 = q_arr
    dq1, dq2 = dq_arr

    C = np.zeros( ( 2, 2 ) )
    C[0, 0] = -(dq2*np.sin(q2))/2
    C[0, 1] = -(np.sin(q2)*(dq1 + dq2))/2
    C[1, 0] = (dq1*np.sin(q2))/2
    C[1, 1] = 0    

    return C

def get2DOF_J( q_arr ):
    q1, q2 = q_arr
    J = np.zeros( ( 2, 2 ) )
    J[0, 0] = - np.sin(q1 + q2) - np.sin(q1)
    J[0, 1] = -np.sin(q1 + q2)
    J[1, 0] = np.cos(q1 + q2) + np.cos(q1)
    J[1, 1] = np.cos(q1 + q2)

    return J   

def get2DOF_dJ( q_arr, dq_arr ):
    q1,   q2 = q_arr
    dq1, dq2 = dq_arr

    dJ = np.zeros( ( 2, 2 ) )

    dJ[0, 0] = - np.cos(q1 + q2)*(dq1 + dq2) - dq1*np.cos(q1)
    dJ[0, 1] = -np.cos(q1 + q2)*(dq1 + dq2)
    dJ[1, 0] = - np.sin(q1 + q2)*(dq1 + dq2) - dq1*np.sin(q1)
    dJ[1, 1] = -np.sin(q1 + q2)*(dq1 + dq2)

    return dJ

def get2DOF_IK( p_arr ):

    q_arr = np.zeros( 2 )

    px = p_arr[ 0 ]
    py = p_arr[ 1 ]

    # Solve the inverse kinematics 
    q_arr[ 1 ] = np.pi - np.arccos( np.clip( 0.5 * ( 2 - px ** 2 - py ** 2  ), -1, 1 ) )
    q_arr[ 0 ] = np.arctan2( py, px ) - q_arr[ 1 ]/2     

    return q_arr