# ========================================================================================== #
#  [Script Name]: DMP.py under example9_pos_and_orient
#       [Author]: Moses Chong-ook Nah
#      [Contact]: mosesnah@mit.edu
# [Date Created]: 2024.03.20
#  [Description]: Simulation using Dynamic Movement Primitives (DMP)
#                 Discrete movement in task-space, both position and orientation
#                 This .py file is for running/generating Figure 14 of the 
#                 following manuscript from Nah, Lachner and Hogan
#                 "Robot Control based on Motor Primitives — A Comparison of Two Approaches" 
#
#                 The code is exhaustive in details, so that people can really try out the 
#                 demonstrations by themselves!
#                 "Code is read more often than it is written" - Guido Van Rossum
# ========================================================================================== #

# ========================================================================================== #
# [Section #1] Imports and Enviromental SETUPS
# ========================================================================================== #

import sys
import numpy as np
import mujoco
import mujoco_viewer
from scipy.io import savemat

from pathlib  import Path

# Add and save the ROOT_PATH to run this code, which is three levels above.
ROOT_PATH = str( Path( __file__ ).parent.parent.parent )
sys.path.append( ROOT_PATH )

# Also save the CURRENT PATH of this File
CURRENT_PATH = str( Path( __file__ ).parent )

# Importing the Minimum-jerk Trajectory Function
from utils.data_process import data_diff
from utils.trajectory   import  min_jerk_traj
from utils.geom_funcs   import rotx, R3_to_SO3, R3_to_so3, so3_to_R3, SO3_to_R3, SO3_to_quat, \
                               get_quat_error, LogQuat, dLogQuat, quat_mul, quat_conj, ExpQuat

# Importing the modules for Dynamic Movement Primitives (DMP)
from DMPmodules.canonical_system        import CanonicalSystem 
from DMPmodules.nonlinear_forcing_term  import NonlinearForcingTerm
from DMPmodules.transformation_system   import TransformationSystem

# Set numpy print options
np.set_printoptions( precision = 4, threshold = 9, suppress = True )

# ========================================================================================== #
# [Section #2] Basic MuJoCo Setups
# ========================================================================================== #

# Basic MuJoCo setups
dir_name   = ROOT_PATH + '/models/iiwa14/'
robot_name = 'iiwa14.xml'
model      = mujoco.MjModel.from_xml_path( dir_name + robot_name )
data       = mujoco.MjData( model )
viewer     = mujoco_viewer.MujocoViewer( model, data, hide_menus = True )

# Parameters for the simulation
T        = 6.                      # Total Simulation Time
dt       = model.opt.timestep       # Time-step for the simulation (set in xml file)
fps      = 30                       # Frames per second
save_ps  = 1000                     # Saving point per second
n_frames = 0                        # The current frame of the simulation
n_saves  = 0                        # Update for the saving point
speed    = 1.0                      # The speed of the simulator
t_update = 1./fps     * speed       # Time for update 
t_save   = 1./save_ps * speed       # Time for saving the data
nq       = model.nq                 # Degrees of freedom of the robot
assert( dt <= t_update and dt <= t_save )

# The initial position of KUKA iiwa14, which was discovered manually
q_init = np.array( [-0.5000, 0.8236, 0.0, -1.0472, 0.8000, 1.5708, 0.0 ] )
data.qpos[ 0:nq ] = q_init
mujoco.mj_forward( model, data )

# Get the end-effector's ID
EE_site = "site_end_effector"
id_EE = model.site( EE_site ).id

# ========================================================================================== #
# [Section #3] Imitation Learning, for Position
# ========================================================================================== #

# Saving the reference for task-space position
p  = data.site_xpos[ id_EE ]

# The parameters of the discrete movement, position
t0 = 0.0
pi = np.copy( p )
pf = pi - 2 * np.array( [ 0.0, pi[ 1 ], 0.0 ] )
D  = 2.0

# Parameters of the DMP
tau    = D                    # Duration of the demonstrated trajectory
alphas = 1.0                  # Decay rate of the Canonical System
N      = 50                   # Number of Basis Functions
P      = 300                  # Number of Sample points for the demonstrated trajectory
az     = 10.0                 # Coefficient for the Transformation System
bz     =  2.5                 # Coefficient for the Transformation System
y0d    = pi                   # Initial Position of the demonstrated trajectory
gd     = pf                   #    Goal Position of the demonstrated trajectory

# Defining the Three elements of DMP
sd        = CanonicalSystem( 'discrete', D, alphas )
fd        = NonlinearForcingTerm( sd, N )
trans_sys = TransformationSystem( az, bz, sd )

# To conduct Imitation Learning, we need P sample points of position, velocity and acceleration 
# of the demonstrated trajectory
# Suffix "d" is added for demonstration
td   = np.linspace( 0, D, num = P )  
pd   = np.zeros( ( 3, P ) )          
dpd  = np.zeros( ( 3, P ) )          
ddpd = np.zeros( ( 3, P ) )          

for i, t in enumerate( td ):
    p_tmp, dp_tmp, ddp_tmp = min_jerk_traj( t, 0, D, pi, pf )

    pd[   :, i ] =   p_tmp
    dpd[  :, i ] =  dp_tmp
    ddpd[ :, i ] = ddp_tmp

# Method1: Locally Weighted Regression (LWR)
W_LWR = np.zeros( ( 3, N ) )

# Iterating for each weight
for i in range( 3 ): # For XYZ coordinates
    for j in range( N ):
        a_arr = sd.calc( td ) * ( gd[ i ] - y0d[ i ] )
        b_arr = trans_sys.get_desired( pd[ i, : ], dpd[ i, : ], ddpd[ i, : ], gd[ i ] )
        phi_arr = fd.calc_ith( td, j ) 
        
        # Element-wise multiplication and summation
        if np.sum( a_arr * a_arr * phi_arr ) == 0:
            W_LWR[ i, j ] = 0
        else:
            W_LWR[ i, j ] = np.sum( a_arr * b_arr * phi_arr ) / np.sum( a_arr * a_arr * phi_arr )

W_LWR = np.nan_to_num( W_LWR )

# Method2: Linear Least-square Regression (LLS)
A_mat = np.zeros( ( N, P ) )
B_mat = trans_sys.get_desired( pd, dpd, ddpd, gd )

# Interating along the time sample points
for i, t in enumerate( td ):
    A_mat[ :, i ] = np.squeeze( fd.calc_multiple_ith( t, np.arange( N, dtype=int ) ) ) / fd.calc_whole_at_t( t ) * sd.calc( t )

# Weight with Linear Least-square Multiplication
W_LLS = B_mat @ A_mat.T @ np.linalg.inv( A_mat @ A_mat.T )

# Scaling down 
for i in range( 2 ):
    if ( gd[ i ] - y0d[ i ] ) != 0:
        W_LLS[ i, : ] = 1./( ( gd[ i ] - y0d[ i ] ) ) * W_LLS[ i, : ]

# Rolling out the trajectory, to get the position, velocity and acceleration array
# The whole time array for the roll out
t_arr = np.arange( 0, T, dt )
t_arr = np.append( t_arr, T )   # Technical detail: Append the final to include T on the array


# One can choose either the weights learned by LWR or LLS
weight = W_LLS  # W_LWR

# Rollout of the position trajectory
input_arr = fd.calc_forcing_term( t_arr[:-1], weight, t0, np.diag( pf - pi ), trimmed = True )
p_arr, _, dp_arr, dz_arr = trans_sys.rollout( pi, dpd[ :, i ], pf, input_arr, t0, t_arr )
ddp_arr = dz_arr/sd.tau

# ========================================================================================== #
# [Section #4] Imitation Learning, for Orientation
# ========================================================================================== #

# The code use the formulation from Koutras and Doulgeri (2020):
# [REF] Koutras, Leonidas, and Zoe Doulgeri. "A correct formulation for the orientation dynamic movement primitives for robot control in the cartesian space." Conference on robot learning. PMLR, 2020.

# Saving the Reference for task-space orientation.
Rsb = data.site_xmat[ id_EE ]

# The initial/goal orientation which we aim to achieve
ang   = 90 
Rinit = np.copy( Rsb ).reshape( 3, -1 )     # Initial Orientation, in SO(3)
Rgoal = Rinit @ rotx( ang*np.pi/180 )       #    Goal Orientation, in SO(3)
quat_init = SO3_to_quat( Rinit )            # Initial Orientation, in Unit Quaternion S(4)
quat_goal = SO3_to_quat( Rgoal )            # Initial Orientation, in Unit Quaternion S(4)

# Calculate the displacement and its Lie Algebra
Rdel  = Rinit.T @ Rgoal
wdel  = SO3_to_R3( Rdel )

# The matrices which we collect for Imitation Learning
R_mat     = np.zeros( ( 3, 3, P ) )         # The Orientation in SO( 3 )
w_mat     = np.zeros( ( 3, P ) )            # The angular velocity of the trajectory
quat_mat  = np.zeros( ( 4, P ) )            # Unit Quaternion representation
dquat_mat = np.zeros( ( 4, P ) )            # The time derivative of the Unit Quaternion

err_mat   = np.zeros( ( 3, P ) )            # Error vector, defined by Koutras and Doulgeri (2020).
derr_mat  = np.zeros( ( 3, P ) )            # Time derivative of the Error vector, defined by Koutras and Doulgeri (2020).

# The parameters for learning the trajectory is identical to those for position
for i, t in enumerate( td ):

    # The minimum-jerk trajectory on so(3) Lie Algebra
    tmp, dtmp, _ = min_jerk_traj( t, 0, D, np.zeros( 3 ), wdel )

    # The current R matrix and its corresponding Unit Quaternion
    R_curr    = Rinit @ R3_to_SO3( tmp )
    quat_curr = SO3_to_quat( R_curr )

    # Saving these values
    R_mat[ :, :, i ] =    R_curr
    quat_mat[ :, i ] = quat_curr

    # Get the angular velocity 
    # The basic left-invariant equation, dR R^T
    w_mat[ :, i ] = so3_to_R3( R_curr @ R3_to_so3( dtmp ) @ R_curr.T  )

    # The Quaternion Error
    err_mat[ :, i ]  = get_quat_error( quat_curr, quat_goal )

    # The time derivative of the quaternion, which is simply:
    # dq = 1/2 * w * q
    tmp_arr       = np.zeros( 4 )
    tmp_arr[ 1: ] = w_mat[ :, i ]       # 3D to 4D pure quaternion

    # Calculation
    dquat_mat[ :, i ] =  0.5 * quat_mul( tmp_arr, quat_curr )

    # The ime derivative of the error vector
    derr_mat[ :, i ] = dLogQuat( quat_mat[ :, i ], dquat_mat[ :, i ] )

# Once the terms are derived, do numerical differentiation for dderr_mat
# Although there is an analytical form, numerical differentiation is "enough" for application [Moses C. Nah]
dderr_mat = data_diff( derr_mat, td )

# Method 1: Locally Weighted Regression
alpha_s   =  1.0
alpha_z   = 2000.0
beta_z    = 0.5 * alpha_z
N         = 50
tau       = D

# The Three Elements of DMP for Orientation
# Note that the benefit of formulation Koutras and Doulgeri (2020) is that 
# the DMP formulation is identical to Rn case. 
# This is because we are learning the "error" vector, not the SO(3) itself.
sd2        = CanonicalSystem( 'discrete', tau, alpha_s )
fd2        = NonlinearForcingTerm( sd2, N )
trans_sys2 = TransformationSystem( alpha_z, beta_z, sd2 )

# Method1: Locally Weighted Regression (LWR)
W_LWR = np.zeros( ( 3, N ) )

# Get the scaling matrix
# Equation (9) of Koutras and Doulgeri (2020)
scl_arr = 2 * get_quat_error( quat_init, quat_goal )

# Iterating to derive each weight
for i in range( 3 ): # For XYZ coordinates
    for j in range( N ):
        a_arr = sd2.calc( td ) * scl_arr[ i ]
        b_arr = trans_sys2.get_desired( err_mat[ i, : ], derr_mat[ i, : ], dderr_mat[ i, : ], 0 )
        phi_arr = fd2.calc_ith( td, j ) 
        
        # Element-wise multiplication and summation
        W_LWR[ i, j ] = np.sum( a_arr * b_arr * phi_arr ) / np.sum( a_arr * a_arr * phi_arr )

W_LWR = np.nan_to_num( W_LWR )

# Method2: Linear Least-square Regression (LLS)
A_mat = np.zeros( ( N, P ) )
B_mat = trans_sys2.get_desired( err_mat, derr_mat, dderr_mat, np.zeros( 3 ) )

# Interating along the time sample points
for i, t in enumerate( td ):
    A_mat[ :, i ] = np.squeeze( fd2.calc_multiple_ith( t, np.arange( N, dtype=int ) ) ) / fd2.calc_whole_at_t( t ) * sd2.calc( t )

# Weight with Linear Least-square Regression
W_LLS = B_mat @ A_mat.T @ np.linalg.inv( A_mat @ A_mat.T )

# Scaling down the weight array
for i in range( 3 ):
    if scl_arr[ i ] != 0:
        W_LLS[ i, : ] = 1./scl_arr[ i ] * W_LLS[ i, : ]

# Choose either weights learned from LWR or LLS
weight = W_LLS # LWR


# Rollout for the Equations
y0 =  err_mat[ :, 0 ]             
z0 = derr_mat[ :, 0 ] * tau
sd2.tau = tau    

# Calculate the nonlinear forcing term and do the rollout
# The "trimmed" argument artifically set the nonlinear forcing term value to zero for time over the movement duration
# Not necessary, but it clears out the "tail" of DMP at the end of the movement. 
input_arr = fd2.calc_forcing_term( t_arr[:-1], weight, t0, np.diag( scl_arr ), trimmed = True )
e_arr, _, de_arr, dz_arr = trans_sys2.rollout( y0, z0, np.zeros( 3 ), input_arr, 0, t_arr )     # The third argument, goal is zero since error eventually converges to zero.
dde_arr = dz_arr/sd.tau

# Reconstruct the unit quaternion trajectory from the error vector.
quat_traj  = np.zeros( ( 4, len( t_arr ) ) )
dquat_traj = np.zeros( ( 4, len( t_arr ) ) )
w_traj     = np.zeros( ( 3, len( t_arr ) ) )

# Reconstruction from the error vector
# Equation (15) of Koutras and Doulgeri (2020)

for i in range( len( t_arr ) ):

    # 3D Error vector to pure unit quaternion
    tmp1 = np.zeros( 4 )
    tmp1[ 1: ] = e_arr[ :, i ]
    quat_traj[ :, i ] = quat_mul( quat_goal, quat_conj( ExpQuat( 0.5 * tmp1 ) ) )

    # Also calculate the angular velocity of the trajectory from error vectors
    # Equation (21) and (22) of Koutras and Doulgeri (2020)
    # Tedious calculation, but step by step!

    quat_tmp = quat_traj[ :, i ]
    tmp0 = quat_mul( quat_goal, quat_conj( quat_tmp ) ) 
    tmp1 = -1/2*quat_mul( quat_tmp, quat_conj( quat_goal ) ) 

    # Calculating the Jacobian matrix for quaternions, Equation (19) of Koutras and Doulgeri (2020)
    Jac_tmp = np.zeros( ( 4, 3 ) )
    theta = np.linalg.norm( LogQuat( tmp0 ) )

    if np.isclose( theta , 0 ):
        n_vec = np.zeros( 3 )
        Jac_tmp[1:, :] = np.eye( 3 )
    else:
        n_vec = LogQuat( tmp0 )/theta 
        Jac_tmp[ 0,: ]   = -np.sin( theta )* n_vec[ 1: ]
        Jac_tmp[ 1:, : ] =  np.sin( theta )/theta * ( np.eye( 3 ) - np.outer( n_vec[ 1: ], n_vec[ 1: ] ) ) + np.cos( theta ) * np.outer( n_vec[ 1: ], n_vec[ 1: ] ) 

    tmp3 = Jac_tmp @ de_arr[ :, i ]

    # Final calculation for dQ
    dquat_traj[ :, i ] = quat_mul( quat_mul( tmp1, tmp3 ), quat_traj[ :, i  ] )

    # The angular velocity!
    tmp = quat_mul( dquat_traj[ :, i ], quat_conj( quat_traj[ :, i ] ) )
    w_traj[ :, i ] = 2*tmp[ 1: ]
    
# ========================================================================================== #
# [Section #4] Main Simulation
# ========================================================================================== #

# Save the references for the q and dq 
q  = data.qpos[ 0:nq ]
dq = data.qvel[ 0:nq ]

# The data for mat save
t_mat   = [ ]
q_mat   = [ ] 
p_mat   = [ ] 
dq_mat  = [ ] 
dp_mat  = [ ] 
p_links_save = [ ]
R_links_save = [ ]
R_mat  = [ ]

# Saving the position and orientation of 7 links
# This is NOT required for the control, but for robot visualization at MATLAB
p_ref = []
R_ref = [] 

for i in range( 7 ):
    name = "iiwa14_link_" + str( i + 1 ) 

    p_ref.append( data.body( name ).xpos )
    R_ref.append( data.body( name ).xmat )

# Other Flags
is_save = True     # To save the data
is_view = True     # To view the simulation

# Step number for the simulation
n_sim = 0

# The parameters of the sliding-mode controller
Gr =  10 * np.eye(  3 )
Gp =  10 * np.eye(  3 )
Gq =   3 * np.eye( nq )

# Get the end-effector's ID and its position, translational and rotational Jacobian matrices
EE_site = "site_end_effector"
id_EE   = model.site( EE_site ).id
p       = data.site_xpos[ id_EE ]
Jp      = np.zeros(  (  3, nq ) )
Jr      = np.zeros(  (  3, nq ) )
J_arr   = np.vstack( ( Jp, Jr ) )



# The main simulation
while data.time <= T:

    # Calculate the Jacobian matrices
    mujoco.mj_jacSite( model, data, Jp, Jr, id_EE )    
    J_arr = np.vstack( ( Jp, Jr ) )

    # The end-effector velocity
    dp = Jp @ dq

    # The current and desired orientation in quaternion formulation
    quat_curr = SO3_to_quat( np.copy( Rsb ).reshape( 3, -1 )  )
    quat_des  = quat_traj[ :, n_sim ]

    # The xp and xrequation
    xp = dp_arr[ :, n_sim ] + Gp @ ( p_arr[ :, n_sim ] - p )
    xr = w_traj[ :, n_sim ] + Gr @ ( quat_curr[ 0 ] * quat_des[ 1: ] - quat_des[ 0 ] * quat_curr[ 1: ] + R3_to_so3( quat_curr[ 1: ] ) @ quat_des[ 1: ] )

    # Pseudo inverse
    x_des_new = np.linalg.pinv( J_arr ) @ np.hstack( ( xp, xr ) )

    # Again, dx is numerically calculated, since the analytical form is already computationally heavy to derive.
    # dx already does the job for small enough dt [Moses C. Nah]
    dx = ( x_des_new - x_des_old ) / dt if n_sim != 0 else np.zeros( nq )

    # Calculating the torque command
    
    # The Mass matrix of the robot
    Mmat = np.zeros( ( nq, nq ) )
    mujoco.mj_fullM( model, Mmat, data.qM )

    # The Torque input
    tau_input = Mmat @ ( dx + Gq @ ( x_des_new - dq ) ) + data.qfrc_bias

    # Torque command to the robot
    data.ctrl[ : ] = tau_input 

    # Update Simulation
    mujoco.mj_step( model, data )    

    # Update Visualization
    if ( ( n_frames != ( data.time // t_update ) ) and is_view ):
        n_frames += 1
        viewer.render( )
        print( "[Time] %6.3f" % data.time )


    # Save Data
    if ( ( n_saves != ( data.time // t_save ) ) and is_save ):
        n_saves += 1

        t_mat.append(   np.copy( data.time ) )
        q_mat.append(   np.copy(  q  ) )
        dq_mat.append(  np.copy( dq  ) )
        p_mat.append(   np.copy(  p  ) )
        dp_mat.append(  np.copy( dp  ) )
        R_mat.append(    np.copy( Rsb ).reshape( 3, -1 ) ) 

        # For this, one needs to also save the link positions
        # Also save the robot's link position and rotation matrices 
        p_tmp = []
        R_tmp = []
        for i in range( 7 ):
            p_tmp.append( np.copy( p_ref[ i ] ) )
            R_tmp.append( np.copy( R_ref[ i ] ).reshape( 3, -1 ) )

        # Also save the robot's link position and rotation matrices 
        p_links_save.append( p_tmp )
        R_links_save.append( R_tmp )     

    n_sim += 1
    x_des_old = x_des_new

# ========================================================================================== #
# [Section #5] Save and Close
# ========================================================================================== #
# Saving the data
if is_save:
    # To many parameters to save, but just saving some of the variables.
    data_dic = { "t_arr": t_mat, "q_arr": q_mat, "p_arr": p_mat, "R_arr": R_mat, "Gr": Gr, "Gp": Gp, "Gq":Gq, "quat_des": quat_traj, 
                "dquat_des": dquat_traj, "w_des": w_traj, "err_arr": e_arr, "derr_arr": de_arr,  "dderr_arr": dde_arr, "t_des": td, 
                "dp_arr": dp_mat, "dq_arr": dq_mat,  "p_links": p_links_save, "R_links": R_links_save }
    savemat( CURRENT_PATH + "/data/DMP.mat", data_dic )

if is_view:            
    viewer.close( )