# ========================================================================================== #
#  [Script Name]: DMP_task.py under example5_rhythmic
#       [Author]: Moses Chong-ook Nah
#      [Contact]: mosesnah@mit.edu
# [Date Created]: 2024.03.18
#  [Description]: Simulation using Dynamic Movement Primitives (DMP)
#                 Rhythmic movement in task-space
#   
#                 This .py file is for running/generating Figure 10 of the 
#                 following manuscript from Nah, Lachner and Hogan
#                 "Robot Control based on Motor Primitives — A Comparison of Two Approaches" 
#
#                 The code is detailed with comments, but not at the level of examples 1, 2, 3 
#                 since after example 4 will be "advanced" application of DMP.
#                 Meaning, we now often skip the details for explanation and assume 
#                 the knowledge presented at example 1, 2, 3. 
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
from utils.inverse_dynamics_models import get2DOF_M, get2DOF_C, get2DOF_IK, get2DOF_J, get2DOF_dJ, get2DOF_IK

# Importing the modules for Dynamic Movement Primitives (DMP
from DMPmodules.canonical_system        import CanonicalSystem 
from DMPmodules.nonlinear_forcing_term  import NonlinearForcingTerm
from DMPmodules.transformation_system   import TransformationSystem

# Set numpy print options
np.set_printoptions( precision = 4, threshold = 9, suppress = True )

# ========================================================================================== #
# [Section #2] Basic MuJoCo Setups
# ========================================================================================== #

# Basic MuJoCo setups
dir_name   = ROOT_PATH + '/models/'
robot_name = '2DOF_planar_torque.xml'
model      = mujoco.MjModel.from_xml_path( dir_name + robot_name )
data       = mujoco.MjData( model )
viewer     = mujoco_viewer.MujocoViewer( model, data, hide_menus = True )

# Parameters for the main simulation
T        = 12.                       # Total simulation time
dt       = model.opt.timestep       # Time-step for the simulation (set in xml file)
fps      = 30                       # Frames per second, for visualization
save_ps  = 1000                     # Hz for saving the data
n_frames = 0                        # The number of current frame of the simulation
n_saves  = 0                        # The number of saved data
speed    = 1.0                      # The speed of the simulator
t_update = 1./fps     * speed       # Time for update 
t_save   = 1./save_ps * speed       # Time for saving the data
nq       = model.nq                 # The number of degrees of freedom of the robot.         

# The time-step defined in the xml file should be smaller than the update rates
assert( dt <= min( t_update, t_save ) )

# ========================================================================================== #
# [Section #3] Imitation Learning of Dynamic Movement Primitives
# ========================================================================================== #

# Defining the parameters of the trajectory which we aim to imitate.
w0     = np.pi
Tp     = 2 * np.pi / w0     # Period of the System
tau    = Tp/(2*np.pi)
alphas = 1.0
N      = 50
P      = 100
az     = 10.0
bz     =  2.5

# Defining the Three elements of DMP
sr        = CanonicalSystem( 'rhythmic', tau, alphas )
fd        = NonlinearForcingTerm( sr, N )
trans_sys = TransformationSystem( az, bz, sr )

# Imitation Learning, getting the P sample points of position, velocity and acceleration
# Suffix "d" is added for demonstration
td   = np.linspace( 0, Tp, num = P )  # Time 
pd   = np.zeros( ( 2, P ) )           # Demonstrated position
dpd  = np.zeros( ( 2, P ) )           # Demonstrated velocity
ddpd = np.zeros( ( 2, P ) )           # Demonstrated acceleration

# Radius, angular velocity and center location
r  = 0.5
w0 = np.pi
c  = np.array( [ 0., np.sqrt( 2 ) ] )

for i, t in enumerate( td ):

    pd[   :, i ] =             r * np.array( [ -np.sin( w0*t ),  np.cos( w0*t ) ] ) + c
    dpd[  :, i ] =        w0 * r * np.array( [ -np.cos( w0*t ), -np.sin( w0*t ) ] )
    ddpd[ :, i ] = ( w0**2 ) * r * np.array( [  np.sin( w0*t ), -np.cos( w0*t ) ] )

# Method1: Locally Weighted Regression
W_LWR = np.zeros( ( 2, N ) )

# Iterating for each weight
for i in range( 2 ):
    for j in range( N ):
        a_arr = 1
        b_arr = trans_sys.get_desired( pd[ i, : ], dpd[ i, : ], ddpd[ i, : ], c[ i ] )
        phi_arr = fd.calc_ith( td, j ) 
        
        # Element-wise multiplication and summation
        W_LWR[ i, j ] = np.sum( a_arr * b_arr * phi_arr ) / np.sum( a_arr * a_arr * phi_arr )

W_LWR = np.nan_to_num( W_LWR )

# Method2: Linear Least-square Regressions
A_mat = np.zeros( ( N, P ) )
B_mat = trans_sys.get_desired( pd, dpd, ddpd, c )

# Interating along the time sample points
for i, t in enumerate( td ):
    A_mat[ :, i ] = np.squeeze( fd.calc_multiple_ith( t, np.arange( N, dtype=int ) ) ) / fd.calc_whole_at_t( t )

# Weight with Linear Least-square Multiplication
W_LLS = B_mat @ A_mat.T @ np.linalg.inv( A_mat @ A_mat.T )

# Rollout
weight = W_LLS  # W_LWR

# The actual time array for the rollout
t_arr = np.arange( 0, T, dt )
t_arr = np.append( t_arr, T )   # Append the final value 

# The conditions of the robot
y0 =  pd[ :, 0 ]
z0 = dpd[ :, 0 ]*tau
g  = c

# The forct input
input_arr = fd.calc_forcing_term( t_arr[:-1], weight, 0, np.eye( nq ), trimmed = False )
p_arr, _, dp_arr, dz_arr = trans_sys.rollout( y0, z0, g, input_arr, 0, t_arr )

# Acceleration
ddp_arr = dz_arr/sr.tau

# ========================================================================================== #
# [Section #4] The Main Simulation Loop
# ========================================================================================== #

# Save the references for the q and dq 
q  = data.qpos[ 0:nq ]
dq = data.qvel[ 0:nq ]

# The data for mat save
t_mat   = [ ]
q_mat   = [ ] 
q0_mat  = [ ] 
dq_mat  = [ ] 
dq0_mat = [ ] 

# Flags
is_save = True      # To save the data
is_view = True      # To view the simulation

# Get the initial joint position and velocity 
q_init  = get2DOF_IK( pd[ :, 0 ] )
dq_init = np.linalg.inv( get2DOF_J( q_init ) ) @ dpd[ :, 0 ]

data.qpos[ 0:nq ] =  q_init
data.qvel[ 0:nq ] = dq_init
mujoco.mj_forward( model, data )

n_sim = 0

while data.time <= T:

    # Inverse Kinematics for Position
    q_arr  = get2DOF_IK( p_arr[ :, n_sim ] )

    # Inverse Kinematics for Velocity
    tmp_Jp = get2DOF_J( q_arr )
    tmp_Jinv = np.linalg.inv( tmp_Jp )
    dq_arr = tmp_Jinv @ dp_arr[ :, n_sim ]

    # Inverse Kinematics for Acceleration
    tmp_dJp = get2DOF_dJ( q_arr, dq_arr )
    ddq_arr = tmp_Jinv @ ( ddp_arr[ :, n_sim ] - tmp_dJp @ dq_arr )

    # Torque input to the robot
    tmpM = get2DOF_M( q_arr )
    tmpC = get2DOF_C( q_arr, dq_arr )
    tau_input = tmpM @ ddq_arr + tmpC @ dq_arr

    # Torque command to the robot
    data.ctrl[ : ] = tau_input

    # Update Simulation
    mujoco.mj_step( model, data )

    # Update Visualization
    if ( ( n_frames != ( data.time // t_update ) ) and is_view ):
        n_frames += 1
        viewer.render( )
        print( "[Time] %6.3f" % data.time )

    # Saving the data
    if ( ( n_saves != ( data.time // t_save ) ) and is_save ):
        n_saves += 1

        t_mat.append(   np.copy( data.time ) )
        q_mat.append(   np.copy( q   ) )
        dq_mat.append(  np.copy( dq  ) )

    n_sim += 1

# ========================================================================================== #
# [Section #5] Save and Close
# ========================================================================================== #
# Save data as mat file for MATLAB visualization
# Saved under ./data directory
if is_save:
    data_dic = { "t_arr": t_mat, "q_arr": q_mat, "dq_arr": dq_mat, "t_des": t_arr, "alphas": alphas, "az":az, "bz":bz, "N": N, "P": P }
    savemat( CURRENT_PATH + "/data/DMP_task.mat", data_dic )
    # Substitute . in float as p for readability.

if is_view:            
    viewer.close( )