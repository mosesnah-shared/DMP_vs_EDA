# ========================================================================================== #
#  [Script Name]: DMP_joint.py under example5_rhythmic
#       [Author]: Moses Chong-ook Nah
#      [Contact]: mosesnah@mit.edu
# [Date Created]: 2024.03.18
#  [Description]: Simulation using Dynamic Movement Primitives (DMP)
#                 Rhythmic movement in joint-space
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
from utils.inverse_dynamics_models import get2DOF_M, get2DOF_C

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
T        = 12.                      # Total simulation time
dt       = model.opt.timestep       # Time-step for the simulation (set in xml file)
fps      = 30                       # Frames per second, for visualization
save_ps  = 1000                     # Hz for saving the data
n_frames = 0                        # The number of current frame of the simulation
n_saves  = 0                        # The number of saved data
speed    = 1.0                      # The speed of the simulator
t_update = 1./fps     * speed       # Time for update 
t_save   = 1./save_ps * speed       # Time for saving the data
nq       = model.nq                 # The degrees of freedom of the robot

# The time-step defined in the xml file should be smaller than the update rates
assert( dt <= min( t_update, t_save ) )

# ========================================================================================== #
# [Section #3] Imitation Learning of Dynamic Movement Primitives
# ========================================================================================== #

# The period of the system
w0 = np.pi
Tp = 2 * np.pi / w0     # Period of the System

# Parameters of the DMP
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

# Defining another transformation system for the goal trajectory g(t)
# Note that sr is passed to the transformation system, but is not used.
trans_sys_g = TransformationSystem( az, bz, sr )

# Imitation Learning, getting the P sample points of position, velocity and acceleration
# Suffix "d" is added for demonstration
td   = np.linspace( 0, Tp, num = P )  # Time 
qd   = np.zeros( ( nq, P ) )         # Demonstrated position
dqd  = np.zeros( ( nq, P ) )         # Demonstrated velocity
ddqd = np.zeros( ( nq, P ) )         # Demonstrated acceleration

# The Initial position, velocity and the amplitude of the oscillatory movement
q_init  = np.array( [ 0.5, 0.5 ] )
qa_init = np.array( [ 0.1, 0.3 ] )
dq_init = w0 * qa_init 

for i, t in enumerate( td ):
    qd[   :, i ] =       q_init + qa_init * np.sin( w0*t )
    dqd[  :, i ] =           w0 * qa_init * np.cos( w0*t )
    ddqd[ :, i ] = -( w0 ** 2 ) * qa_init * np.sin( w0*t )

# Method1: Locally Weighted Regression
W_LWR = np.zeros( ( nq, N ) )

# Iterating for each weight
for i in range( nq ):
    for j in range( N ):
        a_arr = 1
        b_arr = trans_sys.get_desired( qd[ i, : ], dqd[ i, : ], ddqd[ i, : ], q_init[ i ] )
        phi_arr = fd.calc_ith( td, j ) 
        
        # Element-wise multiplication and summation
        W_LWR[ i, j ] = np.sum( a_arr * b_arr * phi_arr ) / np.sum( a_arr * a_arr * phi_arr )

# Method2: Linear Least-square Regressions
A_mat = np.zeros( ( N, P ) )
B_mat = trans_sys.get_desired( qd, dqd, ddqd, q_init )

# Interating along the time sample points
for i, t in enumerate( td ):
    A_mat[ :, i ] = np.squeeze( fd.calc_multiple_ith( t, np.arange( N, dtype=int ) ) ) / fd.calc_whole_at_t( t )

# Weight with Linear Least-square Multiplication
W_LLS = B_mat @ A_mat.T @ np.linalg.inv( A_mat @ A_mat.T )

weight = W_LLS  # W_LWR

# Rolling out the trajectory, to get the position, velocity and acceleration array
# The actual time array for the rollout
t_arr = np.arange( 0, T, dt )
t_arr = np.append( t_arr, T )   # Append the final value 

# Calculating the Nonlinear Forcing Term, which is later used in the main loop
input_arr = fd.calc_forcing_term( t_arr[:-1], weight, 0, np.eye( nq ), trimmed = False )

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
q_des   = [ ]
dq_des  = [ ]
ddq_des = [ ]

# Flags
is_save = True      # To save the data
is_view = True      # To view the simulation

# Updating the Simulation
data.qpos[ 0:nq ] = q_init
data.qvel[ 0:nq ] = dq_init
mujoco.mj_forward( model, data )

# Step for the simulation
n_sim = 0

# To conduct DMP rollout, define the initial conditions of the rhythmic transformation system
y_old = q_init
z_old = dq_init * tau

# The parameters for the goal dynamics, defined by the transformation system
g_old  = q_init
zg_old = np.zeros( 2 )

# The step input for goal g0
g0_1 = q_init
g0_2 = q_init + np.array( [ 1.0, 1.0 ] )

while data.time <= T:

    # Defining the g0 value for the time range 
    if    0.0 <= data.time <=  3.5:
        g0 = g0_1
    elif  3.5 <  data.time <=  8.5:
        g0 = g0_2 
    elif  8.5 <  data.time <= 13.5:
        g0 = g0_1
    elif 13.5 <  data.time <= 18.5:
        g0 = g0_2
    else:
        g0 = g0_1        
        
    # Update the Goal Locations
    g_new, zg_new, _, _ = trans_sys.step( g_old, zg_old, g0, np.zeros( 2 ), dt )

    # Updating the DMP trajectories in the main loop.
    if n_sim == 0:
        y_new, z_new, dy, dz = trans_sys.step( y_old, z_old, g_new, np.zeros( 2 ), dt )
    else:
        y_new, z_new, dy, dz = trans_sys.step( y_old, z_old, g_new, input_arr[ :, n_sim-1 ], dt )

    # The joint position, velocity and acceleration
    q_arr   = y_new
    dq_arr  = dy
    ddq_arr = dz/tau        

    # Input to the Inverse Dynamics model
    tmpM = get2DOF_M( q_arr )
    tmpC = get2DOF_C( q_arr, dq_arr )
    tau_input = tmpM @ ddq_arr + tmpC @ dq_arr

    # Command to robot
    data.ctrl[ : ] = tau_input

    # Update simulation
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
        q_mat.append(   np.copy(         q ) )
        dq_mat.append(  np.copy(        dq ) )
        q_des.append(   np.copy(     q_arr ) )
        dq_des.append(  np.copy(    dq_arr ) )        
        ddq_des.append( np.copy(   ddq_arr ) )                

    n_sim += 1
    g_old  =  g_new
    zg_old = zg_new
    y_old  =  y_new
    z_old  =  z_new    

# ========================================================================================== #
# [Section #5] Save and Close
# ========================================================================================== #
# Save data as mat file for MATLAB visualization
# Saved under ./data directory
if is_save:
    data_dic = { "t_arr": t_mat, "q_arr": q_mat, "dq_arr": dq_mat, "t_des": t_arr, "q_des": q_arr, "dq_des": dq_arr, "ddq_des": ddq_arr, "alphas": alphas, "az":az, "bz":bz, "N": N, "P": P }
    savemat( CURRENT_PATH + "/data/DMP_joint.mat", data_dic )
    # Substitute . in float as p for readability.

if is_view:            
    viewer.close( )