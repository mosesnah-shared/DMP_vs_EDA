# ========================================================================================== #
#  [Script Name]: DMP.py under example2_task_discrete
#       [Author]: Moses Chong-ook Nah
#      [Contact]: mosesnah@mit.edu
# [Date Created]: 2024.03.18
#  [Description]: Simulation using Dynamic Movement Primitives (DMP)
#                 Discrete movement in task-space
#                 This .py file is for running/generating Figure 5 of the 
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

# Importing the Minimum-jerk Trajectory and the Inverse Dynamics Model/Inverse Kinematics
from utils.trajectory              import min_jerk_traj
from utils.inverse_dynamics_models import get2DOF_M, get2DOF_C, get2DOF_IK, get2DOF_J, get2DOF_dJ

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
dir_name   = ROOT_PATH + '/models/'
robot_name = '2DOF_planar_torque.xml'
model      = mujoco.MjModel.from_xml_path( dir_name + robot_name )
data       = mujoco.MjData( model )
viewer     = mujoco_viewer.MujocoViewer( model, data, hide_menus = True )

# Parameters for the main simulation
T        = 4.                       # Total simulation time
dt       = model.opt.timestep       # Time-step for the simulation (set in xml file)
fps      = 30                       # Frames per second, for visualization
save_ps  = 1000                     # Hz for saving the data
n_frames = 0                        # The number of current frame of the simulation
n_saves  = 0                        # The number of saved data
speed    = 1.0                      # The speed of the simulator

t_update = 1./fps     * speed       # Time for update 
t_save   = 1./save_ps * speed       # Time for saving the data

# The time-step defined in the xml file should be smaller than the update rates
assert( dt <= min( t_update, t_save ) )

# Setting the initial position of the robot.
nq     = model.nq
q1     = np.pi * 1/12
q_init = np.array( [ q1, np.pi-2*q1 ] )
data.qpos[ 0:nq ] = q_init
mujoco.mj_forward( model, data )

# Save the references for the q and dq 
q  = data.qpos[ 0:nq ]
dq = data.qvel[ 0:nq ]

# Get the end-effector's ID and its position, translational and rotational Jacobian matrices
EE_site = "site_end_effector"
id_EE   = model.site( EE_site ).id
p       = data.site_xpos[ id_EE ]
Jp      = np.zeros( ( 3, nq ) )
Jr      = np.zeros( ( 3, nq ) )

# ========================================================================================== #
# [Section #3] Imitation Learning of Dynamic Movement Primitives
# ========================================================================================== #

# The parameters of the first minimum-jerk trajectory.
t0   = 0.5       
pdel = np.array( [ -0.7, 0.7, 0.0 ] )
pi   = np.copy( p )
pf   = pi + pdel
D    = 1.0

# Only accounting for the XY coordinates of the initial/final positions for Imitation Learning
pi = pi[ :2 ]
pf = pf[ :2 ]

# Parameters of the DMP
tau    = D                    # Duration of the demonstrated trajectory
alphas = 1.0                  # Decay rate of the Canonical System
N      = 50                   # Number of Basis Functions
P      = 100                  # Number of Sample points for the demonstrated trajectory
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
td   = np.linspace( 0, D, num = P )  # Time 
pd   = np.zeros( ( 2, P ) )         # Demonstrated position
dpd  = np.zeros( ( 2, P ) )         # Demonstrated velocity
ddpd = np.zeros( ( 2, P ) )         # Demonstrated acceleration

for i, t in enumerate( td ):
    q_tmp, dq_tmp, ddq_tmp = min_jerk_traj( t, 0, D, pi, pf )

    pd[   :, i ] =   q_tmp
    dpd[  :, i ] =  dq_tmp
    ddpd[ :, i ] = ddq_tmp

# Method1: Locally Weighted Regression (LWR)
W_LWR = np.zeros( ( 2, N ) )

# Iterating for each weight
for i in range( 2 ): # For XY coordinates
    for j in range( N ):
        a_arr = sd.calc( td ) * ( gd[ i ] - y0d[ i ] )
        b_arr = trans_sys.get_desired( pd[ i, : ], dpd[ i, : ], ddpd[ i, : ], gd[ i ] )
        phi_arr = fd.calc_ith( td, j ) 
        
        # Element-wise multiplication and summation
        W_LWR[ i, j ] = np.sum( a_arr * b_arr * phi_arr ) / np.sum( a_arr * a_arr * phi_arr )

# Method2: Linear Least-square Regression (LLS)
A_mat = np.zeros( ( N, P ) )
B_mat = trans_sys.get_desired( pd, dpd, ddpd, gd )

# Interating along the time sample points
for i, t in enumerate( td ):
    A_mat[ :, i ] = np.squeeze( fd.calc_multiple_ith( t, np.arange( N, dtype=int ) ) ) / fd.calc_whole_at_t( t ) * sd.calc( t )

# Weight with Linear Least-square Multiplication
W_LLS = B_mat @ A_mat.T @ np.linalg.inv( A_mat @ A_mat.T )

# Rollout of the trajectory
# One can choose either the weights learned by LWR or LLS
weight = W_LLS  # W_LWR

# Rolling out the trajectory, to get the position, velocity and acceleration array
# The whole time array for the roll out
t_arr = np.arange( 0, T, dt )
t_arr = np.append( t_arr, T )   # Technical detail: Append the final to include T on the array

# The initial conditions and goal location that we aim to generate
y0 = y0d            
g  = pf             
sd.tau = tau       

# Calculate the nonlinear forcing term and do the rollout
# The "trimmed" argument artifically set the nonlinear forcing term value to zero for time over the movement duration
# Not necessary, but it clears out the "tail" of DMP at the end of the movement. 
input_arr = fd.calc_forcing_term( t_arr[:-1], weight, t0, np.diag( g - y0 ), trimmed = True )

# ========================================================================================== #
# [Section #4] Main Simulation
# ========================================================================================== #

# Save the references for the q and dq 
q  = data.qpos[ 0:nq ]
dq = data.qvel[ 0:nq ]

# The data for mat save
t_mat   = [ ]
q_mat   = [ ] 
dq_mat  = [ ] 

# Other Flags
is_save = True     # To save the data
is_view = True     # To view the simulation

# Step number for the simulation
n_sim = 0

# To conduct DMP rollout, define the initial conditions of the rhythmic transformation system
y_old =  pd[ :, 0 ]
z_old = dpd[ :, 0 ] * tau

# The new goal location
g_new = pf + np.array( [ 1.5, 0.5 ] )

while data.time <= T:

    # Update the goal location 
    if data.time <= t0:
        g = pi 
    else:    
        if data.time <= t0 + D/2:
            g = pf
        else:
            g = g_new + ( pf - g_new ) * np.exp( -sd.tau * ( data.time - ( t0 + D/2 ) ) )

    # Updating the step
    # If it is the first step for simulation, just use the coupling term
    if n_sim == 0:
        y_new, z_new, dy, dz = trans_sys.step( y_old, z_old, g, np.zeros( 2 ), dt )
    else:
        y_new, z_new, dy, dz = trans_sys.step( y_old, z_old, g, input_arr[ :, n_sim-1], dt )

    p_arr   = y_new 
    dp_arr  = dy 
    ddp_arr = dz/tau

    # Additional to the Inverse Dynamics Model, we need to also conduct the Inverse Kinematics
    # Step 1: Inverse Kinematics for joint-position
    q_arr  = get2DOF_IK( p_arr )

    # Step 2: Inverse Kinematics for joint-velocity
    # Getting the Jacobian matrices, translational part
    tmp_Jp = get2DOF_J( q_arr )
    tmp_Jinv = np.linalg.inv( tmp_Jp )

    dq_arr = tmp_Jinv @ dp_arr

    # Step 3: Inverse Kinematics for joint-acceleration
    tmp_dJp = get2DOF_dJ( q_arr, dq_arr )
    ddq_arr = tmp_Jinv @ ( ddp_arr - tmp_dJp @ dq_arr )

    # Now, input to the Inverse Dynamics Model
    tmpM = get2DOF_M( q_arr )
    tmpC = get2DOF_C( q_arr, dq_arr )

    # The equations of motion of the robot
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
    y_old = y_new 
    z_old = z_new

# ========================================================================================== #
# [Section #5] Save and Close
# ========================================================================================== #
# Save data as mat file for MATLAB visualization
# Saved under ./data directory
if is_save:
    data_dic = { "t_arr": t_mat, "q_arr": q_mat, "dq_arr": dq_mat, "t_des": t_arr, "alphas": alphas, "az":az, "bz":bz, "N": N, "P": P }
    
    savemat( CURRENT_PATH + "/data/DMP.mat", data_dic ) 
    
if is_view:            
    viewer.close( )