# ========================================================================================== #
#  [Script Name]: DMP.py under example1_joint_discrete
#       [Author]: Moses Chong-ook Nah
#      [Contact]: mosesnah@mit.edu
# [Date Created]: 2024.03.18
#  [Description]: Simulation using Dynamic Movement Primitives (DMP)
#                 Discrete movement in joint-space
#                 This .py file is for running/generating Figure 3 of the 
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

# Importing the Minimum-jerk Trajectory and the Inverse Dynamics Model
from utils.trajectory              import min_jerk_traj
from utils.inverse_dynamics_models import get2DOF_M, get2DOF_C

# Importing the modules for Dynamic Movement Primitives (DMP)
from DMPmodules.canonical_system        import CanonicalSystem 
from DMPmodules.nonlinear_forcing_term  import NonlinearForcingTerm
from DMPmodules.transformation_system   import TransformationSystem

# Set numpy print options
np.set_printoptions( precision = 4, threshold = 9, suppress = True )

# ========================================================================================== #
# [Section #2] Basic MuJoCo Setups
# ========================================================================================== #

dir_name   = ROOT_PATH + '/models/'
robot_name = '2DOF_planar_torque.xml'
model      = mujoco.MjModel.from_xml_path( dir_name + robot_name )
data       = mujoco.MjData( model )
viewer     = mujoco_viewer.MujocoViewer( model, data, hide_menus = True )

# Parameters for the main simulation
T        = 3.                       # Total simulation time
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

# Setting the initial position/velocity of the robot.
# For this demo, the robot is stretched to the right
nq     = model.nq
q_init = np.zeros( nq )
data.qpos[ 0:nq ] = q_init
mujoco.mj_forward( model, data )

# ========================================================================================== #
# [Section #3] Imitation Learning of Dynamic Movement Primitives
# ========================================================================================== #

# Defining the parameters of the minimum-jerk trajectory which we aim to imitate.
t0   = 0.3                          # Starting time of the Movement
D    = 1.0                          # Duration of the movement
qdel = np.array( [ 1.0, 1.0 ] )     # q_delta
qi   = q_init                       # Initial (virtual) joint posture
qf   = q_init + qdel                #   Final (virtual) joint posture

# Parameters of the DMP
tau    = D                          # Duration of the demonstrated trajectory
alphas = 1.0                        # Decay rate of the Canonical System
N      = 50                         # Number of Basis Functions
P      = 100                        # Number of Sample points for the demonstrated trajectory
az     = 10.0                       # Coefficient for the Transformation System
bz     =  2.5                       # Coefficient for the Transformation System
y0d    = qi                         # Initial Position of the demonstrated trajectory
gd     = qf                         #    Goal Position of the demonstrated trajectory

# Defining the Three elements of DMP
sd        = CanonicalSystem( 'discrete', D, alphas )
fd        = NonlinearForcingTerm( sd, N )
trans_sys = TransformationSystem( az, bz, sd )

# To conduct Imitation Learning, we need P sample points of position, velocity and acceleration 
# of the demonstrated trajectory
# Suffix "d" is added for demonstration
td   = np.linspace( 0, D, num = P )  # Time 
qd   = np.zeros( ( nq, P ) )         # Demonstrated position
dqd  = np.zeros( ( nq, P ) )         # Demonstrated velocity
ddqd = np.zeros( ( nq, P ) )         # Demonstrated acceleration

for i, t in enumerate( td ):
    q_tmp, dq_tmp, ddq_tmp = min_jerk_traj( t, 0, D, qi, qf )

    qd[   :, i ] =   q_tmp
    dqd[  :, i ] =  dq_tmp
    ddqd[ :, i ] = ddq_tmp

# There are two ways to learn the weights, and we present both methods
# -------------------------------------------- #    
# Method1: Locally Weighted Regression (LWR)
#          Discussed in the original paper of Ijspeert et al. 2013
# -------------------------------------------- #    
    
W_LWR = np.zeros( ( nq, N ) )

# Calculating each weight with a for loop
for i in range( nq ):
    for j in range( N ):
        a_arr = sd.calc( td ) * ( gd[ i ] - y0d[ i ] )
        b_arr = trans_sys.get_desired( qd[ i, : ], dqd[ i, : ], ddqd[ i, : ], gd[ i ] )
        phi_arr = fd.calc_ith( td, j ) 
        
        # Element-wise multiplication and summation
        W_LWR[ i, j ] = np.sum( a_arr * b_arr * phi_arr ) / np.sum( a_arr * a_arr * phi_arr )

# Substitute nan to 0 
W_LWR = np.nan_to_num( W_LWR )


# -------------------------------------------- #
# Method2: Linear Least-square Regression (LLS)
#          Compared to Method 1, the accuracy is better, especially for learning on SO(3)  
#          More details on example9_pos_and_orient
# -------------------------------------------- #        
        
A_mat = np.zeros( ( N, P ) )
B_mat = trans_sys.get_desired( qd, dqd, ddqd, gd )

# Filling-in the A matrix
for i, t in enumerate( td ):
    A_mat[ :, i ] = np.squeeze( fd.calc_multiple_ith( t, np.arange( N, dtype=int ) ) ) / fd.calc_whole_at_t( t ) * sd.calc( t )

# Weight with Linear Least-square Multiplication
W_LLS = B_mat @ A_mat.T @ np.linalg.inv( A_mat @ A_mat.T )

# Scaling down is required
# This already shows one of the limitations of this choice of DMP.
# If one of the coordinates end up with zero, despite a non-zero displacement
# The weight matrix of that coordinate is not defined.
# Hence, one can either use the alternative DMPs:
# [1] Pastor, Peter, et al. "Learning and generalization of motor skills by learning from demonstration." ICRA (2009)
# [2] Koutras, Leonidas, and Zoe Doulgeri. "A novel dmp formulation for global and frame independent spatial scaling in the task space." RO-MAN (2020) 
for i in range( nq ):
    if ( gd[ i ] - y0d[ i ] ) != 0:
        W_LLS[ i, : ] = 1./( ( gd[ i ] - y0d[ i ] ) ) * W_LLS[ i, : ]
    else:
        W_LLS[ i, : ]= 0

# Rollout of the trajectory
# One can choose either the weights learned by LWR or LLS
weight = W_LLS  # W_LWR

# Rolling out the trajectory, to get the position, velocity and acceleration array
# The whole time array for the roll out
t_arr = np.arange( 0, T, dt )
t_arr = np.append( t_arr, T )   # Technical detail: Append the final to include T on the array

# The initial conditions and goal location that we aim to generate
# Technical Detail: Since the weights are learned, we can choose "any" initial position and goal locations
#                   For the trajectory rollout. 
#                   This shows the benefit of DMP, which is the spatial/temporal invariance property>

z0 = np.zeros( nq )
y0 = y0d            # Setting the same initial position  with the demonstrated trajectory.
g  = qf             # Setting the same    goal position  with the demonstrated trajectory. [TRYOUT] g  = 2. * qf
sd.tau = tau        # Setting the same movement duration with the demonstrated trajectory. [TRYOUT] sd.tau = 2*tau

# Calculate the nonlinear forcing term and do the rollout
# The "trimmed" argument artifically set the nonlinear forcing term value to zero for time over the movement duration
# Not necessary, but it clears out the "tail" of DMP at the end of the movement. 
input_arr_discrete = fd.calc_forcing_term( t_arr[:-1], weight, t0, np.diag( g - y0 ), trimmed = True )
y_arr, _, dy_arr, dz_arr = trans_sys.rollout( y0, z0, g, input_arr_discrete, t0, t_arr )

# dz_arr can be used to derive the acceleration
ddy_arr = dz_arr/sd.tau

# ========================================================================================== #
# [Section #4] Main Simulation
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

# Step number for the simulation
n_sim = 0

while data.time <= T:

    # For torque-controlled robot, one needs the Inverse Dyamics Model of the robot
    tmpM = get2DOF_M( y_arr[ :, n_sim ] )
    tmpC = get2DOF_C( y_arr[ :, n_sim ], dy_arr[ :, n_sim ] )

    # Equations of motion of the robot
    tau_input = tmpM @ ddy_arr[ :, n_sim ] + tmpC @ dy_arr[ :, n_sim ]

    # Command the Torque input
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
        q_mat.append(   np.copy( q   ) )
        dq_mat.append(  np.copy( dq  ) )

    n_sim += 1

# ========================================================================================== #
# [Section #5] Save and Close
# ========================================================================================== #
# Save data as mat file for MATLAB visualization
# Saved under ./data local directory
if is_save:
    data_dic = { "t_arr": t_mat, "q_arr": q_mat, "dq_arr": dq_mat, "t_des": t_arr, "q_des": y_arr, "dq_des": dy_arr, "ddq_des": ddy_arr, 
                "alphas": alphas, "az":az, "bz":bz, "N": N, "P": P }
    savemat( CURRENT_PATH + "/data/DMP.mat", data_dic )
    # Substitute . in float as p for readability.

if is_view:            
    viewer.close( )