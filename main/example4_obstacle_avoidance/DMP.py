# ========================================================================================== #
#  [Script Name]: EDA.py under example4_obstacle_avoidance
#       [Author]: Moses Chong-ook Nah
#      [Contact]: mosesnah@mit.edu
# [Date Created]: 2024.03.18
#  [Description]: Simulation using Dynamic Movement Primitives (DMP)
#                 Discrete movement in task-space, with obstacle avoidance
#   
#                 This .py file is for running/generating Figure 9 of the 
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
from utils.trajectory              import min_jerk_traj
from utils.inverse_dynamics_models import get2DOF_M, get2DOF_C, get2DOF_IK, get2DOF_J, get2DOF_dJ

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
id_EE = model.site( EE_site ).id

# Saving the references 
p  = data.site_xpos[ id_EE ]
Jp = np.zeros( ( 3, nq ) )
Jr = np.zeros( ( 3, nq ) )

# ========================================================================================== #
# [Section #3] Imitation Learning of Dynamic Movement Primitives
# ========================================================================================== #

# The parameters of the minimum-jerk trajectory.
t0   = 0.    
pdel = np.array( [ 0.0, 1.2, 0.0 ] )
pi   = np.copy( p )
pf   = pi + pdel
D    = 1.0

# Only accounting for the XY coordinates
pi = pi[ :2 ]
pf = pf[ :2 ]

# Parameters of the DMP
tau    = D  
alphas = 1.0
N      = 50
P      = 100
az     = 10.0
bz     =  2.5
y0d    = pi
gd     = pf

# Defining the Three elements of DMP
sd        = CanonicalSystem( 'discrete', D, alphas )
fd        = NonlinearForcingTerm( sd, N )
trans_sys = TransformationSystem( az, bz, sd )

# Imitation Learning, getting the P sample points of position, velocity and acceleration
# Suffix "d" is added for demonstration
td   = np.linspace( 0, D, num = P )  # Time 
qd   = np.zeros( ( 2, P ) )         # Demonstrated position
dqd  = np.zeros( ( 2, P ) )         # Demonstrated velocity
ddqd = np.zeros( ( 2, P ) )         # Demonstrated acceleration

for i, t in enumerate( td ):
    q_tmp, dq_tmp, ddq_tmp = min_jerk_traj( t, 0, D, pi, pf )

    qd[   :, i ] =   q_tmp
    dqd[  :, i ] =  dq_tmp
    ddqd[ :, i ] = ddq_tmp

# Method1: Locally Weighted Regression
W_LWR = np.zeros( ( nq, N ) )

# Iterating for each weight
for i in range( 2 ): # For XY coordinates
    for j in range( N ):
        a_arr = sd.calc( td ) * ( gd[ i ] - y0d[ i ] )
        b_arr = trans_sys.get_desired( qd[ i, : ], dqd[ i, : ], ddqd[ i, : ], gd[ i ] )
        phi_arr = fd.calc_ith( td, j ) 
        
        # Element-wise multiplication and summation
        W_LWR[ i, j ] = np.sum( a_arr * b_arr * phi_arr ) / np.sum( a_arr * a_arr * phi_arr )


# Method2: Linear Least-square Regressions
A_mat = np.zeros( ( N, P ) )
B_mat = trans_sys.get_desired( qd, dqd, ddqd, gd )

# Iterating along the time sample points
for i, t in enumerate( td ):
    A_mat[ :, i ] = np.squeeze( fd.calc_multiple_ith( t, np.arange( N, dtype=int ) ) ) / fd.calc_whole_at_t( t ) * sd.calc( t )

# Weight with Linear Least-square Multiplication
W_LLS = B_mat @ A_mat.T @ np.linalg.inv( A_mat @ A_mat.T )

# Choosing the weight array that we use for the main simulation
weight = W_LLS  # W_LWR

# The actual time array for the rollout
t_arr = np.arange( 0, T, dt )
t_arr = np.append( t_arr, T )   # Append the final value 

# The initial conditions and goal location that we aim to generate
z0 = np.zeros( nq )
y0 = y0d
g  = gd

# The nonlinear forcing term trajectory 
input_arr_discrete = fd.calc_forcing_term( t_arr[:-1], weight, t0, np.diag( g - y0 ), trimmed = True )

# Save the references for the q and dq 
q  = data.qpos[ 0:nq ]
dq = data.qvel[ 0:nq ]

# The obstacle location
o = np.array( [ -0.001, 0.5 * ( pi[ 1 ] + pf[ 1 ] ), 0 ] )

# The data for mat save
t_mat   = [ ]
q_mat   = [ ] 
dq_mat  = [ ] 
p_mat   = [ ]
dp_mat  = [ ]
p_des   = [ ]
dp_des  = [ ]
ddp_des = [ ] 

# Other Flags
is_save = True     # To save the data
is_view = True     # To view the simulation

# ========================================================================================== #
# [Section #4] The Main Simulation Loop
# ========================================================================================== #

n_sim = 0

# To conduct DMP rollout, define the initial conditions
y_old = y0
z_old = np.zeros( 2 )

while data.time <= T:

    # Adding the coupling term for the rollout
    mujoco.mj_jacSite( model, data, Jp, Jr, id_EE )
    dp = Jp @ dq

    # Parameters for the coupling term
    theta = np.arccos( np.inner( o - p, dp  ) / ( np.linalg.norm( o - p ) * np.linalg.norm( dp )  )   )
    R     = np.array( [ [ 0, 1, 0 ], [ -1, 0, 0 ], [0 ,0, 1 ] ] )            

    # The Coupling term for DMP
    # Equation (19) of the main manuscript
    Cp = 300 * R @ dp * np.exp( -3 * theta ) if np.sum( dp ) != 0 else np.zeros( 3 )

    # If it is the first step for simulation, just use the coupling term
    if n_sim == 0:
        y_new, z_new, dy, dz = trans_sys.step( y_old, z_old, g, Cp[ :2 ], dt )
    else:
        y_new, z_new, dy, dz = trans_sys.step( y_old, z_old, g, input_arr_discrete[ :, n_sim-1] + Cp[ :2 ], dt )

    # The position, velocity and acceleration of task-space 
    p_arr   = y_new
    dp_arr  = dy
    ddp_arr = dz/tau

    # Step 1: Inverse Kinematics for joint position
    q_arr  = get2DOF_IK( p_arr )

    # Step 2: Inverse Kinematics for joint velocity
    tmp_Jp   = get2DOF_J( q_arr )
    tmp_Jinv = np.linalg.inv( tmp_Jp ) 
    dq_arr = tmp_Jinv @ dp_arr

    # Step 3: For joint-acceleration 
    tmp_dJp = get2DOF_dJ( q_arr, dq_arr )
    ddq_arr = tmp_Jinv @ ( ddp_arr - tmp_dJp @ dq_arr )

    # Input to the Inverse Dynamics model
    tmpM = get2DOF_M( q_arr )
    tmpC = get2DOF_C( q_arr, dq_arr )
    tau_input = tmpM @ ddq_arr + tmpC @ dq_arr

    # Adding the Torque as an input
    data.ctrl[ : ] = tau_input 

    # Running the first-step for the Simulation
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

        p_mat.append( np.copy( p ) )
        dp_mat.append(    dp    )
        p_des.append(     p_arr )
        dp_des.append(   dp_arr )
        ddp_des.append( ddp_arr )


    # Update the simulation loop
    n_sim += 1
    y_old = y_new
    z_old = z_new

# ========================================================================================== #
# [Section #5] Save and Close
# ========================================================================================== #
# Save data as mat file for MATLAB visualization
# Saved under ./data directory
if is_save:
    data_dic = { "t_arr": t_mat, "q_arr": q_mat, "dq_arr": dq_mat, "p_des": p_arr, "dp_des": dp_arr, "ddp_des": ddp_arr, "t_des": t_arr, "alphas": alphas, "az":az, "bz":bz, "N": N, "P": P }
    
    savemat( CURRENT_PATH + "/data/DMP.mat", data_dic )         
        

if is_view:            
    viewer.close( )