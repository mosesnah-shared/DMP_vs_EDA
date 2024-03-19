import sys
import numpy as np
import mujoco
import mujoco_viewer
from scipy.io import savemat
from pathlib  import Path

import matplotlib.pyplot as plt

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

# The time-step defined in the xml file should be smaller than the update rates
assert( dt <= min( t_update, t_save ) )

# Setting the initial position of the robot.
nq     = model.nq

# Defining the parameters of the trajectory which we aim to imitate.
# The period of the system
w0 = np.pi
Tp = 2 * np.pi / w0     # Period of the System
P  = 100                # Number of Sample Points


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

# Imitation Learning, getting the P sample points of position, velocity and acceleration
# Suffix "d" is added for demonstration
td   = np.linspace( 0, Tp, num = P )  # Time 
pd   = np.zeros( ( 2, P ) )         # Demonstrated position
dpd  = np.zeros( ( 2, P ) )         # Demonstrated velocity
ddpd = np.zeros( ( 2, P ) )         # Demonstrated acceleration

r, omega0, c = 0.5, np.pi, np.array( [ 0., np.sqrt( 2 ) ] )

for i, t in enumerate( td ):

    pd[   :, i ] =             r * np.array( [ -np.sin( w0*t ),  np.cos( w0*t ) ] ) + c
    dpd[  :, i ] =        w0 * r * np.array( [ -np.cos( w0*t ), -np.sin( w0*t ) ] )
    ddpd[ :, i ] = ( w0**2 ) * r * np.array( [  np.sin( w0*t ), -np.cos( w0*t ) ] )


# Two ways to learn the weights
# ========================================================================================== #    
# Method1: Locally Weighted Regression
#          Discussed in the original paper of Ijspeert et al. 2013
# ========================================================================================== #
# Compared to Method 2, each weight must be learned iteratively
W_LWR = np.zeros( ( 2, N ) )

# Iterating for each weight
for i in range( 2 ):
    for j in range( N ):
        a_arr = 1
        b_arr = trans_sys.get_desired( pd[ i, : ], dpd[ i, : ], ddpd[ i, : ], c[ i ] )
        phi_arr = fd.calc_ith( td, j ) 
        
        # Element-wise multiplication and summation
        W_LWR[ i, j ] = np.sum( a_arr * b_arr * phi_arr ) / np.sum( a_arr * a_arr * phi_arr )

# ========================================================================================== #
# Method2: Linear Least-square Regressions
#          Compared to Method 1, the accuracy is better, especially for learning on SO(3)
# The A, B matrices
# ========================================================================================== #        
A_mat = np.zeros( ( N, P ) )
B_mat = trans_sys.get_desired( pd, dpd, ddpd, c )

# Interating along the time sample points
for i, t in enumerate( td ):
    A_mat[ :, i ] = np.squeeze( fd.calc_multiple_ith( t, np.arange( N, dtype=int ) ) ) / fd.calc_whole_at_t( t )

# Weight with Linear Least-square Multiplication
W_LLS = B_mat @ A_mat.T @ np.linalg.inv( A_mat @ A_mat.T )

# ========================================================================================== #        
# ROLLOUT of DMP
# ========================================================================================== #        
# One can choose either one of the weights. 
weight = W_LLS  # W_LWR

# Rolling out the trajectory, to get the position, velocity and acceleration array

# The actual time array for the rollout
t_arr = np.arange( 0, T, dt )
t_arr = np.append( t_arr, T )   # Append the final value 

# The initial conditions and goal location that we aim to generate
y0 =  pd[ :, 0 ]
z0 = dpd[ :, 0 ]*tau
g  = c

# Assuming fs.calc_forcing_term and trans_sys.rollout are properly defined Python methods
input_arr = fd.calc_forcing_term( t_arr[:-1], weight, 0, np.eye( nq ), trimmed = False )
p_arr, _, dp_arr, dz_arr = trans_sys.rollout( y0, z0, g, input_arr, 0, t_arr )

# dz_arr can be used to derived acceleration
ddp_arr = dz_arr/sr.tau

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



# ========================================================================================== #        
# The Main Simulation

q_init  = get2DOF_IK( pd[ :, 0 ] )
dq_init = np.linalg.inv( get2DOF_J( q_init ) ) @ dpd[ :, 0 ]

data.qpos[ 0:nq ] =  q_init
data.qvel[ 0:nq ] = dq_init
mujoco.mj_forward( model, data )

q_arr = q_init

n_sim = 0

while data.time <= T:

    # Step 2: For joint-velocity 
    # Inverse of Jacobian.
    # Getting the Jacobian matrices, translational part

    q_arr  = get2DOF_IK( p_arr[ :, n_sim ] )

    tmp_Jp = get2DOF_J( q_arr )
    tmp_Jinv = np.linalg.inv( tmp_Jp )

    dq_arr = tmp_Jinv @ dp_arr[ :, n_sim ]

    # Step 3: For joint-acceleration 
    tmp_dJp = get2DOF_dJ( q_arr, dq_arr )
    ddq_arr = tmp_Jinv @ ( ddp_arr[ :, n_sim ] - tmp_dJp @ dq_arr )

    # Input 
    # to the Inverse Dynamics Model
    # Get the mass, coriolis matrices
    tmpM = get2DOF_M( q_arr )
    tmpC = get2DOF_C( q_arr, dq_arr )
    tau_input = tmpM @ ddq_arr + tmpC @ dq_arr

    # Adding the Torque as an input
    data.ctrl[ : ] = tau_input

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

    # Running the first-step for the Simulation
    mujoco.mj_step( model, data )
    n_sim += 1

# Save data as mat file for MATLAB visualization
# Saved under ./data directory
if is_save:
    data_dic = { "t_arr": t_mat, "q_arr": q_mat, "dq_arr": dq_mat, "t_des": t_arr, "alphas": alphas, "az":az, "bz":bz, "N": N, "P": P }
    savemat( CURRENT_PATH + "/data/DMP_task.mat", data_dic )
    # Substitute . in float as p for readability.

if is_view:            
    viewer.close( )