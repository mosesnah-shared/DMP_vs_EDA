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
from utils.inverse_dynamics_models import get2DOF_M, get2DOF_C

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

# Movement Parameters
q_init  = np.array( [ 0.5, 0.5 ] )
qa_init = np.array( [ 0.1, 0.3 ] )
dq_init = w0 * qa_init 

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
qd   = np.zeros( ( nq, P ) )         # Demonstrated position
dqd  = np.zeros( ( nq, P ) )         # Demonstrated velocity
ddqd = np.zeros( ( nq, P ) )         # Demonstrated acceleration

for i, t in enumerate( td ):
    qd[   :, i ] =       q_init + qa_init * np.sin( w0*t  )
    dqd[  :, i ] =           qa_init * w0 * np.cos( w0*t  )
    ddqd[ :, i ] = -qa_init * ( w0 ** 2 ) * np.sin( w0*t  )

# Two ways to learn the weights
# ========================================================================================== #    
# Method1: Locally Weighted Regression
#          Discussed in the original paper of Ijspeert et al. 2013
# ========================================================================================== #
# Compared to Method 2, each weight must be learned iteratively
W_LWR = np.zeros( ( nq, N ) )

# Iterating for each weight
for i in range( nq ):
    for j in range( N ):
        a_arr = 1
        b_arr = trans_sys.get_desired( qd[ i, : ], dqd[ i, : ], ddqd[ i, : ], q_init[ i ] )
        phi_arr = fd.calc_ith( td, j ) 
        
        # Element-wise multiplication and summation
        W_LWR[ i, j ] = np.sum( a_arr * b_arr * phi_arr ) / np.sum( a_arr * a_arr * phi_arr )

# ========================================================================================== #
# Method2: Linear Least-square Regressions
#          Compared to Method 1, the accuracy is better, especially for learning on SO(3)
# The A, B matrices
# ========================================================================================== #        
A_mat = np.zeros( ( N, P ) )
B_mat = trans_sys.get_desired( qd, dqd, ddqd, q_init )

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
y0 = q_init
z0 = dq_init*tau
g  = q_init

# Assuming fs.calc_forcing_term and trans_sys.rollout are properly defined Python methods
input_arr = fd.calc_forcing_term( t_arr[:-1], weight, 0, np.eye( nq ), trimmed = False )
y_arr, _, dy_arr, dz_arr = trans_sys.rollout( y0, z0, g, input_arr, 0, t_arr )

# dz_arr can be used to derived acceleration
ddy_arr = dz_arr/sr.tau

# We now know the y_arr, dy_arr and ddy_arr for the inverse dynamics model 

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

data.qpos[ 0:nq ] = q_init
data.qvel[ 0:nq ] = dq_init
mujoco.mj_forward( model, data )


n_sim = 0

while data.time <= T:

    # Input to the Inverse Dynamics Model
    # Get the mass, coriolis matrices
    tmpM = get2DOF_M( y_arr[ :, n_sim ] )
    tmpC = get2DOF_C( y_arr[ :, n_sim ], dy_arr[ :, n_sim ] )
    tau_input = tmpM @ ddy_arr[ :, n_sim ] + tmpC @ dy_arr[ :, n_sim ]

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
    data_dic = { "t_arr": t_mat, "q_arr": q_mat, "dq_arr": dq_mat, "t_des": t_arr, "q_des": y_arr, "dq_des": dy_arr, "ddq_des": ddy_arr, "alphas": alphas, "az":az, "bz":bz, "N": N, "P": P }
    savemat( CURRENT_PATH + "/data/DMP_joint.mat", data_dic )
    # Substitute . in float as p for readability.

if is_view:            
    viewer.close( )