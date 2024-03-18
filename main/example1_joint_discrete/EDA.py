import sys
import numpy as np
import mujoco
import mujoco_viewer
from scipy.io import savemat
from pathlib  import Path

# Add and save the ROOT_PATH to run this code, which is three levels above.
ROOT_PATH = str( Path( __file__ ).parent.parent.parent )
sys.path.append( ROOT_PATH )

# Also save the CURRENT PATH of this File, for saving the data
CURRENT_PATH = str( Path( __file__ ).parent )

# Importing the Minimum-jerk Trajectory Function
from utils.trajectory  import min_jerk_traj

# Set numpy print options
np.set_printoptions( precision = 4, threshold = 9, suppress = True )

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

# Setting the initial configuration of the robot
# For this simulation, it is fully stretched to the right
nq     = model.nq
q_init = np.zeros( nq )
data.qpos[ 0:nq ] = q_init
mujoco.mj_forward( model, data )

# The joint-impedances of the 2-DOF robot 
# Can try any values that you want!
kq = 150
bq =  50
Kq = kq * np.eye( nq )
Bq = bq * np.eye( nq )

# The parameters of the minimum-jerk trajectory.
t0   = 0.3                          # Starting time
D    = 1.0                          # Duration of the movement
qdel = np.array( [ 1.0, 1.0 ] )     # q_delta
qi   = q_init                       # Initial (virtual) joint posture
qf   = q_init + qdel                #   Final (virtual) joint posture

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

# The main simulation loop
while data.time <= T:

    # Get the virtual joint trajectory's position and velocity. 
    q0, dq0, _ = min_jerk_traj( data.time, t0, t0 + D, qi, qf )

    # Module 1: First-order Joint-space Impedance Controller
    tau_imp = Kq @ ( q0 - q ) + Bq @ ( dq0 - dq )

    # Adding the Torque as an input
    data.ctrl[ : ] = tau_imp

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
        q0_mat.append(  np.copy( q0  ) )
        dq_mat.append(  np.copy( dq  ) )
        dq0_mat.append( np.copy( dq0 ) )

    # Running the first-step for the Simulation
    mujoco.mj_step( model, data )

# Save data as mat file for MATLAB visualization
# Saved under ./data directory
if is_save:
    data_dic = { "t_arr": t_mat, "q_arr": q_mat, "q0_arr": q0_mat, "dq_arr": dq_mat, "dq0_arr": dq0_mat, "Kq": Kq, "Bq": Bq }
    savemat( CURRENT_PATH + "/data/EDA_Kq" + f"{kq}".replace('.', 'p') + "_Bq" + f"{bq}".replace('.', 'p') + ".mat", data_dic )
    # Substitute . in float as p for readability.

if is_view:            
    viewer.close( )