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
from utils.trajectory  import min_jerk_traj

# Set numpy print options
np.set_printoptions( precision = 4, threshold = 9, suppress = True )

# Basic MuJoCo setups
dir_name   = ROOT_PATH + '/models/'
robot_name = '2DOF_planar_torque.xml'
model      = mujoco.MjModel.from_xml_path( dir_name + robot_name )
data       = mujoco.MjData( model )
viewer     = mujoco_viewer.MujocoViewer( model, data, hide_menus = True )

# Parameters for the simulation
T        = 10.0                     # Total Simulation Time
dt       = model.opt.timestep       # Time-step for the simulation (set in xml file)
fps      = 30                       # Frames per second
save_ps  = 1000                     # Hz for saving the data
n_frames = 0                        # The number of current frame of the simulation
n_saves  = 0                        # Update for the saving point
speed    = 1.0                      # The speed of the simulator

t_update = 1./fps     * speed       # Time for update 
t_save   = 1./save_ps * speed       # Time for saving the data

# The time-step defined in the xml file should be smaller than the update rates
assert( dt <= min( t_update, t_save ) )

# Setting the initial position of the robot.
nq  = model.nq
qi  = np.array( [ 0.5, 0.5 ] )
qf  = np.array( [ 1.5, 1.5 ] )
qA  = np.array( [ 0.1, 0.3 ] )
w0  = np.pi 
D   = 1.

q_init  = qi
dq_init = qA * w0 
data.qpos[ 0:nq ] =  q_init
data.qvel[ 0:nq ] = dq_init

mujoco.mj_forward( model, data )

# The joint-impedances of the 2-DOF robot 
# Can try any values that you want.
kq = 150
bq =  50
Kq = kq * np.eye( nq )
Bq = bq * np.eye( nq )

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

    mujoco.mj_step( model, data )

    # Get the virtual joint trajectory's position and velocity. 
    q0_sub1, dq0_sub1, _ = min_jerk_traj( data.time, 3.5, 3.5 + D, qi, qf )
    q0_sub2, dq0_sub2, _ = min_jerk_traj( data.time, 3.5 +  5.0, 3.5 +  5.0 + D, np.zeros( nq ), qi - qf )
    q0_sub3, dq0_sub3, _ = min_jerk_traj( data.time, 3.5 + 10.0, 3.5 + 10.0 + D, np.zeros( nq ), qf - qi )
    q0_sub4, dq0_sub4, _ = min_jerk_traj( data.time, 3.5 + 15.0, 3.5 + 15.0 + D, np.zeros( nq ), qf - qi )


    q0_osc  = qA * np.sin( w0 * data.time )
    dq0_osc = qA * w0 * np.cos( w0 * data.time )

    q0  =  q0_sub1 +  q0_sub2 +  q0_sub3 +  q0_osc
    dq0 = dq0_sub1 + dq0_sub2 + dq0_sub3 + dq0_osc

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

# Save Data as mat file for MATLAB visualization
if is_save:
    data_dic = { "t_arr": t_mat, "q_arr": q_mat, "q0_arr": q0_mat, "dq_arr": dq_mat, "dq0_arr": dq0_mat, "Kq": Kq, "Bq": Bq }
    savemat( CURRENT_PATH + "/data/EDA_joint_rhythmic.mat", data_dic )
    # Substitute . in float as p for readability.

if is_view:            
    viewer.close( )