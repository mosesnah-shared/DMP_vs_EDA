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
robot_name = '2DOF_planar_torque_w_obstacle.xml'
model      = mujoco.MjModel.from_xml_path( dir_name + robot_name )
data       = mujoco.MjData( model )
viewer     = mujoco_viewer.MujocoViewer( model, data, hide_menus = True )

# Parameters for the simulation
T        = 3.                       # Total Simulation Time
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
nq     = model.nq
q1     = np.pi * 1/12
q_init = np.array( [ q1, np.pi-2*q1 ] )
data.qpos[ 0:nq ] = q_init
mujoco.mj_forward( model, data )

# The task-space impedances of the 2-DOF robot 
kp = 60
bp = 20
Kp = kp * np.eye( 3 )
Bp = bp * np.eye( 3 )

# Save the references for the q and dq 
q  = data.qpos[ 0:nq ]
dq = data.qvel[ 0:nq ]

# Get the end-effector's ID and its position, translational and rotational Jacobian matrices
EE_site = "site_end_effector"
id_EE = model.site( EE_site ).id

# Getting the obstacle's ID and reference for displacement
obs_name = "body_obstacle"
id_obs = model.body( "body_obstacle").id
p_obs  = model.body_pos[ id_obs ]

# Saving the references 
p  = data.site_xpos[ id_EE ]
Jp = np.zeros( ( 3, nq ) )
Jr = np.zeros( ( 3, nq ) )

# The parameters of the minimum-jerk trajectory.
t0   = 0.3       
pdel = np.array( [ 0.0, 1.2, 0.0 ] )
pi   = np.copy( p )
pf   = pi + pdel
D    = 1.0

# The data for mat save
t_mat     = [ ]
q_mat     = [ ] 
p_mat     = [ ] 
p0_mat    = [ ] 
dq_mat    = [ ] 
dp_mat    = [ ] 
dp0_mat   = [ ] 
Jp_mat    = [ ]
p_obs_mat = [ ]
 
# Flags
is_save = True      # To save the data
is_view = True      # To view the simulation
is_mod  = False     # Modulation of energy on

# The main simulation loop
while data.time <= T:

    mujoco.mj_step( model, data )

    p0, dp0, _ = min_jerk_traj( data.time, t0, t0 + D, pi, pf )
    
    # Getting the Jacobian matrices, translational part
    mujoco.mj_jacSite( model, data, Jp, Jr, id_EE )
    dp = Jp @ dq

    if is_mod:
        NotImplementedError( )
    else:
        # Torque 1: First-order Joint-space Impedance Controller
        # Without Energy monitoring and modulation. 
        tau_imp = Jp.T @ ( Kp @ ( p0 - p ) + Bp @ ( dp0 - dp ) )

    # Adding the Torque as an input
    data.ctrl[ : ] = tau_imp

    # Moving the obstacle, when time between 1. and 2.
    if data.time >= 1.0 and data.time <= 2.0:
        # Move the obstacle to a new position 
        p_obs -= np.array( [ 0.0005, 0.0, 0.0] )    

    # Update Visualization
    if ( ( n_frames != ( data.time // t_update ) ) and is_view ):
        n_frames += 1
        viewer.render( )
        print( "[Time] %6.3f" % data.time )

    # Saving the data
    if ( ( n_saves != ( data.time // t_save ) ) and is_save ):
        n_saves += 1

        t_mat.append(   np.copy( data.time ) )
        q_mat.append(   np.copy(  q  ) )
        dq_mat.append(  np.copy( dq  ) )
        p_mat.append(   np.copy(  p  ) )
        p0_mat.append(  np.copy( p0  ) )    
        dp_mat.append(  np.copy( dp  ) )
        dp0_mat.append( np.copy( dp0 ) )    
        Jp_mat.append(  np.copy(  Jp ) )
        p_obs_mat.append( np.copy( p_obs ) )

# Save Data as mat file for MATLAB visualization
if is_save:
    data_dic = { "t_arr": t_mat, "q_arr": q_mat, "p_arr": p_mat, "dp_arr": dp_mat, "p_obs_arr" :p_obs_mat,
                 "p0_arr": p0_mat, "dq_arr": dq_mat, "dp0_arr": dp0_mat, "Kp": Kp, "Bq": Bp, "Jp_arr": Jp_mat }
    
    if is_mod:
        savemat( CURRENT_PATH + "/data/EDA_w_modulation.mat", data_dic )
    else:
        savemat( CURRENT_PATH + "/data/EDA_wo_modulation.mat", data_dic )


if is_view:            
    viewer.close()