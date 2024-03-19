# ========================================================================================== #
#  [Script Name]: EDA_joint.py under example5_rhythmic
#       [Author]: Moses Chong-ook Nah
#      [Contact]: mosesnah@mit.edu
# [Date Created]: 2024.03.18
#  [Description]: Simulation using Elementary Dynamic Actions (EDA)
#                 Rhythmic movement in joint-space
#   
#                 This .py file is for running/generating Figure 10 of the 
#                 following manuscript from Nah, Lachner and Hogan
#                 "Robot Control based on Motor Primitives — A Comparison of Two Approaches" 
#
#                 The code is detailed with comments, but not at the level of examples 1, 2, 3 
#                 since after example 4 will be "advanced" application of EDA.
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

# Setting the Initial Condition of the robot
nq     = model.nq
q_init = np.array( [ 0.5, 0.5 ] )
q_amp  = np.array( [ 0.1, 0.3 ] )
w0     = np.pi 

data.qpos[ 0:nq ] = q_init
data.qvel[ 0:nq ] = w0 * q_amp
mujoco.mj_forward( model, data )

# ========================================================================================== #
# [Section #3] Parameters for Elementary Dynamic Actions and the Main Simulation
# ========================================================================================== #

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

    # The oscillatory virtual trajectory
    q0  = q_init + q_amp * np.sin( w0 * data.time )
    dq0 =     q_amp * w0 * np.cos( w0 * data.time )

    # Module 1: First-order Joint-space Impedance Controller
    tau_imp = Kq @ ( q0 - q ) + Bq @ ( dq0 - dq )

    # Adding the Torque as an input
    data.ctrl[ : ] = tau_imp

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