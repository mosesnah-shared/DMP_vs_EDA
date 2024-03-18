import sys
import numpy as np
import mujoco
import mujoco_viewer

sys.path += [ "controllers", "utils" ]

from utils    import min_jerk_traj
from scipy.io import savemat

# Call the xml model file + data for MuJoCo
dir_name   = './models/my_robots/'
robot_name = '2DOF_planar_torque.xml'
model = mujoco.MjModel.from_xml_path( dir_name + robot_name )
data  = mujoco.MjData( model )

# Create the viewer object
viewer = mujoco_viewer.MujocoViewer( model, data, hide_menus = True )

# Set numpy print options
np.set_printoptions( precision = 4, threshold = 9, suppress = True )

# Parameters for the simulation
T        = 3.                       # Total Simulation Time
dt       = model.opt.timestep       # Time-step for the simulation (set in xml file)
fps      = 30                       # Frames per second
save_ps  = 1000                     # Saving point per second
n_frames = 0                        # The current frame of the simulation
n_saves  = 0                        # Update for the saving point
speed    = 1.0                      # The speed of the simulator

t_update = 1./fps     * speed       # Time for update 
t_save   = 1./save_ps * speed       # Time for saving the data

# The time-step defined in the xml file should be smaller than update
assert( dt <= t_update and dt <= t_save )

# The number of degrees of freedom
nq = model.nq

q_init = np.zeros( nq )
data.qpos[ 0:nq ] = q_init
mujoco.mj_forward( model, data )

# The impedances of the robot 
Kq = 150 * np.eye( nq )
Bq =  50 * np.eye( nq )

# The parameters of the minimum-jerk trajectory.
t0   = 0.3
D    = 1.0
qdel = np.array( [ 1.0, 1.0 ] )
qi   = q_init
qf   = q_init + qdel

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
is_save = True
is_view = False

while data.time <= T:

    mujoco.mj_step( model, data )
    q0, dq0, _ = min_jerk_traj( data.time, t0, t0 + D, qi, qf )

    # Torque 1: First-order Joint-space Impedance Controller
    tau_imp = Kq @ ( q0 - q ) + Bq @ ( dq0 - dq )

    # Adding the Torque
    data.ctrl[ : ] = tau_imp

    # Update Visualization
    if ( ( n_frames != ( data.time // t_update ) ) and is_view ):
        n_frames += 1
        viewer.render( )
        print( "[Time] %6.3f" % data.time )

    # Save Data
    if ( ( n_saves != ( data.time // t_save ) ) and is_save ):
        n_saves += 1

        t_mat.append(   np.copy( data.time ) )
        q_mat.append(   np.copy( q   ) )
        q0_mat.append(  np.copy( q0  ) )
        dq_mat.append(  np.copy( dq  ) )
        dq0_mat.append( np.copy( dq0 ) )

# Saving the data
if is_save:
    data_dic = { "t_arr": t_mat, "q_arr": q_mat, "q0_arr": q0_mat, "dq_arr": dq_mat, "dq0_arr": dq0_mat, "Kq": Kq, "Bq": Bq }
    savemat( "./ThesisExamples/data/sec511_joint.mat", data_dic )

if is_view:            
    viewer.close()