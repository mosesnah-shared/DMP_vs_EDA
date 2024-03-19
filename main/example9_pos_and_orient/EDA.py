import sys
import numpy as np
import mujoco
import mujoco_viewer
from scipy.io import savemat
from scipy.spatial.transform import Rotation

from pathlib  import Path

# Add and save the ROOT_PATH to run this code, which is three levels above.
ROOT_PATH = str( Path( __file__ ).parent.parent.parent )
sys.path.append( ROOT_PATH )

# Also save the CURRENT PATH of this File
CURRENT_PATH = str( Path( __file__ ).parent )

# Importing the Minimum-jerk Trajectory Function
from utils.trajectory  import min_jerk_traj
from utils.geom_funcs  import rotx, R3_to_SO3, SO3_to_R3

# Set numpy print options
np.set_printoptions( precision = 4, threshold = 9, suppress = True )

# Basic MuJoCo setups
dir_name   = ROOT_PATH + '/models/iiwa14/'
robot_name = 'iiwa14.xml'
model      = mujoco.MjModel.from_xml_path( dir_name + robot_name )
data       = mujoco.MjData( model )
viewer     = mujoco_viewer.MujocoViewer( model, data, hide_menus = True )

# Parameters for the simulation
T        = 14.                       # Total Simulation Time
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

q_init = np.array( [-0.5000, 0.8236,0,-1.0472,0.8000, 1.5708, 0 ] )
data.qpos[ 0:nq ] = q_init
mujoco.mj_forward( model, data )

# The impedances of the robot 
Kp = 1600 * np.eye( 3 )
Bp =  800 * np.eye( 3 )

Bq = 20 * np.eye( model.nq )

kr = 80 
br =  8

# Save the references for the q and dq 
q  = data.qpos[ 0:nq ]
dq = data.qvel[ 0:nq ]

# Get the end-effector's ID
EE_site = "site_end_effector"
id_EE = model.site( EE_site ).id

# Saving the references 
p   = data.site_xpos[ id_EE ]
Rsb = data.site_xmat[ id_EE ]

# Saving the position and orientation of 7 links
p_ref = []
R_ref = [] 

for i in range( 7 ):
    name = "iiwa14_link_" + str( i + 1 ) 


    p_ref.append( data.body( name ).xpos )
    R_ref.append( data.body( name ).xmat )

Jp = np.zeros( ( 3, model.nq ) )
Jr = np.zeros( ( 3, model.nq ) )

mujoco.mj_jacSite( model, data, Jp, Jr, id_EE )

dp = Jp @ dq
w  = Jr @ dq

# Get the initial position of the robot's end-effector
# and also the other parameters
pi = np.copy( p )
pf = pi - 2 * np.array( [0.0, pi[ 1 ], 0.0])
t0 = 2.0
D  = 2.0

ang = 90 
Rinit = np.copy( Rsb ).reshape( 3, -1 )
Rgoal = Rinit @ rotx( ang*np.pi/180 ) 
Rdel  = Rinit.T @ Rgoal
wdel  = SO3_to_R3( Rdel )

# Flags
is_save = True
is_view = True

# The data for mat save
t_mat   = [ ]
q_mat   = [ ] 
p_mat   = [ ] 
p0_mat  = [ ] 
dq_mat  = [ ] 
dp_mat  = [ ] 
dp0_mat = [ ] 
p_links_save = [ ]
R_links_save = [ ]
R_mat  = [ ]
R0_mat = [ ]

while data.time <= T:

    mujoco.mj_step( model, data )
    p0, dp0, _ = min_jerk_traj( data.time, t0, t0 + D, pi, pf )

    # Torque 1: First-order Joint-space Impedance Controller
    mujoco.mj_jacSite( model, data, Jp, Jr, id_EE )

    dp = Jp @ dq

    tau_imp1 = Jp.T @ ( Kp @ ( p0 - p ) + Bp @ ( dp0 - dp ) )

    # For orientation
    Rcurr = np.copy( Rsb ).reshape( 3, -1 )

    # Get the R0 as minimum-jerk trajectory
    tmp, _, _ = min_jerk_traj( data.time, t0, t0 + D, np.zeros( 3 ), wdel )

    R0 = Rinit @ R3_to_SO3( tmp )

    tmp1 = Rotation.from_matrix( Rcurr.T @ R0 )
    tau_imp2 = Jr.T @ ( kr * Rcurr @ tmp1.as_rotvec( ) - br * Jr @ dq )
    tau_imp3 =  -Bq @ dq

    # Adding the Torque
    data.ctrl[ : ] = tau_imp1 + tau_imp2 + tau_imp3

    # Update Visualization
    if ( ( n_frames != ( data.time // t_update ) ) and is_view ):
        n_frames += 1
        viewer.render( )
        print( "[Time] %6.3f" % data.time )

    # Save Data
    if ( ( n_saves != ( data.time // t_save ) ) and is_save ):
        n_saves += 1

        t_mat.append(   np.copy( data.time ) )
        q_mat.append(   np.copy(  q  ) )
        dq_mat.append(  np.copy( dq  ) )
        p_mat.append(   np.copy(  p  ) )
        p0_mat.append(  np.copy( p0  ) )    
        dp_mat.append(  np.copy( dp  ) )
        dp0_mat.append( np.copy( dp0 ) )    
        R_mat.append(    np.copy( Rsb ).reshape( 3, -1 ) ) 
        R0_mat.append(   R0 )    

        # For this, one needs to also save the link positions
        # Also save the robot's link position and rotation matrices 
        p_tmp = []
        R_tmp = []
        for i in range( 7 ):
            p_tmp.append( np.copy( p_ref[ i ] ) )
            R_tmp.append( np.copy( R_ref[ i ] ).reshape( 3, -1 ) )

        # Also save the robot's link position and rotation matrices 
        p_links_save.append( p_tmp )
        R_links_save.append( R_tmp )               

# Saving the data
if is_save:
    data_dic = { "t_arr": t_mat, "q_arr": q_mat, "p_arr": p_mat, "R_arr": R_mat, "R0_arr": R0_mat,
                "dp_arr": dp_mat, "p0_arr": p0_mat, "dq_arr": dq_mat, "dp0_arr": dp0_mat, "Kp": Kp, "Bp": Bp, 
                 "kr": kr, "br": br, "Bq": Bq, "p_links": p_links_save, "R_links": R_links_save }
    savemat( CURRENT_PATH + "/data/EDA.mat", data_dic )

if is_view:            
    viewer.close()