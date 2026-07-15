import sys
from pathlib import Path
import time
import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from so101.mujoco_utils import set_initial_pose, send_position_command, move_to_pose, hold_position
from so101.inverse_kinematics import get_inverse_kinematics
from so101.forward_kinematics import get_forward_kinematics
import numpy as np

xml_path = PROJECT_ROOT / 'asset' / 'so_arm100' / 'SO101' / 'scene.xml'
body_xml_path = PROJECT_ROOT / 'asset' / 'so_arm100' / 'SO101' / 'so101_body.xml'

m = mujoco.MjModel.from_xml_path(str(xml_path))
d = mujoco.MjData(m)

print("=== Model (m) ===")
print("nq:", m.nq, "nv:", m.nv, "nu:", m.nu, "nbody:", m.nbody, "njnt:", m.njnt)
print("joint ranges:\n", m.jnt_range)
print("actuator ctrlrange:\n", m.actuator_ctrlrange)

print("\n=== Data (d) ===")
print("time:", d.time)
print("qpos:", d.qpos.copy())
print("qvel:", d.qvel.copy())
print("ctrl:", d.ctrl.copy())
print("xpos (body world positions):\n", d.xpos.copy())   # shape: (nbody, 3)

# Helper Function to show a cube at a given position and orientation
def show_cube(viewer, position, orientation, geom_num=0, halfwidth=0.013):
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_num],
        type=mujoco.mjtGeom.mjGEOM_BOX, 
        size=[halfwidth, halfwidth, halfwidth],                 
        pos=position,                         
        mat=orientation.flatten(),              
        rgba=[1, 0, 0, 0.2]                           
    )
    viewer.user_scn.ngeom = 1
    viewer.sync()
    return
  
# Initial joint configuration at start of simulation
initial_config = {
    'shoulder_pan': 0.0,
    'shoulder_lift': 0.0,
    'elbow_flex': 0.00,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 0          
}
set_initial_pose(d, initial_config)
send_position_command(d, initial_config)

# Start simulation with mujoco viewer
with mujoco.viewer.launch_passive(m, d) as viewer:
  
  # Specify the desired position of the cube to be picked up
  temp_desired_position = [0.2, 0.15, 0.014]

  # Add a cylinder as a site for visualization
  show_cube(viewer, temp_desired_position, np.eye(3))
  
  # First send the robot to a higher position with the gripper open
  joint_configuration = get_inverse_kinematics(temp_desired_position, viewer)
  move_to_pose(m, d, viewer, joint_configuration, 1.0)
  
  # Hold position for 10 seconds
  hold_position(m, d, viewer, 60.0)
