import time
import mujoco

SO101_JOINT_ORDER = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]


def _read_pose_dict_from_model(m, d):
    q = []
    for joint_name in SO101_JOINT_ORDER:
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            return None
        qpos_adr = m.jnt_qposadr[joint_id]
        q.append(float(d.qpos[qpos_adr]))
    return convert_to_dictionary(q)
        
def convert_to_dictionary(qpos):
    return {
        'shoulder_pan': qpos[0]*180.0/3.14159,    # convert to degrees
        'shoulder_lift': qpos[1]*180.0/3.14159,   # convert to degrees
        'elbow_flex': qpos[2]*180.0/3.14159,      # convert to degrees
        'wrist_flex': qpos[3]*180.0/3.14159,      # convert to degrees
        'wrist_roll': qpos[4]*180.0/3.14159,      # convert to degrees
        'gripper': qpos[5]*100/3.14159            # convert to 0-100 range
    }
    
def convert_to_list(dictionary):
    return [
        dictionary['shoulder_pan']*3.14159/180.0,
        dictionary['shoulder_lift']*3.14159/180.0,
        dictionary['elbow_flex']*3.14159/180.0,
        dictionary['wrist_flex']*3.14159/180.0,
        dictionary['wrist_roll']*3.14159/180.0,
        dictionary['gripper']*3.14159/100.0
    ]

def set_initial_pose(d, position_dict):
    pos = convert_to_list(position_dict)
    d.qpos = pos

def send_position_command(d, position_dict):
    pos = convert_to_list(position_dict)
    d.ctrl = pos

def move_to_pose(m, d, viewer, desired_position, duration, verbose=True):
    start_time = time.time()
    starting_pose = _read_pose_dict_from_model(m, d)
    if starting_pose is None:
        starting_pose = convert_to_dictionary(d.qpos.copy())

    if verbose:
        print(f"Moving to desired position: {desired_position} over {duration} seconds.")
    
    while True:
        t = time.time() - start_time
        if t > duration:
            break

        # Interpolation factor [0,1] (make sure it doesn't exceed 1)
        alpha = min(t / duration, 1)

        # Interpolate each joint
        position_dict = {}
        for joint in desired_position:
            p0 = starting_pose[joint]
            pf = desired_position[joint]
            position_dict[joint] = (1 - alpha) * p0 + alpha * pf

        # Send command
        send_position_command(d, position_dict)
        mujoco.mj_step(m, d)
        
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
    
def hold_position(m, d, viewer, duration):
    current_pos_dict = _read_pose_dict_from_model(m, d)
    if current_pos_dict is None:
        current_pos_dict = convert_to_dictionary(d.qpos.copy())
    
    start_time = time.time()
    while True:
        t = time.time() - start_time
        if t > duration:
            break
        send_position_command(d, current_pos_dict)
        mujoco.mj_step(m, d)
        viewer.sync()
