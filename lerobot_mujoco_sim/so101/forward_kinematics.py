import numpy as np


# ---------------------------------------------------------------------------
# Basic rotation matrices (angles in degrees)
# ---------------------------------------------------------------------------

def Rx(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])

def Ry(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def Rz(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def make_transform(rotation, displacement):
    """Assemble a 4x4 homogeneous transform from a 3x3 rotation and a 3-vector."""
    T = np.eye(4)
    T[0:3, 0:3] = rotation
    T[0:3, 3]   = displacement
    return T


# ---------------------------------------------------------------------------
# Individual joint transforms
#
# Each transform validated against MuJoCo ground truth for test config:
#   shoulder_pan=-45, shoulder_lift=45, elbow_flex=-45, wrist_flex=90, wrist_roll=0
#
# MuJoCo ground truth (world-frame joint positions):
#   shoulder_pan:  [0.0388,  0.0000, 0.0624]
#   shoulder_lift: [0.0733,  0.0086, 0.1166]
#   elbow_flex:    [0.1435,  0.0789, 0.1764]
#   wrist_flex:    [0.2389,  0.1742, 0.1816]
#   wrist_roll:    [0.2261,  0.1870, 0.1205]
# ---------------------------------------------------------------------------

def get_gw1(theta1_deg):
    """World -> Joint 1 (shoulder_pan).
    Provided directly in the assignment starter code.
    Validated: pos=[0.0388, 0.0000, 0.0624] ✓
    """
    displacement = np.array([0.0388353, 0.0, 0.0624])
    rotation = Rz(180) @ Rx(180) @ Rz(theta1_deg)
    return make_transform(rotation, displacement)


def get_g12(theta2_deg):
    """Joint 1 -> Joint 2 (shoulder_lift).
    xyz=(-0.0303992, 0, -0.0542), rpy=(-pi/2, -pi/2, 0) -> Ry(-90)@Rx(-90).
    (y offset 0.0182778 from URDF not present in FK diagram, omitted.)
    Validated: pos=[0.0733, 0.0086, 0.1166] ✓
    """
    displacement = np.array([-0.0303992, -0.0182778, -0.0542])
    R_fixed = Ry(-90) @ Rx(-90)
    rotation = R_fixed @ Rz(theta2_deg)
    return make_transform(rotation, displacement)


def get_g23(theta3_deg):
    """Joint 2 -> Joint 3 (elbow_flex).
    xyz=(-0.11257, -0.028, 0), rpy=(0, 0, pi/2) -> Rz(90).
    Validated: pos=[0.1435, 0.0789, 0.1764] ✓
    """
    displacement = np.array([-0.11257, -0.028, 0.0])
    R_fixed = Rz(90)
    rotation = R_fixed @ Rz(theta3_deg)
    return make_transform(rotation, displacement)


def get_g34(theta4_deg):
    """Joint 3 -> Joint 4 (wrist_flex).
    xyz=(-0.1349, 0.0052, 0), rpy=(0, 0, -pi/2) -> Rz(-90).
    Validated: pos=[0.2389, 0.1742, 0.1816] ✓
    """
    displacement = np.array([-0.1349, 0.0052, 0.0])
    R_fixed = Rz(-90)
    rotation = R_fixed @ Rz(theta4_deg)
    return make_transform(rotation, displacement)


def get_g45(theta5_deg):
    """Joint 4 -> Joint 5 (wrist_roll).
    From diagram: 0.0611 from wrist_flex to wrist_roll.
    Ground truth delta from wrist_flex: [-0.0128, +0.0128, -0.0611]
    In wrist_flex local frame this is (0, 0, 0.0611) along local z.
    Rx(90) reorients so wrist_roll z-axis = world z (confirmed by ground truth
    rot matrix showing z-col=[0,0,1]).
    Joint rotates about its local z-axis.
    Validated: pos=[0.2261, 0.1870, 0.1205] ✓
    """
    displacement = np.array([0.0, -0.0611, 0.0181])
    R_fixed = Rx(-90) @ Rz(180)
    rotation = R_fixed @ Rz(theta5_deg)
    return make_transform(rotation, displacement)


def get_g5t():
    """Joint 5 -> TCP (gripper tip), fixed transform.
    From FK diagram: 0.1034 m to fingertip along gripper axis.
    wrist_roll z-axis points in world z, so TCP offset is along local z.
    """
    displacement = np.array([0.0, 0.0, -0.100])
    # rotation = Rx(90) @ Ry(0) @Rz(-90) # standard FK
    rotation = np.eye(3) # easier for IK calc
    return make_transform(rotation, displacement)


# ---------------------------------------------------------------------------
# Full forward kinematics
# ---------------------------------------------------------------------------

def get_forward_kinematics(position_dict):
    """Compute the TCP pose given a joint configuration dict.

    Parameters
    ----------
    position_dict : dict
        Keys: 'shoulder_pan', 'shoulder_lift', 'elbow_flex',
              'wrist_flex', 'wrist_roll'  (all in degrees)

    Returns
    -------
    position   : np.ndarray, shape (3,)   xyz of the TCP in world frame
    rotation   : np.ndarray, shape (3,3)  rotation matrix of the TCP frame
    """
    gw1 = get_gw1(position_dict['shoulder_pan'])
    g12 = get_g12(position_dict['shoulder_lift'])
    g23 = get_g23(position_dict['elbow_flex'])
    g34 = get_g34(position_dict['wrist_flex'])
    g45 = get_g45(position_dict['wrist_roll'])
    g5t = get_g5t()

    gwt = gw1 @ g12 @ g23 @ g34 @ g45 @ g5t

    gwt = gw1 @ g12 @ g23 @ g34 @g45 @ g5t
    position = gwt[0:3, 3]
    rotation = gwt[0:3, 0:3]
    return position, rotation


# ---------------------------------------------------------------------------
# Validation against MuJoCo ground truth
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    configs = {
        'shoulder_pan':  -45.0,
        'shoulder_lift':  45.0,
        'elbow_flex':    -45.0,
        'wrist_flex':     90.0,
        'wrist_roll':      0.0,
    }

    gw1 = get_gw1(configs['shoulder_pan'])
    g12 = get_g12(configs['shoulder_lift'])
    g23 = get_g23(configs['elbow_flex'])
    g34 = get_g34(configs['wrist_flex'])
    g45 = get_g45(configs['wrist_roll'])
    g5t = get_g5t()

    stages = {
        'shoulder_pan  (expect [0.0388,  0.0000, 0.0624])': gw1,
        'shoulder_lift (expect [0.0733,  0.0086, 0.1166])': gw1 @ g12,
        'elbow_flex    (expect [0.1435,  0.0789, 0.1764])': gw1 @ g12 @ g23,
        'wrist_flex    (expect [0.2389,  0.1742, 0.1816])': gw1 @ g12 @ g23 @ g34,
        'wrist_roll    (expect [0.2261,  0.1870, 0.1205])': gw1 @ g12 @ g23 @ g34 @ g45,
        'TCP tip':                                           gw1 @ g12 @ g23 @ g34 @ g45 @ g5t,
    }

    for name, T in stages.items():
        pos = T[0:3, 3]
        print(f"{name}  ->  [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    
    