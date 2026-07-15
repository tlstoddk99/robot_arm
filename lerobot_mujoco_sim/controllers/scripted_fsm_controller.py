# scripted_fsm_controller_with_mink.py
#
# Robust scripted controller for MuJoCo dataset generation using mink IK.
# Designed for long unattended demo collection (10k+ episodes).
#
# Features
# - mink-based IK solver with QP optimization
# - declarative phase table
# - IK retry logic
# - timeout protection
# - grasp validation
# - clean state handling

import numpy as np
import mujoco
import mink
from typing import Optional, Tuple, Callable

# -------------------------------------------------------------
# GRASP CALIBRATION (same as before)
# -------------------------------------------------------------

GRASP_Z_OFFSET = -0.012       # final Z offset for grasp
GRASP_X_OFFSET = -0.005       # small XY tweak to center over block
GRASP_Y_OFFSET = 0.0
APPROACH_Z_OFFSET = 0.075     # hover height above block / box
PLACE_Z_OFFSET = 0.040        # depth when placing in box
# Max Z step per tick when descending onto block (m/tick).
DESCEND_Z_STEP = 0.0035
# Max Z step per tick when descending into bin (avoids overshoot/oscillation)
PLACE_DESCEND_Z_STEP = 0.0028
# When within this distance in place_in_box, scale down dq to avoid overshoot/oscillation
PLACE_DAMP_DIST = 0.04
FSM_DEBUG = True
FSM_DEBUG_EVERY = 5
# Max joint delta per control step (rad/tick); larger = faster motion
MAX_DQ_STEP = 1.0

# Orientation policy: only yaw (Z-rotation) is allowed to change.
LOCK_XY_ROTATION = True       # keep roll/pitch fixed, free yaw

# -------------------------------------------------------------
# PHASE TABLE - ADD NEW ROTATION PHASE
# -------------------------------------------------------------

def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize_wxyz(q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)
    return quat_normalize_wxyz(q)


def quat_wxyz_to_rpy(q: np.ndarray) -> Tuple[float, float, float]:
    r = quat_wxyz_to_rotmat(q)
    roll = np.arctan2(r[2, 1], r[2, 2])
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return float(roll), float(pitch), float(yaw)

# Top-down quaternion for gripperframe site (wxyz).
# The gripperframe site has a 90-deg Y rotation baked in (quat="0.707107 0 0.707107 0")
# so the site's X-axis = gripper body -Z (approach direction).
# For top-down approach we need site X-axis -> world -Z,
# i.e. R(q)*[1,0,0] = [0,0,-1], solved by q = [0.707107, 0, 0.707107, 0].
TOP_DOWN_QUAT = np.array([0.707107, 0.0, 0.707107, 0.0], dtype=np.float64)
DEFAULT_EE_SITE_NAME = "gripperframe"


def resolve_ee_site_name(env, preferred: Optional[str] = None) -> str:
    """Resolve a valid end-effector site name for the current model."""
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([DEFAULT_EE_SITE_NAME, "tip"])

    seen = set()
    for name in candidates:
        if name in seen or name is None:
            continue
        seen.add(name)
        try:
            env.model.site(name).id
            return name
        except KeyError:
            continue

    valid_names = [env.model.site(i).name for i in range(env.model.nsite)]
    raise KeyError(
        f"No valid end-effector site found. Tried {candidates}. "
        f"Available sites: {valid_names}"
    )

def quat_angle_between(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute angle between two quaternions (wxyz format)."""
    q1 = quat_normalize_wxyz(q1)
    q2 = quat_normalize_wxyz(q2)
    dot = np.abs(np.clip(np.dot(q1, q2), -1.0, 1.0))
    return 2.0 * np.arccos(dot)


def topdown_axis_error(q_wxyz: np.ndarray) -> float:
    """Yaw-invariant top-down error for ee site X-axis vs world -Z (radians)."""
    rot = quat_wxyz_to_rotmat(q_wxyz)
    ee_x_axis = rot[:, 0]
    desired_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    cosang = np.clip(np.dot(ee_x_axis, desired_down), -1.0, 1.0)
    return float(np.arccos(cosang))

PHASES = [
    # 0) On episode start, move from stow to a safe "home" pose
    dict(
        name="stow_to_home",
        target=None,          # filled from fsm["home_xyz"]
        gripper=1.0,
        tol=0.04,
        timeout=250,
    ),
    # 1) Move above the block with gripper pointing down, yaw-only rotation
    dict(
        name="above_block",
        target=lambda cube, box: [
            cube[0] + GRASP_X_OFFSET,
            cube[1] + GRASP_Y_OFFSET,
            cube[2] + APPROACH_Z_OFFSET,
        ],
        # Start episodes with the gripper closed so we clearly
        # see the open gesture at the beginning of the descend phase.
        gripper=1.0,
        tol=0.02,
        timeout=150,
    ),
    # 2) Descend straight down onto the block
    dict(
        name="descend",
        target=lambda cube, box: [
            cube[0] + GRASP_X_OFFSET,
            cube[1] + GRASP_Y_OFFSET,
            cube[2] + GRASP_Z_OFFSET,
        ],
        # Open as we descend so the jaws clear any obstacles before closing.
        gripper=0.0,
        tol=0.015,
        timeout=150,
    ),
    # 3) Close gripper in place
    dict(
        name="grip",
        target=None,          # hold current pose
        gripper=1.0,
        tol=None,             # purely time-based
        timeout=15,
    ),
    # 4) Lift straight up from the table
    dict(
        name="lift",
        target=lambda cube, box: [
            cube[0] + GRASP_X_OFFSET,
            cube[1] + GRASP_Y_OFFSET,
            cube[2] + APPROACH_Z_OFFSET,
        ],
        gripper=1.0,
        tol=0.02,
        timeout=150,
    ),
    # 5) Move to a position above the box
    dict(
        name="move_to_box",
        target=lambda cube, box: [
            box[0],
            box[1],
            box[2] + APPROACH_Z_OFFSET,
        ],
        gripper=1.0,
        tol=0.02,
        timeout=200,
    ),
    # 6) Lower into box and release
    dict(
        name="place_in_box",
        target=lambda cube, box: [
            box[0],
            box[1],
            box[2] + PLACE_Z_OFFSET,
        ],
        gripper=0.0,          # open while at bottom
        tol=0.02,
        timeout=150,
    ),
    # 7) Return to home (initial) pose
    dict(
        name="return_home",
        target=None,          # filled from fsm["home_xyz"]
        gripper=0.0,
        tol=0.03,
        timeout=250,
    ),
]

# -------------------------------------------------------------
# MINK IK SOLVER SETUP (unchanged)
# -------------------------------------------------------------

class MinkIKSolver:
    """Wrapper for mink IK solver with SO-101 arm configuration."""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, 
                 arm_joint_names: list, ee_site_name: str = DEFAULT_EE_SITE_NAME):
        """
        Initialize mink solver for SO-101 arm.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            arm_joint_names: List of joint names for the arm
            ee_site_name: Name of the end-effector site
        """
        self.model = model
        self.data = data
        self.arm_joint_names = arm_joint_names
        self.ee_site_name = ee_site_name
        self.lock_xy_rotation = LOCK_XY_ROTATION
        self.ee_site_id = self.model.site(self.ee_site_name).id
        
        # Create mink configuration
        self.configuration = mink.Configuration(model)
        
        # Set up tasks
        self.tasks = []
        
        # End-effector task
        self.ee_task = mink.FrameTask(
            frame_name=ee_site_name,
            frame_type="site",
            position_cost=8.0,
            orientation_cost=1.0,
            lm_damping=1.0,
            gain=1.0,
        )
        self.tasks.append(self.ee_task)
        
        # Posture task to keep joints near preferred positions
        self.posture_task = mink.PostureTask(
            model=model,
            cost=1e-2,               # Low priority
            gain=1.0,                 # Feedback gain
            lm_damping=1.0,            # Damping
        )
        self.tasks.append(self.posture_task)
        
        # Set up limits
        self.limits = []
        
        # Joint position limits (from model)
        self.limits.append(mink.ConfigurationLimit(model=model))
        
        # Velocity limits (tune these for SO-101)
        max_vel = 3.0  # rad/s
        self.limits.append(mink.VelocityLimit(
            model,
            {name: max_vel for name in arm_joint_names}
        ))
        
        # Solver parameters
        self.solver = "daqp"  # or "quadprog" if installed
        self.damping = 1e-6
        self.max_iters = 20
        self.pos_threshold = 1e-4
        self.dt = model.opt.timestep
        
        # Store preferred posture (current configuration)
        self._update_preferred_posture()

    def _get_current_site_yaw(self) -> float:
        rot = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        return float(np.arctan2(rot[1, 0], rot[0, 0]))

    def _lock_roll_pitch_allow_yaw(self, reference_quat_wxyz: np.ndarray) -> np.ndarray:
        ref_roll, ref_pitch, _ = quat_wxyz_to_rpy(reference_quat_wxyz)
        current_yaw = self._get_current_site_yaw()
        return rpy_to_quat_wxyz(ref_roll, ref_pitch, current_yaw)
        
    def _update_preferred_posture(self):
        """Update the posture task target to current configuration."""
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        
    def solve(self, target_xyz: np.ndarray, target_quat: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Solve IK for target position and orientation.
        
        Args:
            target_xyz: Target position [x, y, z]
            target_quat: Target orientation as [w, x, y, z] (optional)
            
        Returns:
            Joint positions if solution found, None otherwise
        """
        try:
            # Update configuration from current state
            self.configuration.update(self.data.qpos)
            
            # Set end-effector target
            if target_quat is not None:
                target_quat_arr = np.array(target_quat, dtype=np.float64)
                if self.lock_xy_rotation:
                    target_quat_arr = self._lock_roll_pitch_allow_yaw(target_quat_arr)

                # Use both position and orientation
                # target_quat is [w, x, y, z] which matches SO3's wxyz convention
                target_se3 = mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3(wxyz=target_quat_arr),
                    translation=np.array(target_xyz, dtype=np.float64),
                )
            else:
                # No target quat provided: keep current orientation while tracking position
                current_transform = self.configuration.get_transform_frame_to_world(
                    self.ee_site_name, "site"
                )
                target_se3 = mink.SE3.from_rotation_and_translation(
                    rotation=current_transform.rotation(),
                    translation=np.array(target_xyz, dtype=np.float64),
                )
            
            self.ee_task.set_target(target_se3)
            
            # Update posture task (optional - uncomment to keep current posture)
            # self._update_preferred_posture()
            
            # Solve IK
            vel = mink.solve_ik(
                configuration=self.configuration,
                tasks=self.tasks,
                dt=self.dt,
                solver=self.solver,
                damping=self.damping,
                limits=self.limits,
            )
            
            # Integrate solution
            self.configuration.integrate_inplace(vel, self.dt)
            
            # Extract joint positions
            q_solution = []
            for name in self.arm_joint_names:
                joint_id = self.model.joint(name).id
                qpos_addr = self.model.jnt_qposadr[joint_id]
                q_solution.append(self.configuration.q[qpos_addr])
            
            return np.array(q_solution)
            
        except Exception as e:
            print(f"[MINK IK] Solver failed: {e}")
            return None


# -------------------------------------------------------------
# FSM STATE (updated with mink solver)
# -------------------------------------------------------------

def make_fsm():
    return dict(
        phase=0,
        tick=0,
        pick_xyz=None,
        bin_xyz=None,
        home_xyz=None,
        approach_z=None,
        box_approach_z=None,
        ik_failures=0,
        ee_site_name=DEFAULT_EE_SITE_NAME,
        mink_solver=None,  # Will store mink solver instance
    )

# -------------------------------------------------------------
# smoothing curve (same as before)
# -------------------------------------------------------------

def smoothstep(x):
    return x * x * (3 - 2 * x)

def next_ee_waypoint(env, target_xyz, get_ee_xyz_fn, speed: float = 6.0):
    """
    Simple first-order trajectory towards target in Cartesian space.
    """
    ee_xyz = get_ee_xyz_fn(env)
    direction = target_xyz - ee_xyz
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return target_xyz.copy()
    step = min(1.0, speed / max(dist, 1e-6))
    return ee_xyz + step * direction

# -------------------------------------------------------------
# IK PLANNING (using mink)
# -------------------------------------------------------------

def plan_ik_with_mink(fsm: dict, target_xyz: np.ndarray, 
                      use_top_down: bool = True) -> Optional[np.ndarray]:
    """
    Plan IK using mink solver.
    
    Args:
        fsm: FSM state dictionary containing mink_solver
        target_xyz: Target position
        use_top_down: Whether to enforce top-down orientation
        
    Returns:
        Joint positions if solution found, None otherwise
    """
    solver = fsm.get("mink_solver")
    if solver is None:
        return None
    
    # Try different Z offsets for retry logic
    for dz in [0.0, -0.01, -0.02, 0.01, -0.03, 0.02]:
        trial_xyz = target_xyz.copy()
        trial_xyz[2] += dz

        if use_top_down:
            q = solver.solve(trial_xyz, TOP_DOWN_QUAT)
            if q is not None:
                return q
        else:
            q = solver.solve(trial_xyz)
            if q is not None:
                return q
    
    return None

def get_current_orientation(env, ee_site_name: str = DEFAULT_EE_SITE_NAME) -> np.ndarray:
    """Get current end-effector orientation as quaternion (wxyz)."""
    resolved_site_name = resolve_ee_site_name(env, ee_site_name)
    site_id = env.model.site(resolved_site_name).id
    # MuJoCo stores site orientations as unit quaternions in xmat (as rotation matrix)
    # We need to convert rotation matrix to quaternion
    rotmat = env.data.site_xmat[site_id].reshape(3, 3)
    
    # Convert rotation matrix to quaternion (wxyz)
    # Using standard algorithm
    trace = rotmat[0,0] + rotmat[1,1] + rotmat[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotmat[2,1] - rotmat[1,2]) * s
        y = (rotmat[0,2] - rotmat[2,0]) * s
        z = (rotmat[1,0] - rotmat[0,1]) * s
    else:
        if rotmat[0,0] > rotmat[1,1] and rotmat[0,0] > rotmat[2,2]:
            s = 2.0 * np.sqrt(1.0 + rotmat[0,0] - rotmat[1,1] - rotmat[2,2])
            w = (rotmat[2,1] - rotmat[1,2]) / s
            x = 0.25 * s
            y = (rotmat[0,1] + rotmat[1,0]) / s
            z = (rotmat[0,2] + rotmat[2,0]) / s
        elif rotmat[1,1] > rotmat[2,2]:
            s = 2.0 * np.sqrt(1.0 + rotmat[1,1] - rotmat[0,0] - rotmat[2,2])
            w = (rotmat[0,2] - rotmat[2,0]) / s
            x = (rotmat[0,1] + rotmat[1,0]) / s
            y = 0.25 * s
            z = (rotmat[1,2] + rotmat[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotmat[2,2] - rotmat[0,0] - rotmat[1,1])
            w = (rotmat[1,0] - rotmat[0,1]) / s
            x = (rotmat[0,2] + rotmat[2,0]) / s
            y = (rotmat[1,2] + rotmat[2,1]) / s
            z = 0.25 * s
    
    return quat_normalize_wxyz(np.array([w, x, y, z]))

# -------------------------------------------------------------
# GRASP VALIDATION (same as before)
# -------------------------------------------------------------

def cube_lifted(env, cube_xyz_initial, cube_fn, min_z_delta=0.01):
    cube_now = cube_fn(env)
    return cube_now[2] > cube_xyz_initial[2] + min_z_delta

# -------------------------------------------------------------
# INITIALIZATION FUNCTION
# -------------------------------------------------------------

def init_mink_for_so101(env, arm_joint_names: list, ee_site_name: str = DEFAULT_EE_SITE_NAME):
    """
    Initialize mink solver for SO-101 arm and add to FSM.
    
    Args:
        env: Environment with model and data attributes
        arm_joint_names: List of joint names for the arm (e.g., 
                        ["joint1", "joint2", "joint3", "joint4", "joint5"])
        ee_site_name: Name of the end-effector site in the model
    """
    # Create mink solver
    solver = MinkIKSolver(
        model=env.model,
        data=env.data,
        arm_joint_names=arm_joint_names,
        ee_site_name=ee_site_name
    )
    
    return solver

# -------------------------------------------------------------
# MAIN FSM STEP (modified for rotation-before-descent)
# -------------------------------------------------------------

def fsm_step(
    env,
    fsm,
    get_ee_xyz_fn,
    cube_pos_fn,
):
    """
    FSM step function using mink for IK.
    
    Args:
        env: Environment with model and data
        fsm: FSM state dictionary
        get_ee_xyz_fn: Function to get current end-effector position
        cube_pos_fn: Function to get current cube position
    """
    ee_xyz = get_ee_xyz_fn(env)
    cube_xyz_live = cube_pos_fn(env)
    _, bin_xyz_live = env.get_obj_pose()

    phase_cfg = PHASES[fsm["phase"]]

    if phase_cfg["name"] != "descend":
        fsm["descend_xy_recover"] = False

    # Capture pick location on every approach-phase entry so retries track
    # the current cube position (cube can drift after failed attempts).
    if fsm["pick_xyz"] is None or (phase_cfg["name"] == "approach_cube" and fsm["tick"] == 0):
        fsm["pick_xyz"] = cube_xyz_live.copy()
    if fsm["bin_xyz"] is None:
        fsm["bin_xyz"] = np.asarray(bin_xyz_live, dtype=np.float32)

    cube_xyz = fsm["pick_xyz"]
    # Track live cube only in pre-grasp positioning phases.
    # For rotate/descend, keep a locked pick target so we don't chase a moving
    # cube that was already nudged by incidental contact.
    if phase_cfg["name"] in {"approach_cube", "align_cube"}:
        cube_xyz = cube_xyz_live.copy()
        fsm["pick_xyz"] = cube_xyz.copy()
    bin_xyz = fsm["bin_xyz"]
    target_fn = phase_cfg["target"]
    gripper = phase_cfg["gripper"]

    if FSM_DEBUG and fsm["tick"] == 0:
        print(
            f"[FSM DEBUG] ENTER phase={fsm['phase']} name={phase_cfg['name']} "
            f"gripper={gripper:.2f}"
        )

    # -------------------------------------------------
    # Update target position
    # -------------------------------------------------
    should_refresh_target = False
    if fsm["planned_phase"] != fsm["phase"]:
        should_refresh_target = True
    elif phase_cfg["name"] in REFRESH_PHASES and fsm["tick"] > 0:
        should_refresh_target = (fsm["tick"] % TARGET_REFRESH_EVERY) == 0

    if should_refresh_target:
        prev_target = fsm.get("phase_target_xyz")
        if target_fn is None:
            new_target = None
        else:
            new_target = np.asarray(target_fn(cube_xyz, bin_xyz), dtype=np.float32)

        fsm["phase_target_xyz"] = new_target
        fsm["planned_phase"] = fsm["phase"]

        # IMPORTANT: if target changes mid-phase, reset trajectory state so
        # `next_ee_waypoint` does not keep integrating toward stale traj_target.
        target_changed = False
        if (prev_target is None) != (new_target is None):
            target_changed = True
        elif prev_target is not None and new_target is not None:
            target_changed = float(np.linalg.norm(new_target - prev_target)) > 1e-4

        if target_changed:
            fsm["traj_start"] = None
            fsm["traj_target"] = None
            fsm["traj_progress"] = 0.0

    if target_fn is None:
        target_xyz = ee_xyz.copy()
    else:
        target_xyz = fsm["phase_target_xyz"]

    completion_target_xyz = target_xyz.copy() if target_xyz is not None else None

    align_phase_idx = 1

    # If explicit rotate phase is disabled, skip it entirely.
    if phase_cfg["name"] == "rotate_to_topdown" and not ENABLE_ROTATE_PHASE:
        if FSM_DEBUG and fsm["tick"] == 0:
            print("[FSM DEBUG] rotate_to_topdown disabled -> skipping to descend")
        fsm["phase"] = min(fsm["phase"] + 1, len(PHASES) - 1)
        fsm["planned_phase"] = None
        fsm["phase_target_xyz"] = None
        fsm["tick"] = 0
        fsm["traj_start"] = None
        fsm["traj_target"] = None
        fsm["traj_progress"] = 0.0
        fsm["descend_xy_recover"] = False
        action = np.zeros(env.n_arm_joints + 1, dtype=np.float32)
        action[-1] = PHASES[fsm["phase"]]["gripper"]
        return action

    # If rotation drifts too far from the approach anchor, re-run align before
    # attempting more orientation correction.
    if phase_cfg["name"] == "rotate_to_topdown":
        rotate_pos_err = float(np.linalg.norm(ee_xyz - target_xyz))
        if fsm["tick"] >= 5 and rotate_pos_err > DESCEND_REALIGN_XY_ERR:
            if FSM_DEBUG:
                print(
                    f"[FSM DEBUG] rotate drift -> align: pos_err={rotate_pos_err:.4f}>"
                    f"{DESCEND_REALIGN_XY_ERR:.4f}"
                )
            fsm["phase"] = align_phase_idx
            fsm["planned_phase"] = None
            fsm["phase_target_xyz"] = None
            fsm["tick"] = 0
            fsm["traj_start"] = None
            fsm["traj_target"] = None
            fsm["traj_progress"] = 0.0
            fsm["descend_xy_recover"] = True
            action = np.zeros(env.n_arm_joints + 1, dtype=np.float32)
            action[-1] = gripper
            return action

    # Never continue descent when lateral alignment is poor; go back to align.
    if phase_cfg["name"] == "descend":
        descend_xy_err = float(np.linalg.norm((ee_xyz - target_xyz)[:2]))
        if descend_xy_err > DESCEND_XY_RECOVER_ENTER:
            if FSM_DEBUG:
                print(
                    f"[FSM DEBUG] descend xy recover -> align: xy_err={descend_xy_err:.4f}>"
                    f"{DESCEND_XY_RECOVER_ENTER:.4f}"
                )
            fsm["phase"] = align_phase_idx
            fsm["planned_phase"] = None
            fsm["phase_target_xyz"] = None
            fsm["tick"] = 0
            fsm["traj_start"] = None
            fsm["traj_target"] = None
            fsm["traj_progress"] = 0.0
            fsm["descend_xy_recover"] = True
            action = np.zeros(env.n_arm_joints + 1, dtype=np.float32)
            action[-1] = gripper
            return action

    # If previous descend/lift failed, first raise vertically to safe hover height
    # at current XY to avoid dragging on/near tabletop before lateral movement.
    if fsm.get("recover_lift_pending") and phase_cfg["name"] == "approach_cube":
        safe_hover_z = float(cube_xyz[2] + APPROACH_Z_OFFSET)
        if ee_xyz[2] < safe_hover_z - RECOVERY_Z_MARGIN:
            target_xyz = np.array([ee_xyz[0], ee_xyz[1], safe_hover_z], dtype=np.float32)
            if FSM_DEBUG and (fsm["tick"] == 0 or (fsm["tick"] % FSM_DEBUG_EVERY) == 0):
                print(
                    f"[FSM DEBUG] recovery-lift active: ee_z={ee_xyz[2]:.4f} -> safe_hover_z={safe_hover_z:.4f}"
                )
        else:
            fsm["recover_lift_pending"] = False
            fsm["traj_start"] = None
            fsm["traj_target"] = None
            fsm["traj_progress"] = 0.0

    # Simple approach policy:
    # 1) move over the cube in XY while staying at a safe transit height,
    # 2) only then settle to the hover height above the cube.
    if phase_cfg["name"] == "approach_cube":
        hover_z = float(cube_xyz[2] + APPROACH_Z_OFFSET)
        transit_z = max(float(ee_xyz[2]), hover_z)
        final_hover_target = np.array([target_xyz[0], target_xyz[1], hover_z], dtype=np.float32)
        completion_target_xyz = final_hover_target.copy()
        xy_to_hover = float(np.linalg.norm(ee_xyz[:2] - final_hover_target[:2]))

        if xy_to_hover > 0.02:
            staged_target = np.array([final_hover_target[0], final_hover_target[1], transit_z], dtype=np.float32)
        else:
            staged_target = final_hover_target

        if float(np.linalg.norm(staged_target - target_xyz)) > 1e-4:
            fsm["traj_start"] = None
            fsm["traj_target"] = None
            fsm["traj_progress"] = 0.0

        target_xyz = staged_target

    # Keep XY alignment from approach and reuse it during descend and rotate.
    if phase_cfg["name"] == "approach_cube":
        fsm["approach_xy"] = target_xyz[:2].copy()
    elif phase_cfg["name"] in ["descend", "rotate_to_topdown"] and fsm.get("approach_xy") is not None:
        target_xyz = target_xyz.copy()
        target_xyz[:2] = fsm["approach_xy"]

    # -------------------------------------------------
    # Compute waypoint and solve IK with mink
    # -------------------------------------------------
    waypoint = next_ee_waypoint(env, target_xyz, get_ee_xyz_fn, speed=6.0)

    # During recovery-lift, move strictly vertically at current XY.
    if fsm.get("recover_lift_pending") and phase_cfg["name"] == "approach_cube":
        waypoint = target_xyz.copy()

    wp_err = waypoint - ee_xyz
    wp_dist = float(np.linalg.norm(wp_err))
    
    # Determine orientation strategy based on phase
    if phase_cfg["name"] == "rotate_to_topdown":
        # In rotation phase, focus on achieving top-down orientation while maintaining position
        use_top_down = True
    elif phase_cfg["name"] == "descend":
        # In descend phase, maintain top-down orientation
        use_top_down = ENFORCE_TOPDOWN_ON_DESCEND
    else:
        # Other phases: keep current orientation (no orientation constraint)
        use_top_down = False

    # For descend phase, use direct target instead of incremental stepping
    if phase_cfg["name"] == "descend":
        # Directly target the final position - this is cleaner and avoids oscillations
        waypoint = target_xyz.copy()
        
        # But limit downward speed for safety
        current_z = ee_xyz[2]
        target_z = target_xyz[2]
        if current_z - target_z > DESCEND_Z_STEP:
            waypoint[2] = current_z - DESCEND_Z_STEP

    q_target = plan_ik_with_mink(fsm, waypoint, use_top_down=use_top_down)

    ee_site_name = fsm.get("ee_site_name")
    if not ee_site_name:
        solver = fsm.get("mink_solver")
        ee_site_name = getattr(solver, "ee_site_name", DEFAULT_EE_SITE_NAME)

    if q_target is None:
        # If mink fails, stay at current position
        print(f"[FSM] Mink IK failed for phase {phase_cfg['name']}, holding position")
        q_target = env.get_joint_state()[:env.n_arm_joints].copy()
        fsm["ik_failures"] += 1
    else:
        fsm["ik_failures"] = 0

    # -------------------------------------------------
    # Generate action
    # -------------------------------------------------
    q_current = env.get_joint_state()[:env.n_arm_joints]
    dq = np.clip(q_target - q_current, -MAX_DQ_STEP, MAX_DQ_STEP)

    action = np.zeros(env.n_arm_joints + 1, dtype=np.float32)
    action[:env.n_arm_joints] = dq
    action[-1] = gripper

    # -------------------------------------------------
    # Completion check
    # -------------------------------------------------
    done = False
    done_reason = ""
    xy_err = np.nan
    z_err = np.nan
    dist = np.nan
    
    if phase_cfg["name"] == "rotate_to_topdown":
        # Special completion check for rotation phase
        current_quat = get_current_orientation(env, ee_site_name)
        angle_diff = quat_angle_between(current_quat, TOP_DOWN_QUAT)
        axis_err = topdown_axis_error(current_quat)
        
        # Also check position tolerance to ensure we're still at approach height
        err = ee_xyz - completion_target_xyz
        xy_err = np.linalg.norm(err[:2])
        z_err = abs(err[2])
        dist = np.linalg.norm(err)
        pos_err = dist
        
        done = (axis_err < ROTATION_THRESHOLD) and (pos_err < 0.025)
        done_reason = (
            f"axis_err={axis_err:.4f}<{ROTATION_THRESHOLD:.4f} "
            f"and pos_err={pos_err:.4f}<0.025"
        )
        
        if FSM_DEBUG and (fsm["tick"] % FSM_DEBUG_EVERY) == 0:
            print(
                f"[FSM DEBUG] rotation progress: axis_err={axis_err:.4f}, "
                f"quat_err={angle_diff:.4f}, pos_err={pos_err:.4f}"
            )
            
    elif phase_cfg["tol"] is not None:
        err = ee_xyz - completion_target_xyz
        xy_err = np.linalg.norm(err[:2])
        z_err = abs(err[2])
        dist = np.linalg.norm(err)

        if phase_cfg["name"] == "approach_cube":
            done = (xy_err < 0.015) and (z_err < 0.030)
            done_reason = f"xy_err={xy_err:.4f}<0.015 and z_err={z_err:.4f}<0.030"
        elif phase_cfg["name"] == "align_cube":
            done = (xy_err < 0.012) and (z_err < 0.020)
            done_reason = f"xy_err={xy_err:.4f}<0.012 and z_err={z_err:.4f}<0.020"
        elif phase_cfg["name"] == "descend":
            pos_ok = (xy_err < 0.012) and (z_err < 0.010)
            done = pos_ok
            done_reason = f"xy_err={xy_err:.4f}<0.012, z_err={z_err:.4f}<0.010"
        else:
            done = dist < phase_cfg["tol"]
            done_reason = f"dist={dist:.4f}<tol={phase_cfg['tol']:.4f}"
    else:
        done = fsm["tick"] >= phase_cfg["timeout"]
        done_reason = f"tick={fsm['tick']}>=timeout={phase_cfg['timeout']}"

    if FSM_DEBUG and (fsm["tick"] == 0 or (fsm["tick"] % FSM_DEBUG_EVERY) == 0):
        print(
            f"[FSM DEBUG] phase={phase_cfg['name']} tick={fsm['tick']} "
            f"ee={np.round(ee_xyz, 4)} target={np.round(target_xyz, 4)} "
            f"waypoint={np.round(waypoint, 4)} wp_dist={wp_dist:.4f} "
            f"xy_err={xy_err:.4f} z_err={z_err:.4f} dist={dist:.4f} done={done}"
        )

    timed_out = fsm["tick"] > phase_cfg["timeout"]
    allow_timeout_advance = False
    if timed_out:
        print(f"[FSM] timeout phase {phase_cfg['name']}")
        if FSM_DEBUG:
            print(
                f"[FSM DEBUG] TIMEOUT phase={phase_cfg['name']} tick={fsm['tick']} "
                f"ee={np.round(ee_xyz, 4)} target={np.round(target_xyz, 4)}"
            )

        # If align is very close but stalls on residual wrist/orientation motion,
        # accept and continue to rotation phase instead of restarting.
        if phase_cfg["name"] == "align_cube" and np.isfinite(xy_err) and np.isfinite(z_err):
            if (xy_err < 0.014) and (z_err < 0.025):
                done = True
                allow_timeout_advance = True
                done_reason = f"align timeout fallback: xy_err={xy_err:.4f}<0.014 and z_err={z_err:.4f}<0.025"
            else:
                done = False
                done_reason = f"timeout tick={fsm['tick']}>{phase_cfg['timeout']}"
        elif phase_cfg["name"] == "rotate_to_topdown":
            # If rotation times out but we're close enough, proceed to descend anyway
            current_quat = get_current_orientation(env, ee_site_name)
            axis_err = topdown_axis_error(current_quat)
            if axis_err < ROTATION_THRESHOLD * 2.0:  # Within 2x threshold
                done = True
                allow_timeout_advance = True
                done_reason = f"rotate timeout fallback: axis_err={axis_err:.4f}"
            else:
                done = False
                done_reason = f"timeout tick={fsm['tick']}>{phase_cfg['timeout']}"
        else:
            done = False
            done_reason = f"timeout tick={fsm['tick']}>{phase_cfg['timeout']}"

    # Critical phases must not advance on timeout; restart pick attempt instead.
    if timed_out and (not allow_timeout_advance) and phase_cfg["name"] in {"approach_cube", "align_cube", "rotate_to_topdown", "descend", "lift"}:
        if phase_cfg["name"] in {"rotate_to_topdown", "descend"}:
            print(f"[FSM] {phase_cfg['name']} timeout -> return to align")
            fsm["phase"] = 1
            fsm["planned_phase"] = None
            fsm["phase_target_xyz"] = None
            fsm["tick"] = 0
            fsm["traj_start"] = None
            fsm["traj_target"] = None
            fsm["traj_progress"] = 0.0
            fsm["descend_xy_recover"] = True
            return action

        print(f"[FSM] {phase_cfg['name']} timeout -> restart from approach")
        if phase_cfg["name"] in {"descend", "lift"}:
            fsm["recover_lift_pending"] = True
        fsm["phase"] = 0
        fsm["planned_phase"] = None
        fsm["phase_target_xyz"] = None
        fsm["pick_xyz"] = None
        fsm["tick"] = 0
        fsm["traj_start"] = None
        fsm["traj_target"] = None
        fsm["traj_progress"] = 0.0
        fsm["approach_xy"] = None
        fsm["descend_xy_recover"] = False
        fsm["lift_fail_streak"] = 0
        return action

    # -------------------------------------------------
    # Grasp validation
    # -------------------------------------------------
    if phase_cfg["name"] == "lift":
        LIFT_CHECK_DELAY = 12
        LIFT_MIN_Z_DELTA = 0.004
        LIFT_FAIL_STREAK_MAX = 10

        if fsm["tick"] > LIFT_CHECK_DELAY:
            if cube_lifted(env, cube_xyz, cube_pos_fn, min_z_delta=LIFT_MIN_Z_DELTA):
                fsm["lift_fail_streak"] = 0
            else:
                fsm["lift_fail_streak"] += 1

            if fsm["lift_fail_streak"] >= LIFT_FAIL_STREAK_MAX:
                print("[FSM] grasp failed, restarting")
                fsm["recover_lift_pending"] = True
                fsm["phase"] = 0
                fsm["planned_phase"] = None
                fsm["phase_target_xyz"] = None
                fsm["pick_xyz"] = None
                fsm["tick"] = 0
                fsm["traj_start"] = None
                fsm["traj_target"] = None
                fsm["traj_progress"] = 0.0
                fsm["descend_xy_recover"] = False
                fsm["lift_fail_streak"] = 0
                return action

    # -------------------------------------------------
    # Phase transition
    # -------------------------------------------------
    if done and fsm["phase"] < len(PHASES) - 1:
        if FSM_DEBUG:
            print(
                f"[FSM DEBUG] COMPLETE phase={phase_cfg['name']} "
                f"reason=({done_reason}) -> next={PHASES[fsm['phase'] + 1]['name']}"
            )
        fsm["phase"] += 1
        if fsm["phase"] < len(PHASES) and PHASES[fsm["phase"]]["name"] == "rotate_to_topdown" and not ENABLE_ROTATE_PHASE:
            if FSM_DEBUG:
                print("[FSM DEBUG] skipping rotate_to_topdown phase")
            fsm["phase"] += 1
        fsm["planned_phase"] = None
        fsm["phase_target_xyz"] = None
        if PHASES[fsm["phase"]]["name"] not in ["descend", "rotate_to_topdown"]:
            fsm["approach_xy"] = None
        if PHASES[fsm["phase"]]["name"] != "descend":
            fsm["descend_xy_recover"] = False
        fsm["lift_fail_streak"] = 0
        fsm["tick"] = 0
        fsm["traj_start"] = None
        fsm["traj_progress"] = 0.0
    else:
        if FSM_DEBUG and done and fsm["phase"] == len(PHASES) - 1:
            print(
                f"[FSM DEBUG] COMPLETE terminal phase={phase_cfg['name']} reason=({done_reason})"
            )
        fsm["tick"] += 1

    return action


# -------------------------------------------------------------
# USAGE EXAMPLE
# -------------------------------------------------------------

def setup_so101_controller(env, arm_joint_names=None, ee_site_name=DEFAULT_EE_SITE_NAME):
    """
    Setup function to initialize FSM with mink for SO-101.
    
    Args:
        env: Environment with model and data
        arm_joint_names: List of joint names (if None, will try to detect)
        ee_site_name: Name of end-effector site
        
    Returns:
        fsm: Initialized FSM state
    """
    # If joint names not provided, try to detect them
    if arm_joint_names is None:
        # SO-101 arm joint names (matches y_env.py robot_profile='so101')
        arm_joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
    
    # Create FSM
    fsm = make_fsm()

    resolved_ee_site_name = resolve_ee_site_name(env, ee_site_name)
    fsm["ee_site_name"] = resolved_ee_site_name

    # Compute a safe "home" Cartesian pose (used for unstow + return_home).
    # For SO101 resets, the arm starts stowed; we want a neutral pose over the workspace
    # before attempting pick-and-place.
    cube_xyz, bin_xyz = env.get_obj_pose()
    cube_xyz = np.asarray(cube_xyz, dtype=np.float32)
    bin_xyz = np.asarray(bin_xyz, dtype=np.float32)
    home_xy = 0.5 * (cube_xyz[:2] + bin_xyz[:2])
    home_z = float(max(cube_xyz[2], bin_xyz[2]) + APPROACH_Z_OFFSET + 0.05)
    fsm["home_xyz"] = np.asarray([home_xy[0], home_xy[1], home_z], dtype=np.float32)

    # Initialize and attach mink solver
    fsm["mink_solver"] = init_mink_for_so101(
        env=env,
        arm_joint_names=arm_joint_names,
        ee_site_name=resolved_ee_site_name
    )

    return fsm


# -------------------------------------------------------------
# SIMPLE 7-PHASE FSM STEP
# -------------------------------------------------------------

def _phase_target_xyz(fsm, phase_cfg, cube_xyz, bin_xyz):
    """
    Compute Cartesian target for the current phase.
    """
    name = phase_cfg["name"]
    if phase_cfg["target"] is None:
        if name in {"stow_to_home", "return_home"}:
            # Go back to recorded home pose
            return np.asarray(fsm["home_xyz"], dtype=np.float32)
        # For pure timing phases ("grip") stay where we are
        return None

    return np.asarray(phase_cfg["target"](cube_xyz, bin_xyz), dtype=np.float32)


def fsm_step(
    env,
    fsm,
    get_ee_xyz_fn,
    cube_pos_fn,
):
    """
    Very simple 7‑phase pick-and-place:
      1) above_block     – hover above cube, gripper open
      2) descend         – move straight down
      3) grip            – close gripper
      4) lift            – move straight up
      5) move_to_box     – move above box
      6) place_in_box    – lower and open gripper
      7) return_home     – go back to initial pose
    """
    ee_xyz = get_ee_xyz_fn(env)
    cube_xyz_live = cube_pos_fn(env)
    _, bin_xyz_live = env.get_obj_pose()

    # Cache pick and box positions once at episode start
    if fsm["pick_xyz"] is None:
        fsm["pick_xyz"] = cube_xyz_live.copy()
    if fsm["bin_xyz"] is None:
        fsm["bin_xyz"] = np.asarray(bin_xyz_live, dtype=np.float32)

    cube_xyz = fsm["pick_xyz"]
    bin_xyz = fsm["bin_xyz"]

    phase_cfg = PHASES[fsm["phase"]]
    phase_name = phase_cfg["name"]

    if FSM_DEBUG and (fsm["tick"] == 0 or (fsm["tick"] % FSM_DEBUG_EVERY) == 0):
        print(
            f"[FSM] phase={phase_name} idx={fsm['phase']} tick={fsm['tick']} "
            f"cube={np.round(cube_xyz, 3)} box={np.round(bin_xyz, 3)}"
        )

    # ------------------------------------------------------------------
    # Determine Cartesian target for this phase
    # ------------------------------------------------------------------
    target_xyz = _phase_target_xyz(fsm, phase_cfg, cube_xyz, bin_xyz)

    # For the approach-to-block phase we want a shaped path, not a straight
    # line in XYZ. The motion is:
    #   1) pure XY at a fixed high approach_z until over the cube,
    #   2) then pure Z down to the final hover height.
    if phase_name == "above_block" and target_xyz is not None:
        ee_z = float(ee_xyz[2])
        cube_hover_z = float(cube_xyz[2] + APPROACH_Z_OFFSET)

        # Initialise a fixed approach height on first entry to this phase.
        if fsm.get("approach_z") is None or fsm["tick"] == 0:
            margin = 0.05  # extra clearance above cube hover height
            fsm["approach_z"] = max(ee_z, cube_hover_z + margin)

        approach_z = float(fsm["approach_z"])

        # Stage selection based on current pose
        lateral_vec = target_xyz[:2] - ee_xyz[:2]
        lateral_dist = float(np.linalg.norm(lateral_vec))

        VERT_STEP = 0.035
        XY_STEP = 0.05

        if lateral_dist > 0.02:
            # Stage 1: move only in XY at fixed approach_z.
            if lateral_dist > 1e-6:
                direction_xy = lateral_vec / lateral_dist
            else:
                direction_xy = np.zeros(2, dtype=np.float32)
            step = min(XY_STEP, lateral_dist)
            new_xy = ee_xyz[:2] + step * direction_xy
            target_xyz = np.array(
                [new_xy[0], new_xy[1], approach_z],
                dtype=np.float32,
            )
        else:
            # Stage 2: we're roughly above the cube; descend vertically to hover.
            if ee_z > cube_hover_z + 1e-3:
                new_z = max(ee_z - VERT_STEP, cube_hover_z)
            else:
                new_z = cube_hover_z
            target_xyz = np.array(
                [target_xyz[0], target_xyz[1], new_z],
                dtype=np.float32,
            )

    # For the move-to-box phase we want the same shaped path:
    #   1) pure XY at a fixed high box_approach_z until over the box,
    #   2) then pure Z down to the box hover height.
    if phase_name == "move_to_box" and target_xyz is not None:
        ee_z = float(ee_xyz[2])
        box_hover_z = float(bin_xyz[2] + APPROACH_Z_OFFSET)

        if fsm.get("box_approach_z") is None or fsm["tick"] == 0:
            margin = 0.05
            fsm["box_approach_z"] = max(ee_z, box_hover_z + margin)

        box_approach_z = float(fsm["box_approach_z"])

        lateral_vec = target_xyz[:2] - ee_xyz[:2]
        lateral_dist = float(np.linalg.norm(lateral_vec))

        VERT_STEP = 0.035
        XY_STEP = 0.055

        if lateral_dist > 0.02:
            # Stage 1: move only in XY at fixed box_approach_z.
            if lateral_dist > 1e-6:
                direction_xy = lateral_vec / lateral_dist
            else:
                direction_xy = np.zeros(2, dtype=np.float32)
            step = min(XY_STEP, lateral_dist)
            new_xy = ee_xyz[:2] + step * direction_xy
            target_xyz = np.array(
                [new_xy[0], new_xy[1], box_approach_z],
                dtype=np.float32,
            )
        else:
            # Stage 2: we're roughly above the box; keep hover Z (no further descent here).
            target_xyz = np.array(
                [target_xyz[0], target_xyz[1], box_hover_z],
                dtype=np.float32,
            )

    # Gripper command for this phase
    gripper_cmd = float(phase_cfg["gripper"])
    # In place_in_box: keep gripper closed until arm has stopped at target, then release
    if phase_name == "place_in_box" and target_xyz is not None and phase_cfg.get("tol") is not None:
        place_dist = float(np.linalg.norm(ee_xyz - target_xyz))
        if place_dist >= phase_cfg["tol"]:
            gripper_cmd = 1.0  # still moving – keep closed
        # else use phase_cfg["gripper"] (0.0) once at target

    # ------------------------------------------------------------------
    # IK + action
    # ------------------------------------------------------------------
    if target_xyz is None:
        # No motion phase – hold joints, only drive gripper
        q_current = env.get_joint_state()[:env.n_arm_joints]
        dq = np.zeros_like(q_current, dtype=np.float32)
        waypoint = ee_xyz.copy()
    else:
        # For the approach phase, we already shaped target_xyz to move in
        # pure-Z / pure-XY segments; use it directly as the waypoint so the
        # path follows that shaping rather than a straight line.
        if phase_name == "above_block":
            waypoint = target_xyz.copy()
        elif phase_name == "place_in_box":
            # Direct target with limited Z step to avoid overshoot/oscillation
            # before dropping the block (same idea as descend phase).
            waypoint = np.array(target_xyz, dtype=np.float32)
            current_z = float(ee_xyz[2])
            target_z = float(target_xyz[2])
            if current_z - target_z > PLACE_DESCEND_Z_STEP:
                waypoint[2] = current_z - PLACE_DESCEND_Z_STEP
        else:
            waypoint = next_ee_waypoint(env, target_xyz, get_ee_xyz_fn, speed=6.0)

        q_target = plan_ik_with_mink(fsm, waypoint, use_top_down=True)

        if q_target is None:
            print(f"[FSM] Mink IK failed in phase '{phase_name}', holding position.")
            q_target = env.get_joint_state()[:env.n_arm_joints].copy()
            fsm["ik_failures"] += 1
        else:
            fsm["ik_failures"] = 0

        q_current = env.get_joint_state()[:env.n_arm_joints]
        dq = np.clip(q_target - q_current, -MAX_DQ_STEP, MAX_DQ_STEP)
        # In place_in_box, damp motion when close to target to avoid oscillation at drop
        if phase_name == "place_in_box" and target_xyz is not None:
            place_dist = float(np.linalg.norm(ee_xyz - target_xyz))
            if place_dist < PLACE_DAMP_DIST:
                scale = place_dist / PLACE_DAMP_DIST
                dq = dq * scale

    action = np.zeros(env.n_arm_joints + 1, dtype=np.float32)
    action[:env.n_arm_joints] = dq
    action[-1] = gripper_cmd

    # ------------------------------------------------------------------
    # Debug: monitor Z heights to understand approach behaviour
    # ------------------------------------------------------------------
    if FSM_DEBUG and (fsm["tick"] % FSM_DEBUG_EVERY) == 0:
        cube_z = float(cube_xyz[2])
        ee_z = float(ee_xyz[2])
        waypoint_z = float(waypoint[2]) if target_xyz is not None else ee_z
        print(
            f"[FSM Z] phase={phase_name} tick={fsm['tick']} "
            f"ee_z={ee_z:.4f} waypoint_z={waypoint_z:.4f} cube_z={cube_z:.4f}"
        )

    # ------------------------------------------------------------------
    # Phase completion
    # ------------------------------------------------------------------
    done = False
    if phase_cfg["tol"] is None:
        # Purely time‑based phase (e.g., "grip")
        done = fsm["tick"] >= phase_cfg["timeout"]
    else:
        err = ee_xyz - (target_xyz if target_xyz is not None else ee_xyz)
        dist = float(np.linalg.norm(err))
        done = dist < phase_cfg["tol"] or fsm["tick"] >= phase_cfg["timeout"]

        if FSM_DEBUG and (fsm["tick"] % FSM_DEBUG_EVERY) == 0:
            print(
                f"[FSM] phase={phase_name} dist={dist:.4f} "
                f"tol={phase_cfg['tol']:.4f} done={done}"
            )

    # Advance phase
    if done and fsm["phase"] < len(PHASES) - 1:
        fsm["phase"] += 1
        fsm["tick"] = 0
    else:
        fsm["tick"] += 1

    return action