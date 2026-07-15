import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
import mujoco
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy, r2quat, quat2r
import os
import copy
import glfw

class SimpleEnv:
    def __init__(self, 
                 xml_path,
                action_type='eef_pose', 
                state_type='joint_angle',
                robot_profile='so101',
                seed = None,
                pick_body_name='body_obj_mug_5',
                place_body_name='body_obj_plate_11',
                spawn_x_range=(0.20, 0.3),
                spawn_y_range=(0.02, 0.1),
                spawn_z_range=(0.815, 0.815),
                spawn_min_dist=0.2,
                spawn_xy_margin=0.0,
                spawn_fallback_min_dist=0.1):
        """
        args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            robot_profile: str, robot kinematic profile ('omy', 'so100', 'so101')
            seed: int, seed for random number generator
            pick_body_name: str, body name for the pick object in the MuJoCo model
            place_body_name: str, body name for the place/target object in the MuJoCo model
            spawn_x_range: tuple[float, float], x sampling range for objects
            spawn_y_range: tuple[float, float], y sampling range for objects
            spawn_z_range: tuple[float, float], z sampling range for objects
            spawn_min_dist: float, minimum distance between sampled objects
            spawn_xy_margin: float, xy margin applied to ranges during sampling
            spawn_fallback_min_dist: float, retry distance when initial sampling fails
        """
        # Load the xml file
        self.env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type
        self.pick_body_name = pick_body_name
        self.place_body_name = place_body_name
        self.spawn_x_range = spawn_x_range
        self.spawn_y_range = spawn_y_range
        self.spawn_z_range = spawn_z_range
        self.spawn_min_dist = spawn_min_dist
        self.spawn_xy_margin = spawn_xy_margin
        self.spawn_fallback_min_dist = spawn_fallback_min_dist
        self.spawn_bin_exclusion_margin = 0.015
        self.robot_profile = robot_profile
        # Pick-and-place gating: only count success after the target object has
        # clearly been lifted off the table at least once.
        self._target_was_lifted = False
        self._target_rest_height = None
        self._lift_height_delta = 0.035

        if robot_profile == 'omy':
            self.joint_names = ['joint1',
                        'joint2',
                        'joint3',
                        'joint4',
                        'joint5',
                        'joint6',]
            self.tcp_body_name = 'tcp_link'
            self.gripper_joint_name = 'rh_r1'
            self.gripper_actuator_scales = np.array([1.0, 0.8, 1.0, 0.8], dtype=np.float32)
            self.gripper_max_rad = 1.0   # preserve existing OMY behaviour
            self.gripper_continuous = False
            # After placing, the OMY arm must be retracted above this Z (metres) to
            # signal episode success; prevents false positives during the pick phase.
            self.success_height_threshold = 0.9
        elif robot_profile in ['so100', 'so101']:
            self.joint_names = [
                'shoulder_pan',
                'shoulder_lift',
                'elbow_flex',
                'wrist_flex',
                'wrist_roll',
            ]
            self.tcp_body_name = 'gripper'
            self.gripper_joint_name = 'gripper'
            self.gripper_actuator_scales = np.array([1.0], dtype=np.float32)
            # Gripper joint range: [-0.17453, 1.74533] rad.  Action is normalised
            # to [0, 1] where 0 = fully open and 1 = fully closed.
            self.gripper_max_rad = 1.74533
            self.gripper_continuous = True
            # SO100/SO101: arm-retract height is checked relative to the bin
            # geometry inside check_success() rather than as an absolute value.
            # Set to None here to select the SO100/SO101 code path.
            self.success_height_threshold = None
            # Conservative XY reach limit for table sampling relative to robot base.
            self.spawn_reach_radius = 0.50
        else:
            raise ValueError(f"Unsupported robot_profile: {robot_profile}")

        self.n_arm_joints = len(self.joint_names)
        self.n_gripper_actuators = len(self.gripper_actuator_scales)
        self._prev_key_state = {}
        self._teleop_debug_counter = 0
        # Disable gripper pad debug overlays by default; they are useful for
        # debugging contact geometry but visually noisy during data collection.
        self.show_gripper_pad_debug = False
        # Mouse jogging state (middle mouse button drag)
        self._mouse_last_x = None
        self._mouse_last_y = None
        # Mocap IK tracking state
        self._mocap_body_id = -1
        self._mocap_id = -1
        self._last_mocap_pos = None
        self._last_mocap_quat = None
        self.joint_mins = np.array([self.env.model.joint(name).range[0] for name in self.joint_names], dtype=np.float32)
        self.joint_maxs = np.array([self.env.model.joint(name).range[1] for name in self.joint_names], dtype=np.float32)

        available_joint_names = set(self.env.joint_names)
        missing_joints = [jname for jname in self.joint_names if jname not in available_joint_names]
        if missing_joints:
            raise ValueError(
                f"robot_profile='{self.robot_profile}' is incompatible with xml_path='{xml_path}'. "
                f"Missing joints: {missing_joints}. "
                f"Available joints: {self.env.joint_names}"
            )

        if self.gripper_continuous:
            gripper_joint = self.env.model.joint(self.gripper_joint_name)
            gripper_range = np.asarray(gripper_joint.range, dtype=np.float32)
            # SO100/SO101 jaw kinematics are inverted relative to the task-space API:
            # larger raw joint values open the jaw wider, while callers expect
            # 0 = open and 1 = closed.
            self.gripper_closed_rad = float(gripper_range[0])
            self.gripper_open_rad = float(gripper_range[1])
            self.gripper_motion_span = max(
                self.gripper_open_rad - self.gripper_closed_rad, 1e-6
            )
        else:
            self.gripper_closed_rad = float(self.gripper_max_rad)
            self.gripper_open_rad = 0.0
            self.gripper_motion_span = max(self.gripper_closed_rad - self.gripper_open_rad, 1e-6)

        self._gripper_pad_debug_geoms = [
            ("fixed_jaw_pad", [1.0, 0.15, 0.15, 0.45]),
            ("moving_jaw_pad", [0.15, 0.55, 1.0, 0.45]),
            ("gripper_palm_pad", [1.0, 0.8, 0.15, 0.30]),
        ]

        # Will be populated on each reset with the sampled spawn configuration
        # of all movable block objects (excluding the fixed bin/plate).
        self.spawn_obj_names = []
        self.spawn_obj_xyzs = None

        self.init_viewer()
        self.reset(seed)

    @property
    def model(self):
        """Expose the underlying MuJoCo model directly."""
        return self.env.model

    @property
    def data(self):
        """Expose the underlying MuJoCo data directly."""
        return self.env.data

    def _key_is_down(self, key):
        if self.env.is_key_pressed_repeat(key=key):
            return True

        viewer = getattr(self.env, 'viewer', None)
        window = getattr(viewer, 'window', None)
        if window is None:
            return False

        try:
            return glfw.get_key(window, key) == glfw.PRESS
        except Exception:
            return False

    def _key_is_pressed_once(self, key):
        is_down = self._key_is_down(key)
        was_down = self._prev_key_state.get(key, False)
        self._prev_key_state[key] = is_down
        return is_down and not was_down

    def _make_gripper_ctrl(self, gripper_fraction):
        """
        Convert the normalized task-space command (0=open, 1=closed) to raw
        actuator targets.
        """
        gripper_fraction = float(np.clip(gripper_fraction, 0.0, 1.0))
        if self.gripper_continuous:
            gripper_target = self.gripper_open_rad - gripper_fraction * self.gripper_motion_span
        else:
            gripper_target = gripper_fraction * self.gripper_max_rad
        gripper_cmd = np.array(
            [gripper_target] * self.n_gripper_actuators,
            dtype=np.float32,
        )
        gripper_cmd *= self.gripper_actuator_scales
        return gripper_cmd

    def _get_bin_exclusion_bounds(self):
        """
        Return a rectangular XY exclusion zone for the fixed bin.
        """
        try:
            body_names = self.env.get_body_names(prefix='body_obj_')
            if 'body_obj_bin' not in body_names:
                return None
            p_bin = self.env.get_p_body('body_obj_bin')
        except Exception:
            return None

        # obj_bin.xml outer footprint half-size is 0.06m in x and y.
        half_extent = 0.06 + self.spawn_bin_exclusion_margin
        return (
            p_bin[0] - half_extent,
            p_bin[0] + half_extent,
            p_bin[1] - half_extent,
            p_bin[1] + half_extent,
        )

    def _is_within_reach_xy(self, x, y):
        """
        Check whether an XY point is inside the SO100/SO101 reachable table area.
        """
        if self.robot_profile not in ['so100', 'so101']:
            return True
        try:
            p_base = self.env.get_p_body('base')
        except Exception:
            p_base = np.zeros(3, dtype=np.float32)
        d_xy = np.linalg.norm(np.array([x - p_base[0], y - p_base[1]], dtype=np.float32))
        return bool(d_xy <= self.spawn_reach_radius)

    def _sample_object_xyzs(self, n_obj, min_dist):
        """
        Sample object positions while respecting reachability and bin exclusion.
        """
        x_min, x_max = self.spawn_x_range
        y_min, y_max = self.spawn_y_range
        z_min, z_max = self.spawn_z_range
        m = self.spawn_xy_margin
        x_lo, x_hi = x_min + m, x_max - m
        y_lo, y_hi = y_min + m, y_max - m
        if x_lo > x_hi or y_lo > y_hi:
            raise ValueError("spawn_xy_margin is too large for the configured spawn range.")

        bin_bounds = self._get_bin_exclusion_bounds()
        xyzs = []
        max_attempts = 5000
        attempts = 0
        while len(xyzs) < n_obj and attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(x_lo, x_hi)
            y = np.random.uniform(y_lo, y_hi)
            z = np.random.uniform(z_min, z_max)
            cand = np.array([x, y, z], dtype=np.float32)

            if not self._is_within_reach_xy(x, y):
                continue

            if bin_bounds is not None:
                bx0, bx1, by0, by1 = bin_bounds
                if (bx0 <= x <= bx1) and (by0 <= y <= by1):
                    continue

            if any(np.linalg.norm(cand - other) < min_dist for other in xyzs):
                continue

            xyzs.append(cand)

        if len(xyzs) != n_obj:
            raise ValueError(
                f"Could not sample {n_obj} objects within constrained spawn region "
                f"after {max_attempts} attempts."
            )
        return np.stack(xyzs, axis=0)

    def _sample_object_xyzs_with_retries(self, n_obj):
        """
        Sample object positions with progressively relaxed spacing constraints.
        """
        candidate_min_dists = []
        for value in [
            self.spawn_min_dist,
            self.spawn_fallback_min_dist,
            0.08,
            0.06,
        ]:
            value = float(value)
            if value < 0:
                continue
            if not any(np.isclose(value, existing) for existing in candidate_min_dists):
                candidate_min_dists.append(value)

        last_exc = None
        for idx, min_dist in enumerate(candidate_min_dists):
            try:
                if idx > 0:
                    print(f"[SimpleEnv.reset] retrying with min_dist={min_dist}")
                return self._sample_object_xyzs(n_obj=n_obj, min_dist=min_dist)
            except ValueError as exc:
                last_exc = exc
                print(f"[SimpleEnv.reset] object sampling failed: {exc}")

        raise last_exc

    def _teleop_debug_status(self):
        key_map = {
            'W': glfw.KEY_W,
            'A': glfw.KEY_A,
            'S': glfw.KEY_S,
            'D': glfw.KEY_D,
            'R': glfw.KEY_R,
            'F': glfw.KEY_F,
            'Q': glfw.KEY_Q,
            'E': glfw.KEY_E,
            'LEFT': glfw.KEY_LEFT,
            'RIGHT': glfw.KEY_RIGHT,
            'UP': glfw.KEY_UP,
            'DOWN': glfw.KEY_DOWN,
            'SPACE': glfw.KEY_SPACE,
            'Z': glfw.KEY_Z,
        }
        active_keys = [name for name, key in key_map.items() if self._key_is_down(key)]

        viewer = getattr(self.env, 'viewer', None)
        window = getattr(viewer, 'window', None)
        focused = False
        if window is not None:
            try:
                focused = bool(glfw.get_window_attrib(window, glfw.FOCUSED))
            except Exception:
                focused = False

        return focused, active_keys

    def _get_mouse_delta(self):
        """Poll middle mouse button and return (dx, dy, ctrl_held) cursor pixel deltas.
        Resets tracking when button is released so there is no position jump on re-engage."""
        viewer = getattr(self.env, 'viewer', None)
        window = getattr(viewer, 'window', None)
        if window is None:
            self._mouse_last_x = None
            self._mouse_last_y = None
            return 0.0, 0.0, False
        try:
            middle_held = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
            xpos, ypos = glfw.get_cursor_pos(window)
            ctrl_held = (
                glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or
                glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
            )
        except Exception:
            self._mouse_last_x = None
            self._mouse_last_y = None
            return 0.0, 0.0, False
        if not middle_held:
            self._mouse_last_x = None
            self._mouse_last_y = None
            return 0.0, 0.0, ctrl_held
        if self._mouse_last_x is None:
            # First frame of hold — anchor position, no delta yet
            self._mouse_last_x = xpos
            self._mouse_last_y = ypos
            return 0.0, 0.0, ctrl_held
        dx = xpos - self._mouse_last_x
        dy = ypos - self._mouse_last_y
        self._mouse_last_x = xpos
        self._mouse_last_y = ypos
        return dx, dy, ctrl_held

    def init_viewer(self):
        '''
        Initialize the viewer
        '''
        self.env.reset()
        try:
            p_base = self.env.get_p_body('base')
            p_tcp = self.env.get_p_body(self.tcp_body_name)
            # Start the free camera close to the front workspace instead of the
            # default far zoomed-out view.
            lookat = 0.5 * (p_base + p_tcp)
        except Exception:
            lookat = np.array([0.25, 0.0, 0.95], dtype=np.float32)
        self.env.init_viewer(
            distance=0.9,
            azimuth=180,
            elevation=-20,
            lookat=lookat,
            transparent=False,
            black_sky=True,
            use_rgb_overlay=False,
            loc_rgb_overlay='top right',
        )
        # Pre-assign pert.select to the mocap target body so that
        # Ctrl + right-drag immediately moves it, with no click-to-select needed.
        self._init_mocap_pert()

    def _init_mocap_pert(self):
        '''Find the 'target' mocap body and assign it as the permanent pert selection.'''
        try:
            body_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
            self._mocap_body_id = body_id
            self._mocap_id = (
                int(self.env.model.body_mocapid[body_id])
                if body_id >= 0 else -1
            )
            if body_id >= 0 and self._mocap_id >= 0:
                viewer = getattr(self.env, 'viewer', None)
                if viewer is not None and hasattr(viewer, 'pert'):
                    # Only set select — mjv_initPerturb is called by _mouse_button_callback
                    # when the user actually starts Ctrl+dragging, by which point xpos is valid.
                    viewer.pert.select = body_id
                print(f'[mocap] target body_id={body_id} mocap_id={self._mocap_id} — Ctrl+right-drag to move arm')
            else:
                print('[mocap] no target body found — mocap drag disabled')
        except Exception as e:
            print(f'[mocap init] {e}')
            self._mocap_body_id = -1
            self._mocap_id = -1
    def reset(self, seed = None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        if seed is not None:
            np.random.seed(seed=seed)
        
        # Set robot-specific initial configurations and home positions
        if self.robot_profile == 'so101':
            # SO101 stowed: joint 1 = 0, joint 2 = clockwise limit, joint 3 = anti-clockwise limit,
            # wrist = 90°. Uses model limits for j2/j3 so stowed stays correct if model changes.
            q_zero = np.array(
                [
                    0.0,
                    self.joint_mins[1],   # shoulder_lift at clockwise limit
                    self.joint_maxs[2],   # elbow_flex at anti-clockwise limit
                    np.pi / 2,            # wrist_flex 90°
                    0.0,                  # wrist_roll
                ],
                dtype=np.float32,
            )
            self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)
        else:
            # Use IK for Cartesian / eef_pose controlled robots
            if self.robot_profile == 'so100':
                q_init = np.zeros(self.n_arm_joints, dtype=np.float32)
                p_trgt = np.array([0.25, -0.35, 0.95])
                R_trgt = rpy2r(np.deg2rad([90, 0, 90]))
            else:  # omy
                q_init = np.zeros(self.n_arm_joints, dtype=np.float32)
                p_trgt = np.array([0.3, 0.0, 1.0])
                R_trgt = rpy2r(np.deg2rad([90, 0, 90]))

            q_zero,ik_err_stack,ik_info = solve_ik(
                env = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = self.tcp_body_name,
                q_init       = q_init,
                p_trgt       = p_trgt,
                R_trgt       = R_trgt,
            )
            self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)

        # Sync mocap target sphere to current TCP position and orientation
        try:
            if self._mocap_id >= 0:
                mujoco.mj_forward(self.env.model, self.env.data)
                p_tcp, R_tcp = self.env.get_pR_body(body_name=self.tcp_body_name)
                self.env.data.mocap_pos[self._mocap_id] = p_tcp
                # Convert rotation matrix to quaternion for mocap_quat (MuJoCo: [w,x,y,z])
                q_tcp = r2quat(R_tcp)
                self.env.data.mocap_quat[self._mocap_id] = q_tcp
                self._last_mocap_pos = p_tcp.copy()
                self._last_mocap_quat = q_tcp.copy()
        except Exception:
            pass

        # Legacy block: also sync via body_id lookup if _init_mocap_pert not yet called
        try:
            target_body_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
            if target_body_id >= 0:
                mocap_id = self.env.model.body_mocapid[target_body_id]
                if mocap_id >= 0 and self._mocap_id < 0:
                    self.env.forward(increase_tick=False)
                    p_tcp, _ = self.env.get_pR_body(body_name=self.tcp_body_name)
                    self.env.data.mocap_pos[mocap_id] = p_tcp
        except Exception:
            pass

        # Randomise pick-object positions.  The plate/target body is fixed in
        # the scene XML (no free joint) and is never touched here.
        all_obj_names = self.env.get_body_names(prefix='body_obj_')
        obj_names = [n for n in all_obj_names if n != self.place_body_name]
        n_obj = len(obj_names)
        obj_xyzs = self._sample_object_xyzs_with_retries(n_obj=n_obj)

        # Record and apply the initial spawn configuration of all movable blocks.
        self.spawn_obj_names = list(obj_names)
        self.spawn_obj_xyzs = obj_xyzs.copy()
        for obj_idx in range(n_obj):
            self.env.set_p_base_body(body_name=obj_names[obj_idx], p=obj_xyzs[obj_idx, :])
            self.env.set_R_base_body(body_name=obj_names[obj_idx], R=np.eye(3, 3))

        # Clear residual dynamics so newly spawned objects start at rest
        self.env.data.qvel[:] = 0.0
        self.env.data.qacc[:] = 0.0
        if self.env.data.ctrl is not None and self.env.data.ctrl.size > 0:
            self.env.data.ctrl[:] = 0.0
            # Initialise gripper in the closed state at reset so scripted episodes
            # visibly begin by opening during the descend phase.
            self.env.data.ctrl[-self.n_gripper_actuators:] = self._make_gripper_ctrl(1.0)
        self.env.forward(increase_tick=False)

        # Let objects settle under gravity before starting teleoperation
        settle_ctrl = np.zeros(self.env.n_ctrl, dtype=np.float32)
        if self.env.data.ctrl is not None and self.env.data.ctrl.size > 0:
            self.env.data.ctrl[:] = 0.0
            # Keep gripper closed while objects settle under gravity.
            settle_ctrl[-self.n_gripper_actuators:] = self._make_gripper_ctrl(1.0)
            self.env.data.ctrl[:] = settle_ctrl
        for _ in range(20):
            self.env.step(settle_ctrl)

        # Set the initial pose of the robot
        self.last_q = copy.deepcopy(q_zero)
        self.prev_q = copy.deepcopy(q_zero)
        # Append closed-gripper command to the joint configuration.
        self.q = np.concatenate([q_zero, self._make_gripper_ctrl(1.0)])
        self.p0, self.R0 = self.env.get_pR_body(body_name=self.tcp_body_name)
        mug_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_init_pose, plate_init_pose],dtype=np.float32)
        for _ in range(100):
            self.step_env()
        print("DONE INITIALIZATION")
        # gripper_state is a float in [0, 1]: 0 = fully open, 1 = fully closed.
        # Start episodes with the gripper closed so the first scripted/teleop
        # action does not immediately open it.
        self.gripper_state = 1.0
        self._target_was_lifted = False
        # Record post-settling baseline height for the target object.  Lift
        # detection is measured relative to this baseline (not absolute z).
        self._target_rest_height = float(self.env.get_p_body(self.pick_body_name)[2])
        self.past_chars = []

    def step(self, action):
        '''
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
        returns:
            state: np.array, state of the environment after taking the action
                - ee_pose: [px,py,pz,r,p,y]
                - joint_angle: [j1,j2,j3,j4,j5,j6]

        '''
        prev_q = copy.deepcopy(self.last_q)
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            
            # Adjust IK parameters based on robot profile
            if self.robot_profile in ('so100', 'so101'):
                ik_stepsize = 0.3
                ik_eps = 0.5
                max_ik_tick = 50
            else:  # omy
                ik_stepsize = 1.0
                ik_eps = 1e-2
                max_ik_tick = 50
            
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = self.tcp_body_name,
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = max_ik_tick,
                ik_stepsize        = ik_stepsize,
                ik_eps             = ik_eps,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
                restore_state      = False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:self.n_arm_joints] + prev_q
        elif self.action_type == 'joint_angle':
            q = action[:self.n_arm_joints]
        else:
            raise ValueError('action_type not recognized')

        q = np.clip(q, self.joint_mins, self.joint_maxs)
        self.prev_q = prev_q
        self.last_q = copy.deepcopy(q)
        
        # Map the task-space convention 0=open, 1=closed to the underlying
        # actuator command. SO100/SO101 use the opposite raw joint direction:
        # larger angles open the jaw wider.
        gripper_cmd = self._make_gripper_ctrl(action[-1])
        self.compute_q = q
        q = np.concatenate([q, gripper_cmd])

        self.q = q
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq =  self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def step_env(self):
        # Ensure perturbation target stays locked to mocap, preventing accidental
        # selection of blocks or gripper which can cause free joint corruption
        if self._mocap_body_id >= 0:
            viewer = getattr(self.env, 'viewer', None)
            if viewer is not None and hasattr(viewer, 'pert'):
                if viewer.pert.select != self._mocap_body_id:
                    viewer.pert.select = self._mocap_body_id
        
        self.env.step(self.q)

    def grab_image(self):
        '''
        grab images from the environment
        returns:
            rgb_top:   np.array, rgb image from the topview camera
            rgb_front: np.array, rgb image from the frontview camera
        '''
        # Primary stream: top-down view
        self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name='topview')
        # Secondary stream: frontview camera (uses MuJoCo 'sideview' handle)
        self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name='sideview')
        # Keep rgb_side in sync with the secondary stream for overlays in teleop.
        self.rgb_side = self.rgb_ego.copy()
        return self.rgb_agent, self.rgb_ego
        

    def render(self, teleop=False):
        '''
        Render the environment
        '''
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name=self.tcp_body_name)
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        if self.show_gripper_pad_debug:
            self._plot_gripper_contact_pads()

        # Topview camera is always shown in the top-right.
        rgb_top_view = add_title_to_img(self.rgb_agent, text='Top View', shape=(640,480))
        self.env.viewer_rgb_overlay(rgb_top_view, loc='top right')

        # Frontview camera:
        # - During teleoperation (data collection), show it in the top-left.
        # - During dataset replay, show it in the bottom-right.
        if self.rgb_ego is not None:
            if teleop:
                rgb_front_view = add_title_to_img(self.rgb_ego, text='Front View', shape=(640,480))
                self.env.viewer_rgb_overlay(rgb_front_view, loc='top left')
            else:
                rgb_front_view = add_title_to_img(self.rgb_ego, text='Front View', shape=(640,480))
                self.env.viewer_rgb_overlay(rgb_front_view, loc='bottom right')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list(as_text=True)))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list(as_text=True)))
            focused, active_keys = self._teleop_debug_status()
            self.env.viewer_text_overlay(text1='Window Focus', text2='YES' if focused else 'NO (click viewer)')
            self.env.viewer_text_overlay(text1='Active Keys (raw)', text2='%s' % active_keys)
        self.env.render()

    def _plot_gripper_contact_pads(self):
        """
        Draw translucent boxes over the gripper collision pads so they are easy
        to inspect in the viewer.
        """
        for geom_name, rgba in self._gripper_pad_debug_geoms:
            geom_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            if geom_id < 0:
                continue
            p = self.env.data.geom_xpos[geom_id].copy()
            R = self.env.data.geom_xmat[geom_id].reshape(3, 3).copy()
            size = self.env.model.geom_size[geom_id].copy()
            self.env.plot_box(
                p=p,
                R=R,
                xlen=2.0 * size[0],
                ylen=2.0 * size[1],
                zlen=2.0 * size[2],
                rgba=rgba,
                label=geom_name,
            )

    def get_joint_state(self):
        '''
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + normalised gripper position
            [j1,j2,j3,j4,j5,j6,gripper]  where gripper ∈ [0,1] (0=open, 1=closed)
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper_val = self._get_gripper_fraction()
        return np.concatenate([qpos, [gripper_val]], dtype=np.float32)
    
    def teleop_robot(self):
        '''
        Teleoperate the robot using keyboard or middle mouse button.
        returns:
            action: np.array, action to take
            done: bool, True if the user wants to reset the teleoperation

        Mouse (middle button drag):
            Drag left/right : +/- Y  (world frame)
            Drag up/down    : +/- Z  (world frame)
            Ctrl + drag L/R : +/- X  (world frame)

        Keyboard:
            W/S : +/- X (Forward/Backward)
            A/D : +/- Y (Left/Right)
            R/F : +/- Z (Up/Down)
            Q/E : Yaw left/right
            UP/DOWN     : Pitch up/down
            LEFT/RIGHT  : Roll left/right
            SPACE : gripper open/close
            Z     : reset
        '''
        # Joint-space control (SO101 default)
        if self.action_type == 'delta_joint_angle':
            return self._teleop_joint_space()

        # Mocap / IK Cartesian control — drag the red target sphere
        if self.action_type == 'eef_pose':
            return self._teleop_mocap()

        # Fallback: keyboard-only Cartesian control
        dpos = np.zeros(3)
        drot = np.eye(3)

        # Mouse jogging: middle button drag
        # No Ctrl : dx -> Y,  -dy -> Z
        # With Ctrl: dx -> X, -dy -> Z
        _mouse_pos_scale = 0.00015  # metres per pixel
        _mdx, _mdy, _mctrl = self._get_mouse_delta()
        if abs(_mdx) > 0.2 or abs(_mdy) > 0.2:
            if _mctrl:
                dpos[0] += _mdx * _mouse_pos_scale
            else:
                dpos[1] += _mdx * _mouse_pos_scale
            dpos[2] -= _mdy * _mouse_pos_scale

        # Keyboard: WASD/RF/QE/arrows
        if self._key_is_down(glfw.KEY_W):
            dpos += np.array([0.007,0.0,0.0])  # Forward (+X)
        if self._key_is_down(glfw.KEY_S):
            dpos += np.array([-0.007,0.0,0.0])  # Backward (-X)
        if self._key_is_down(glfw.KEY_A):
            dpos += np.array([0.0,-0.007,0.0])  # Left (-Y)
        if self._key_is_down(glfw.KEY_D):
            dpos += np.array([0.0,0.007,0.0])  # Right (+Y)
        if self._key_is_down(glfw.KEY_R):
            dpos += np.array([0.0,0.0,0.007])  # Up (+Z)
        if self._key_is_down(glfw.KEY_F):
            dpos += np.array([0.0,0.0,-0.007])  # Down (-Z)
        if self._key_is_down(glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]  # Yaw left
        if self._key_is_down(glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]  # Yaw right
        if  self._key_is_down(glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]  # Roll left
        if  self._key_is_down(glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]  # Roll right
        if self._key_is_down(glfw.KEY_UP):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]  # Pitch up
        if self._key_is_down(glfw.KEY_DOWN):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]  # Pitch down
        if self._key_is_pressed_once(glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self._key_is_pressed_once(glfw.KEY_SPACE):
            self.gripper_state = 0.0 if self.gripper_state > 0.5 else 1.0  # toggle open/close
        if self._key_is_down(glfw.KEY_RIGHT_BRACKET):   # ] → close gradually
            self.gripper_state = min(1.0, self.gripper_state + 0.05)
        if self._key_is_down(glfw.KEY_LEFT_BRACKET):    # [ → open gradually
            self.gripper_state = max(0.0, self.gripper_state - 0.05)
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        self._teleop_debug_counter += 1
        if np.linalg.norm(action[:6]) > 1e-8 or self._teleop_debug_counter % 200 == 0:
            focused, active_keys = self._teleop_debug_status()
            print(f"[teleop-debug] focused={focused} keys={active_keys} action={np.round(action, 4)}")
        return action, False
    
    def _teleop_mocap(self):
        '''
        Mocap / IK Cartesian teleoperation.

        In the MuJoCo viewer: double-click the red sphere to select it, then
        drag to reposition the arm.  MuJoCo moves the mocap body natively;
        this method reads its position each step and feeds it to the IK solver
        via the eef_pose action (delta_p = mocap_pos - current_IK_target).

        Keyboard rotation still works:
            Q/E        : yaw left / right
            LEFT/RIGHT : roll
            UP/DOWN    : pitch
        SPACE: gripper open/close
        Z    : reset episode
        '''
        delta_p = np.zeros(3, dtype=np.float32)

        try:
            target_body_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'target')
            if target_body_id >= 0:
                mocap_id = self.env.model.body_mocapid[target_body_id]
                if mocap_id >= 0:
                    # Pre-assign pert.select every step so Ctrl+drag always
                    # controls this body — no double-click selection required.
                    viewer = getattr(self.env, 'viewer', None)
                    if viewer is not None and hasattr(viewer, 'pert'):
                        if viewer.pert.select != target_body_id:
                            viewer.pert.select = target_body_id
                            mujoco.mjv_initPerturb(
                                self.env.model, self.env.data,
                                viewer.scn, viewer.pert)

                    mocap_pos = self.env.data.mocap_pos[mocap_id].copy()
                    delta_p = (mocap_pos - self.p0).astype(np.float32)
                    # Clamp to avoid huge jumps on first frame
                    delta_p = np.clip(delta_p, -0.02, 0.02)
        except Exception as e:
            print(f'[mocap] {e}')

        # Keyboard rotation
        drot = np.eye(3)
        if self._key_is_down(glfw.KEY_Q):
            drot = rotation_matrix(angle=0.03, direction=[0, 0, 1])[:3, :3]
        if self._key_is_down(glfw.KEY_E):
            drot = rotation_matrix(angle=-0.03, direction=[0, 0, 1])[:3, :3]
        if self._key_is_down(glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.03, direction=[0, 1, 0])[:3, :3]
        if self._key_is_down(glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.03, direction=[0, 1, 0])[:3, :3]
        if self._key_is_down(glfw.KEY_UP):
            drot = rotation_matrix(angle=0.03, direction=[1, 0, 0])[:3, :3]
        if self._key_is_down(glfw.KEY_DOWN):
            drot = rotation_matrix(angle=-0.03, direction=[1, 0, 0])[:3, :3]

        if self._key_is_pressed_once(glfw.KEY_SPACE):
            self.gripper_state = 0.0 if self.gripper_state > 0.5 else 1.0  # toggle open/close
        if self._key_is_down(glfw.KEY_RIGHT_BRACKET):   # ] → close gradually
            self.gripper_state = min(1.0, self.gripper_state + 0.05)
        if self._key_is_down(glfw.KEY_LEFT_BRACKET):    # [ → open gradually
            self.gripper_state = max(0.0, self.gripper_state - 0.05)
        if self._key_is_pressed_once(glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True

        drpy = r2rpy(drot).astype(np.float32)
        action = np.concatenate([delta_p, drpy,
                                  np.array([self.gripper_state], dtype=np.float32)])
        return action, False

    def _teleop_joint_space(self):
        '''
        Joint-space teleoperation for SO101.

        Mouse (middle button drag):
            Drag left/right     : shoulder_pan
            Drag up/down        : shoulder_lift  (drag up = lift up)
            Ctrl + drag L/R     : elbow_flex
            Ctrl + drag up/down : wrist_flex

        Keyboard (unchanged):
            A/D        : shoulder_pan
            W/S        : shoulder_lift
            R/F        : elbow_flex
            UP/DOWN    : wrist_flex
            LEFT/RIGHT : wrist_roll
            SPACE      : gripper open/close
            Z          : reset
        '''
        dq = np.zeros(self.n_arm_joints, dtype=np.float32)
        joint_vel = 0.03  # Joint velocity magnitude per keyboard press

        # Mocap IK tracking: Ctrl + right-drag the red sphere to move the arm.
        # pert.select is pre-assigned in init_viewer so drag works immediately.
        if self._mocap_id >= 0:
            mocap_pos = self.env.data.mocap_pos[self._mocap_id].copy()
            mocap_quat = self.env.data.mocap_quat[self._mocap_id].copy()
            if self._last_mocap_pos is None:
                self._last_mocap_pos = mocap_pos.copy()
                self._last_mocap_quat = mocap_quat.copy()
            else:
                pos_changed = np.linalg.norm(mocap_pos - self._last_mocap_pos) > 5e-4
                quat_changed = np.linalg.norm(mocap_quat - self._last_mocap_quat) > 5e-4
                if pos_changed or quat_changed:
                    q_current = self.env.get_qpos_joints(joint_names=self.joint_names).copy()
                    R_trgt = quat2r(mocap_quat) if quat_changed else None
                    q_ik, _, _ = solve_ik(
                        env                = self.env,
                        joint_names_for_ik = self.joint_names,
                        body_name_trgt     = self.tcp_body_name,
                        q_init             = q_current,
                        p_trgt             = mocap_pos,
                        R_trgt             = R_trgt,
                        max_ik_tick        = 30,
                        ik_stepsize        = 0.5,
                        ik_eps             = 0.01,
                        ik_err_th          = 0.01,
                        verbose_warning    = False,
                        restore_state      = False,
                    )
                    dq += np.clip(q_ik - q_current, -joint_vel * 4, joint_vel * 4)
            self._last_mocap_pos = mocap_pos.copy()
            self._last_mocap_quat = mocap_quat.copy()

        # Middle-mouse-button drag: fine per-joint jog (unchanged)
        _mouse_jnt_scale = 0.0005
        _mdx, _mdy, _mctrl = self._get_mouse_delta()
        if abs(_mdx) > 0.2 or abs(_mdy) > 0.2:
            if _mctrl:
                dq[2] += _mdx * _mouse_jnt_scale
                dq[3] -= _mdy * _mouse_jnt_scale
            else:
                dq[0] += _mdx * _mouse_jnt_scale
                dq[1] -= _mdy * _mouse_jnt_scale

        # W/S: shoulder lift
        if self._key_is_down(glfw.KEY_W):
            dq[1] += joint_vel  # Shoulder lift up
        if self._key_is_down(glfw.KEY_S):
            dq[1] -= joint_vel  # Shoulder lift down

        # A/D: left-right (shoulder pan)
        if self._key_is_down(glfw.KEY_A):
            dq[0] -= joint_vel  # Shoulder pan left
        if self._key_is_down(glfw.KEY_D):
            dq[0] += joint_vel  # Shoulder pan right

        # UP/DOWN: up-down (wrist flex)
        if self._key_is_down(glfw.KEY_UP):
            dq[3] += joint_vel  # Wrist flex up
        if self._key_is_down(glfw.KEY_DOWN):
            dq[3] -= joint_vel  # Wrist flex down
        
        # RF: Control elbow
        if self._key_is_down(glfw.KEY_R):
            dq[2] += joint_vel  # Elbow flex up
        if self._key_is_down(glfw.KEY_F):
            dq[2] -= joint_vel  # Elbow flex down
        
        # LEFT/RIGHT: Control wrist roll
        if self._key_is_down(glfw.KEY_LEFT):
            dq[4] += joint_vel  # Wrist roll
        if self._key_is_down(glfw.KEY_RIGHT):
            dq[4] -= joint_vel  # Wrist roll
        
        # UP/DOWN reserved for future use or finer control
        
        # Reset and gripper
        if self._key_is_pressed_once(glfw.KEY_Z):
            return np.zeros(self.n_arm_joints + 1, dtype=np.float32), True
        if self._key_is_pressed_once(glfw.KEY_SPACE):
            self.gripper_state = 0.0 if self.gripper_state > 0.5 else 1.0  # toggle open/close
        if self._key_is_down(glfw.KEY_RIGHT_BRACKET):   # ] → close gradually
            self.gripper_state = min(1.0, self.gripper_state + 0.05)
        if self._key_is_down(glfw.KEY_LEFT_BRACKET):    # [ → open gradually
            self.gripper_state = max(0.0, self.gripper_state - 0.05)

        # Combine joint deltas with gripper state
        action = np.concatenate([dq, np.array([self.gripper_state], dtype=np.float32)])
        self._teleop_debug_counter += 1
        if np.linalg.norm(dq) > 1e-8 or self._teleop_debug_counter % 200 == 0:
            focused, active_keys = self._teleop_debug_status()
            print(f"[teleop-debug] focused={focused} keys={active_keys} dq={np.round(dq, 4)} gripper={self.gripper_state:.2f}")
        return action, False
    
    def get_delta_q(self):
        '''
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + normalised gripper position
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]  where gripper ∈ [0,1] (0=open, 1=closed)
        '''
        delta = self.compute_q - self.prev_q
        gripper_val = self._get_gripper_fraction()
        return np.concatenate([delta, [gripper_val]], dtype=np.float32)

    def _get_gripper_fraction(self):
        """
        Return the normalized task-space gripper command where 0=open, 1=closed.
        """
        gripper = self.env.get_qpos_joint(self.gripper_joint_name)
        raw_qpos = float(gripper[0])
        if self.gripper_continuous:
            return float(
                np.clip(
                    (self.gripper_open_rad - raw_qpos) / self.gripper_motion_span,
                    0.0,
                    1.0,
                )
            )
        return 1.0 if raw_qpos > 0.5 else 0.0

    def _is_gripper_open(self, threshold=0.2):
        """
        Return True when the normalized close fraction is below the open threshold.
        """
        return self._get_gripper_fraction() < threshold

    def check_success(self):
        '''
        Check if the object has been placed in/on the target.

        SO100/SO101 (bin task):
            The episode ends only when the block is truly inside the bin AND the
            gripper has been retracted above the bin opening.
            Conditions:
              1. Block center is within bin inner XY bounds (not just near bin).
              2. Block center Z is inside bin cavity bounds (resting on bin floor).
              3. Block was previously lifted off the table (prevents pushing/rolling in).
              4. Block is in contact with the bin (confirms physical placement).
              5. Gripper is open (not holding the block).
              6. TCP is above the bin wall tops (gripper retracted after release).
              7. Gripper is not in contact with the block.

        OMY (plate task):
            1. Object XY within 10 cm of plate centre.
            2. Object Z within 60 cm of plate Z.
            3. Gripper open (raw joint < 0.1 rad).
            4. TCP Z > success_height_threshold (arm retracted upward).
        '''
        p_obj = self.env.get_p_body(self.pick_body_name)
        p_tgt = self.env.get_p_body(self.place_body_name)

        if self.success_height_threshold is None:
            # SO100/SO101 bin geometry from obj_bin.xml:
            # - inner half-width ~= 0.054 - 0.006 = 0.048 m
            # - floor top offset   = 0.012 m
            # - wall top offset    = 0.076 m
            # Block is 25 mm cube (half-size 0.0125 m) in obj_blocks.xml.
            inner_half_extent = 0.048
            block_half_size = 0.0125
            floor_top = p_tgt[2] + 0.012
            wall_top = p_tgt[2] + 0.076

            dx = abs(p_obj[0] - p_tgt[0])
            dy = abs(p_obj[1] - p_tgt[1])
            xy_ok = (dx < (inner_half_extent - 0.003)) and (dy < (inner_half_extent - 0.003))

            # Block must be resting on the bin floor.  A 25 mm cube resting on
            # the floor has center height floor_top + block_half_size.
            expected_floor_contact_center_z = floor_top + block_half_size
            z_min = expected_floor_contact_center_z - 0.006
            z_max = expected_floor_contact_center_z + 0.010
            z_ok = (p_obj[2] > z_min) and (p_obj[2] < z_max)

            # Task intent is pick-and-place; pushing/rolling directly into the
            # bin should not terminate the episode.  Require a clear lift above
            # the settled reset height first.
            rest_h = self._target_rest_height if self._target_rest_height is not None else p_obj[2]
            if p_obj[2] > (rest_h + self._lift_height_delta):
                self._target_was_lifted = True
            if not self._target_was_lifted:
                return False

            # Contact gating: success requires actual block-bin contact and no
            # active block-gripper contact.
            contact_pairs = self.env.get_contact_body_names()

            def _has_contact(body_a, body_b):
                for c1, c2 in contact_pairs:
                    if (c1 == body_a and c2 == body_b) or (c1 == body_b and c2 == body_a):
                        return True
                return False

            block_bin_contact = _has_contact(self.pick_body_name, self.place_body_name)
            block_gripper_contact = _has_contact(self.pick_body_name, 'gripper') or _has_contact(
                self.pick_body_name, 'moving_jaw_so101_v1'
            )

            gripper_open = self._is_gripper_open(threshold=0.20)

            # The gripper jaw tip must be retracted above the bin wall tops.
            # IMPORTANT: tcp_body_name ('gripper') is the palm, ~98 mm above the
            # actual jaw tips.  Using the palm position means the check passes even
            # when the jaws are inside the bin cavity, which is the root cause of
            # false-positive success when merely touching the block.
            # Use the 'gripperframe' site (jaw tip) for an accurate measurement.
            _site_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
            if _site_id >= 0:
                tip_z = float(self.env.data.site_xpos[_site_id][2])
            else:
                # Fallback: subtract the nominal jaw-tip offset from the palm body.
                tip_z = float(self.env.get_p_body(self.tcp_body_name)[2]) - 0.098
            tcp_above = tip_z > wall_top + 0.015  # jaw tip ≥ 15 mm above bin wall tops

            success = bool(xy_ok and z_ok and gripper_open and block_bin_contact
                        and tcp_above and not block_gripper_contact)
            
            # Debug logging
            if success or (xy_ok and z_ok):  # Log when close to success
                print(f"[DEBUG check_success] xy_ok={xy_ok} z_ok={z_ok} lifted={self._target_was_lifted} "
                      f"gripper_open={gripper_open} block_bin={block_bin_contact} "
                      f"tcp_above={tcp_above} no_grip_contact={not block_gripper_contact} => {success}")
            
            return success
        else:
            # OMY: original plate-placement logic.
            xy_ok        = np.linalg.norm(p_obj[:2] - p_tgt[:2]) < 0.1
            z_ok         = np.linalg.norm(p_obj[2]  - p_tgt[2])  < 0.6
            gripper_open = self._is_gripper_open(threshold=0.10)
            if not (xy_ok and z_ok and gripper_open):
                return False
            tcp_z = self.env.get_p_body(self.tcp_body_name)[2]
            return tcp_z > self.success_height_threshold
    
    def get_obj_pose(self):
        '''
        returns: 
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        p_pick = self.env.get_p_body(self.pick_body_name)
        p_place = self.env.get_p_body(self.place_body_name)
        return p_pick, p_place
    
    def set_obj_pose(self, p_pick, p_place):
        '''
        Set the object poses
        args:
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name=self.pick_body_name,p=p_pick)
        self.env.set_R_base_body(body_name=self.pick_body_name,R=np.eye(3,3))
        # Only attempt to move the plate/target body if it actually has a free joint.
        # In the SO100/SO101 bin task, the bin is a fixed body with no free joint,
        # so calling set_p_base_body on it leads to qpos shape mismatches.
        try:
            body = self.env.model.body(self.place_body_name)
            n_joint = body.jntnum
            if n_joint > 0:
                first_joint = self.env.model.joint(body.jntadr[0])
                if first_joint.type[0] == mujoco.mjtJoint.mjJNT_FREE:
                    self.env.set_p_base_body(body_name=self.place_body_name, p=p_place)
                    self.env.set_R_base_body(body_name=self.place_body_name, R=np.eye(3, 3))
        except Exception:
            # If anything goes wrong (e.g. body not found), fall back to leaving the plate fixed.
            pass
        self.step_env()


    def get_ee_pose(self):
        '''
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name=self.tcp_body_name)
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)