# RGB 3-block tabletop task for SO-ARM101 (teleop-oriented)
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from isaac_so_arm101.tasks.lift.joint_pos_env_cfg import SoArm101LiftCubeEnvCfg


def _euler_deg_to_quat(rx, ry, rz):
    """오일러각(도, XYZ) → 쿼터니언 (w, x, y, z)"""
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    return (cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz)


# ============ 실측값만 여기서 수정 (단위: m, 각도: deg) ============
#색상
RED_RGB   = (0.9, 0.05, 0.05)
GREEN_RGB = (0.05, 0.8, 0.05)
BLUE_RGB  = (0.05, 0.1, 0.9)
BLACK_RGB = (0.01, 0.01, 0.01)

# 책상
TABLE_SIZE   = (0.70, 0.40, 0.015)
TABLE_CENTER = (0.33, 0.00, -0.015)
TABLE_RGB    = (0.95, 0.95, 0.95)

# 블록
BLOCK_SIZE = 0.035
BLOCK_MASS = 0.02
BLOCK_XY  = {"red": (0.30, -0.08), "green": (0.30, 0.00), "blue": (0.30, 0.08)}
BLOCK_RGB = {"red": RED_RGB, "green": GREEN_RGB, "blue": BLUE_RGB}

# 카메라 공통
CAM_W, CAM_H = 640, 480
CAM_PLATE  = (0.035, 0.035, 0.005)
CAM_LENS_R = 0.0075
CAM_LENS_H = 0.018
CAM_NEAR   = 0.05                          
# CAM_FOCAL    = 8.5      # Innomaker U20CAM-720P: 수평 FOV 102도
CAM_FOCAL    = 20.0
CAM_APERTURE = 20.955   # 센서 가로(mm)

# 그리퍼캠
# GRIPCAM_OFFSET_POS = (0.0, -0.05, 0.00)
GRIPCAM_OFFSET_POS = (0.0, -0.07, 0.0035)  
GRIPCAM_EULER      = (-25.0, 180.0, 0.0)  

# 탑뷰 거치대/카메라
TOPCAM_STAND_XY   = (0.43, 0.035)           
TOPCAM_STAND_SIZE = (0.025, 0.035, 0.39)   
TOPCAM_POS        = (0.40, 0.038, 0.40)     
# TOPCAM_EULER      = (0.0, -160.0, -90.0)      
TOPCAM_EULER      = (-150.0, 4.5, -270.0)
# ================================================================

GRIPCAM_ROT = _euler_deg_to_quat(*GRIPCAM_EULER)
TOPCAM_ROT  = _euler_deg_to_quat(*TOPCAM_EULER)

_BLOCK_Z = TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0 + BLOCK_SIZE / 2.0
_LENS_LOCAL_Z = CAM_PLATE[2] / 2 + CAM_LENS_H / 2   # 판 기준 렌즈 로컬 오프셋

_RIGID = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)


def _cube(suffix, xy, rgb):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + suffix,
        init_state=RigidObjectCfg.InitialStateCfg(pos=[xy[0], xy[1], _BLOCK_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
            rigid_props=_RIGID,
            mass_props=sim_utils.MassPropertiesCfg(mass=BLOCK_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
        ),
    )


def _cam_plate(prim_path, pos, rot):
    """카메라 몸체(판) — 부모. 여기에 pos/rot 적용."""
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=CAM_PLATE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BLACK_RGB),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def _cam_lens(parent_path):
    """렌즈 — 판의 자식. 판 로컬 z앞에 고정되어 판과 함께 회전."""
    return AssetBaseCfg(
        prim_path=parent_path + "/lens",
        spawn=sim_utils.CylinderCfg(
            radius=CAM_LENS_R, height=CAM_LENS_H,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BLACK_RGB),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, _LENS_LOCAL_Z)),
    )


@configclass
class SoArm101RGBBlocksEnvCfg(SoArm101LiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 책상
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=list(TABLE_CENTER), rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=TABLE_SIZE,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_RGB),
            ),
        )

        # RGB 블록 3개
        self.scene.object = _cube("Object", BLOCK_XY["red"], BLOCK_RGB["red"])
        self.scene.block_green = _cube("BlockGreen", BLOCK_XY["green"], BLOCK_RGB["green"])
        self.scene.block_blue = _cube("BlockBlue", BLOCK_XY["blue"], BLOCK_RGB["blue"])

        # 탑뷰 카메라 + 모듈(판+렌즈 묶음) + 거치대
        self.scene.top_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/top_cam",
            update_period=0.0, height=CAM_H, width=CAM_W, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=CAM_FOCAL, horizontal_aperture=CAM_APERTURE, clipping_range=(CAM_NEAR, 10.0)),
            offset=CameraCfg.OffsetCfg(pos=TOPCAM_POS, rot=TOPCAM_ROT, convention="ros"),
        )
        self.scene.top_cam_body = _cam_plate("{ENV_REGEX_NS}/top_cam_body", TOPCAM_POS, TOPCAM_ROT)
        self.scene.top_cam_lens = _cam_lens("{ENV_REGEX_NS}/top_cam_body")
        self.scene.top_cam_stand = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/top_cam_stand",
            spawn=sim_utils.CuboidCfg(
                size=TOPCAM_STAND_SIZE,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BLACK_RGB),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(TOPCAM_STAND_XY[0], TOPCAM_STAND_XY[1], TOPCAM_STAND_SIZE[2] / 2.0),
            ),
        )

        # 그리퍼 카메라 + 모듈(판+렌즈 묶음)
        self.scene.gripper_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_link/grip_cam",
            update_period=0.0, height=CAM_H, width=CAM_W, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=CAM_FOCAL, horizontal_aperture=CAM_APERTURE, clipping_range=(CAM_NEAR, 10.0)),
            offset=CameraCfg.OffsetCfg(pos=GRIPCAM_OFFSET_POS, rot=GRIPCAM_ROT, convention="ros"),
        )
        self.scene.gripper_cam_body = _cam_plate(
            "{ENV_REGEX_NS}/Robot/gripper_link/grip_cam_body", GRIPCAM_OFFSET_POS, GRIPCAM_ROT)
        self.scene.gripper_cam_lens = _cam_lens(
            "{ENV_REGEX_NS}/Robot/gripper_link/grip_cam_body")


@configclass
class SoArm101RGBBlocksEnvCfg_PLAY(SoArm101RGBBlocksEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False