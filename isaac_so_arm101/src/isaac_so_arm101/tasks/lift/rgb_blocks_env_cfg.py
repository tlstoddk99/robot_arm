# RGB 3-block tabletop task for SO-ARM101 (teleop-oriented, visual scene)
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaac_so_arm101.tasks.lift.joint_pos_env_cfg import SoArm101LiftCubeEnvCfg

# 공통 물리 속성
_RIGID = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

# 색 있는 큐브를 스폰하는 헬퍼 (USD 대신 절차적 큐브 + 색 머티리얼)
def _colored_cube(prim_suffix, pos, rgb):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_suffix,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.025, 0.025, 0.025),  # 2.5cm 큐브 (실측값으로 나중에 조정)
            rigid_props=_RIGID,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
        ),
    )


@configclass
class SoArm101RGBBlocksEnvCfg(SoArm101LiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- 메인 object(빨강)를 색 있는 큐브로 교체 (reward가 이걸 참조하므로 유지) ---
        self.scene.object = _colored_cube("Object", [0.2, -0.06, 0.015], (0.9, 0.05, 0.05))

        # --- 초록/파랑 블록 추가 (reward와 무관, 시각/조작 대상) ---
        self.scene.block_green = _colored_cube("BlockGreen", [0.2, 0.0, 0.015], (0.05, 0.8, 0.05))
        self.scene.block_blue = _colored_cube("BlockBlue", [0.2, 0.06, 0.015], (0.05, 0.1, 0.9))

        # --- 테이블을 흰색으로 override ---
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
            spawn=sim_utils.CuboidCfg(
                size=(0.8, 0.6, 0.02),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
        )


@configclass
class SoArm101RGBBlocksEnvCfg_PLAY(SoArm101RGBBlocksEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
