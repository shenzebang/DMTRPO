from gym.envs.registration import register

register(
    'HalfCheetah_FLVel-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.half_cheetah:HalfCheetahVelEnv_FL'},
    max_episode_steps=1000
)

register(
    'HalfCheetah_FLBias-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.half_cheetah:HalfCheetahEnv_Bias'},
    max_episode_steps=1000
)

register(
    'HopperBias-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.hopper:HopperEnv_Bias'},
    max_episode_steps=1000
)

register(
    'HopperBias10-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.hopper:HopperEnv_10Bias'},
    max_episode_steps=1000
)

register(
    'HopperBias20-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.hopper:HopperEnv_20Bias'},
    max_episode_steps=1000
)

register(
    'HopperBias50-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.hopper:HopperEnv_50Bias'},
    max_episode_steps=1000
)

register(
    'Walker2dBias-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.walker2d:Walker2dEnv_Bias'},
    max_episode_steps=1000
)

register(
    'Walker2dBias10-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.walker2d:Walker2dEnv_10Bias'},
    max_episode_steps=1000
)

register(
    'Walker2dBias20-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.walker2d:Walker2dEnv_20Bias'},
    max_episode_steps=1000
)

register(
    'Walker2dBias50-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.walker2d:Walker2dEnv_50Bias'},
    max_episode_steps=1000
)

register(
    'HumanoidBias-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.humanoid:HumanoidEnv_Bias'},
    max_episode_steps=1000
)

register(
    'HumanoidBias10-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.humanoid:HumanoidEnv_10Bias'},
    max_episode_steps=1000
)

register(
    'HumanoidBias20-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.humanoid:HumanoidEnv_20Bias'},
    max_episode_steps=1000
)

register(
    'HumanoidBias50-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.humanoid:HumanoidEnv_50Bias'},
    max_episode_steps=1000
)

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)

