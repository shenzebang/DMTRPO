from gym.envs.registration import register

# Biased Mujoco

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
    'Hopper_FLBias-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.hopper:HopperEnv_Bias'},
    max_episode_steps=1000
)

register(
    'Walker2d_FLBias-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.walker2d:Walker2dEnv_Bias'},
    max_episode_steps=1000
)

register(
    'Humanoid_FLBias-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.humanoid:HumanoidEnv_Bias'},
    max_episode_steps=1000
)


# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)

register(
    '2DNavigation-v1',
    entry_point='envs.navigation:Navigation2DEnv_FL',
    max_episode_steps=100
)

# Quantized Mujoco
register(
    id='HopperQuantized-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.hopper:HopperEnvQuantized'},
    max_episode_steps=1000
)
register(
    id='HalfCheetahQuantized-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.half_cheetah:HalfCheetahEnvQuantized'},
    max_episode_steps=1000
)
register(
    id='Walker2dQuantized-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.walker2d:Walker2dEnvQuantized'},
    max_episode_steps=1000
)
register(
    id='SwimmerQuantized-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.swimmer:SwimmerEnvQuantized'},
    max_episode_steps=1000
)
register(
    id='ReacherQuantized-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.reacher:ReacherEnvQuantized'},
    max_episode_steps=50
)
register(
    id='AntQuantized-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.ant:AntEnvQuantized'},
    max_episode_steps=1000
)
register(
    id='HumanoidQuantized-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.humanoid:HumanoidEnvQuantized'},
    max_episode_steps=1000
)