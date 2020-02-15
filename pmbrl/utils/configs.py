import pprint

MOUNTAIN_CAR_CONFIG = "mountain_car"
CUP_CATCH_CONFIG = "cup_catch"
HALF_CHEETAH_RUN_CONFIG = "half_cheetah_run"
HALF_CHEETAH_FLIP_CONFIG = "half_cheetah_flip"
REACHER_CONFIG = "reacher"
AMT_MAZE = "ant_maze"
DEBUG_CONFIG = "debug"

BULLET_HALF_CHEETAH_CONFIG = "bullet_halfcheetah"
BULLET_ANT_CONFIG = "bullet_ant"
BULLET_INVERTED_PENDULUM_CONFIG = "bullet_cartpole"
LUNAR_LANDER_CONFIG = "lunar_lander"

ROBOSCHOOL_INVERTED_PENDULUM = "roboschool_inverted_pendulum"
ROBOSCHOOL_HALF_CHEETAH = "roboschool_half_cheetah"
ROBOSCHOOL_ANT  = "roboschool_ant"
ROBOSCHOOL_HUMANOID = "roboschool_humanoid"
ROBOSCHOOL_REACHER = "roboschool_reacher"
ROBOSCHOOL_HOPPER = "roboschool_hopper"


MUJOCO_INVERTED_PENDULUM ="mujoco_inverted_pendulum"
MUJOCO_HALF_CHEETAH = "mujuco_half_cheetah"
MUJOCO_ANT = "mujoco_ant"
MUJOCO_REACHER = "mujoco_reacher"
MUJOCO_HOPPER = "mujoco_hopper"
MUJOCO_HUMANOID = "mujoco_humanoid"


def get_config(args):
    if args.config_name == MOUNTAIN_CAR_CONFIG:
        config = MountainCarConfig()
    elif args.config_name == CUP_CATCH_CONFIG:
        config = CupCatchConfig()
    elif args.config_name == HALF_CHEETAH_RUN_CONFIG:
        config = HalfCheetahRunConfig()
    elif args.config_name == HALF_CHEETAH_FLIP_CONFIG:
        config = HalfCheetahFlipConfig()
    elif args.config_name == REACHER_CONFIG:
        config = ReacherConfig()
    elif args.config_name == AMT_MAZE:
        config = AntMazeConfig()
    elif args.config_name == DEBUG_CONFIG:
        config = DebugConfig()
    elif args.config_name == BULLET_HALF_CHEETAH_CONFIG:
        config = BulletHalfCheetahConfig()
    elif args.config_name == BULLET_ANT_CONFIG:
        config = BulletAntConfig()
    elif args.config_name == BULLET_INVERTED_PENDULUM_CONFIG:
        config = BulletInvertedPendulumSwingupConfig()
    elif args.config_name == LUNAR_LANDER_CONFIG:
        config = LunarLanderContinuousConfig()
    elif args.config_name == ROBOSCHOOL_INVERTED_PENDULUM:
        config = RoboschoolInvertedPendulumConfig()
    elif args.config_name == ROBOSCHOOL_HALF_CHEETAH:
        config = RoboschoolHalfCheetahConfig()
    elif args.config_name == ROBOSCHOOL_ANT:
        config = RoboschoolAntConfig()
    elif args.config_name == ROBOSCHOOL_HUMANOID:
        config = RoboschoolHumanoidConfig()
    elif args.config_name == ROBOSCHOOL_REACHER:
        config = RoboschoolReacherConfig()
    elif args.config_name == ROBOSCHOOL_HOPPER:
        config = RoboschoolHopperConfig()


    elif args.config_name == MUJOCO_INVERTED_PENDULUM:
        config = MujocoInvertedPendulumConfig()
    elif args.config_name == MUJOCO_HALF_CHEETAH:
        config = MujocoHalfCheetahConfig()
    elif args.config_name == MUJOCO_ANT:
        config = MujocoAntConfig()
    elif args.config_name == MUJOCO_REACHER:
        config = MujocoReacherConfig()
    elif args.config_name == MUJOCO_HOPPER:
        config = MujocoHopperConfig()
    elif args.config_name == MUJOCO_HUMANOID:
        config = MujocoHumanoidConfig()
    else:
        raise ValueError("`{}` is not a valid config ID".format(args.config_name))

    config = apply_parseargs(config, args)
    return config

def apply_parseargs(config, args):
    arg_dict = vars(args)
    for key in arg_dict.keys():
        val = arg_dict[key]
        if val is not None:
            setattr(config, key, val)
    return config


class Config(object):
    def __init__(self):
        self.logdir = "log"
        self.seed = 0
        self.n_episodes = 50
        self.n_seed_episodes = 5
        self.record_every = None
        self.coverage = False

        self.env_name = None
        self.max_episode_len = 500
        self.action_repeat = 1
        self.action_noise = None

        self.ensemble_size = 10
        self.hidden_size = 200

        self.n_train_epochs = 100
        self.batch_size = 50
        self.learning_rate = 1e-3
        self.epsilon = 1e-8
        self.grad_clip_norm = 1000

        self.plan_horizon = 30
        self.optimisation_iters = 5
        self.n_candidates = 500
        self.top_candidates = 50

        self.expl_strategy = "information"
        self.use_reward = True
        self.use_exploration = True
        self.use_mean = False

        self.expl_scale = 1.0
        self.reward_scale = 1.0

    def set_logdir(self, logdir):
        self.logdir = logdir

    def set_seed(self, seed):
        self.seed = seed

    def set_strategy(self, strategy):
        self.strategy = strategy

    def __repr__(self):
        return pprint.pformat(vars(self))


class DebugConfig(Config):
    def __init__(self):
        super().__init__()
        self.env_name = "Pendulum-v0"
        self.n_episodes = 5
        self.max_episode_len = 100
        self.hidden_size = 64
        self.plan_horizon = 5


class MountainCarConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "mountain_car"
        self.env_name = "SparseMountainCar"
        self.max_episode_len = 500
        self.n_train_epochs = 100
        self.n_seed_episodes = 1
        self.expl_scale = 1.
        self.n_episodes = 30
        self.ensemble_size = 25
        self.record_every = None
        self.n_episodes = 50


class CupCatchConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "catch"
        self.env_name = "DeepMindCatch"
        self.max_episode_len = 1000
        self.action_repeat = 4
        self.plan_horizon = 12
        self.expl_scale = 0.1
        self.record_every = None
        self.n_episodes = 50


class HalfCheetahRunConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "half_cheetah_run"
        self.env_name = "HalfCheetahRun"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class HalfCheetahFlipConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "half_cheetah_flip"
        self.env_name = "HalfCheetahFlip"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 30
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1


class BulletHalfCheetahConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "half_cheetah"
        self.env_name = "HalfCheetahBulletEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class BulletAntConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "ant_logs"
        self.env_name = "AntBulletEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 30
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1


class BulletInvertedPendulumSwingupConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "cartpole_logs"
        self.env_name = "InvertedPendulumSwingupBulletEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 30
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class LunarLanderContinuousConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "lunar_lander_logs"
        self.env_name = "LunarLanderContinuous-v2"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 3

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 30
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class RoboschoolInvertedPendulumConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "roboschool_inverted_pendulum"
        self.env_name = "InvertedPendulumPyBulletEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class RoboschoolAntConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "roboschool_ant"
        self.env_name = "AntPyBulletEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class RoboschoolHalfCheetahConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "roboschool_half_cheetah"
        self.env_name = "HalfCheetahPyBulletEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class RoboschoolHumanoidConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "roboschool_humanoid"
        self.env_name = "HumanoidPyBulletEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class RoboschoolReacherConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "roboschool_reacher"
        self.env_name = "ReacherPyBulletEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class RoboschoolHopperConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "roboschool_hopper"
        self.env_name = "HopperPyBulletEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

###########
#MUJOCO CONFIGS
############

class MujocoInvertedPendulumConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "mujoco_inverted_pendulum"
        self.env_name = "InvertedPendulumMuJoCoEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class MujocoAntConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "mujoco_ant"
        self.env_name = "AntMuJoCoEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class MujocoHalfCheetahConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "mujoco_half_cheetah"
        self.env_name = "HalfCheetahMuJoCoEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class MujocoHumanoidConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "mujoco_humanoid"
        self.env_name = "HumanoidMuJoCoEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class MujocoReacherConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "mujoco_reacher"
        self.env_name = "HopperMuJoCoEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class MujocoHopperConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "mujoco_hopper"
        self.env_name = "HopperMuJoCoEnv-v0"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 100
        self.action_repeat = 2

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 15
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_mean = True
        self.expl_scale = 0.1

class AntMazeConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "ant_maze"
        self.env_name = "AntMaze"
        self.n_episodes = 50
        self.n_seed_episodes = 5
        self.max_episode_len = 300
        self.action_repeat = 4
        self.coverage = True

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 200
        self.batch_size = 50

        self.plan_horizon = 30
        self.optimisation_iters = 7
        self.n_candidates = 700
        self.top_candidates = 70

        self.use_exploration = True
        self.use_reward = False
        self.use_mean = True
        self.expl_scale = 1.


class ReacherConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "reacher"
        self.env_name = "SparseReacher"
        self.n_episodes = 100
        self.n_seed_episodes = 5
        self.max_episode_len = 1000
        self.action_repeat = 4

        self.ensemble_size = 15
        self.hidden_size = 400

        self.n_train_epochs = 100
        self.batch_size = 50

        self.plan_horizon = 30
        self.optimisation_iters = 5
        self.n_candidates = 500
        self.top_candidates = 50

        self.use_exploration = True
        self.use_reward = True
        self.use_mean = True
        self.expl_scale = 0.1




""""
def config_parse_args(config, action_repeat, ensemble_size, plan_horizon,n_candidates, optimisation_iters):
    if action_repeat != -1:
        config.action_repeat = action_repeat
    if ensemble_size != -1:
        config.ensemble_size = ensemble_size
    if plan_horizon != -1:
        config.plan_horizon = plan_horizon
    if n_candidates != -1:
        config.n_candidates = n_candidates
    if optimisation_iters != -1:
        config.optimisation_iters = optimisation_iters


    return config
"""
