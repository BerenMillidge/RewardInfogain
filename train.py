# pylint: disable=not-callable
# pylint: disable=no-member

import sys
import time
import pathlib
import argparse
import subprocess
from datetime import datetime

import numpy as np
import torch
import baselines
from baselines.envs import TorchEnv, NoisyEnv, const
from gym.wrappers.monitoring.video_recorder import VideoRecorder

sys.path.append(str(pathlib.Path(__file__).parent.parent))

#from pmbrl.envs import GymEnv
#from pmbrl.envs.envs.ant import rate_buffer
from pmbrl.training import Normalizer, Buffer, Trainer
from pmbrl.models import EnsembleModel, RewardModel
from pmbrl.control import Planner, Agent
from pmbrl import utils

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    logger = utils.Logger(args.logdir, args.seed)
    logger.log("\n=== Loading experiment [device: {}] ===\n".format(DEVICE))
    logger.log(args)

    # create correct save path directory if it doesn't already exist
    if args.save_path != "":
        subprocess.call(["mkdir","-p",str(args.save_path)])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    env = TorchEnv(
        args.env_name,
        args.max_episode_len,
        action_repeat=args.action_repeat,
        seed=args.seed,
    )
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    normalizer = Normalizer()
    buffer = Buffer(
        state_size, action_size, args.ensemble_size, normalizer, device=DEVICE
    )

    ensemble = EnsembleModel(
        state_size + action_size,
        state_size,
        args.hidden_size,
        args.ensemble_size,
        normalizer,
        device=DEVICE,
    )
    reward_model = RewardModel(
        state_size + action_size, args.hidden_size, device=DEVICE
    )
    trainer = Trainer(
        ensemble,
        reward_model,
        buffer,
        n_train_epochs=args.n_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        grad_clip_norm=args.grad_clip_norm,
        logger=logger,
    )

    planner = Planner(
        ensemble,
        reward_model,
        action_size,
        args.ensemble_size,
        plan_horizon=args.plan_horizon,
        optimisation_iters=args.optimisation_iters,
        n_candidates=args.n_candidates,
        top_candidates=args.top_candidates,
        use_reward=args.use_reward,
        use_exploration=args.use_exploration,
        use_mean=args.use_mean,
        expl_scale=args.expl_scale,
        reward_scale=args.reward_scale,
        strategy=args.strategy,
        device=DEVICE,
    )
    agent = Agent(env, planner, logger=logger)

    agent.get_seed_episodes(buffer, args.n_seed_episodes)
    msg = "\nCollected seeds: [{} episodes | {} frames]"
    logger.log(msg.format(args.n_seed_episodes, buffer.total_steps))

    for episode in range(1, args.n_episodes):
        logger.log("\n=== Episode {} ===".format(episode))
        subprocess.call(["echo", "Beginning episode: " + str(episode)])

        start_time = time.time()

        msg = "Training on [{}/{}] data points"
        logger.log(
            msg.format(buffer.total_steps, buffer.total_steps * args.action_repeat)
        )
        trainer.reset_models()
        ensemble_loss, reward_loss = trainer.train()
        logger.log_losses(ensemble_loss, reward_loss)

        recorder = None
        if args.record_every is not None and args.record_every % episode == 0:
            filename = logger.get_video_path(episode)
            recorder = VideoRecorder(env.unwrapped, path=filename)
            logger.log("Setup recoder @ {}".format(filename))

        logger.log("\n=== Collecting data [{}] ===".format(episode))
        reward, steps, stats = agent.run_episode(
            buffer, action_noise=args.action_noise, recorder=recorder
        )
        logger.log_episode(reward, steps)
        logger.log_stats(stats)

        if args.coverage:
            coverage = rate_buffer(buffer=buffer)
            logger.log_coverage(coverage)

        logger.log_time(time.time() - start_time)
        logger.save()
        if episode % args.save_every == 0:
            subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logger.path),str(args.save_path)])
            logger.log("Rsynced files from: " + str(logger.path) + "/ " + " to" + str(args.save_path))
            now = datetime.now()
            current_time = str(now.strftime("%H:%M:%S"))
            subprocess.call(['echo', "TIME_OF_SAVE: " + str(current_time)])


if __name__ == "__main__":

    def boolcheck(x):
        return str(x).lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()
    #Environment envs
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", type=str, default="information")
    parser.add_argument("--save_path", type=str, default="/home/s1686853/default_save")

    # config additional envs
    parser.add_argument("--max_episode_len", type=int)
    parser.add_argument("--action_repeat", type=int)
    parser.add_argument("--env_std", type=float, default=0.00)
    parser.add_argument("--action_noise", type=float, default=0.0)
    parser.add_argument("--ensemble_size", type=int)
    parser.add_argument("--buffer_size", type=int, default=10 ** 6)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--plan_horizon", type=int)
    parser.add_argument("--n_candidates", type=int)
    parser.add_argument("--optimisation_iters", type=int)
    parser.add_argument("--top_candidates", type=int)
    parser.add_argument("--n_seed_episodes", type=int, default=5)
    parser.add_argument("--n_train_epochs", type=int)
    parser.add_argument("--n_episodes", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_clip_norm", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--use_reward", type=boolcheck, default=True)
    parser.add_argument("--use_exploration", type=boolcheck, default=False)
    parser.add_argument("--render", type=boolcheck, default=False)
    parser.add_argument("--expl_scale", type=float)
    parser.add_argument("--planner", type=str, default="CEM")
    parser.add_argument("--use_ensemble_reward_model", type=boolcheck, default=False)
    parser.add_argument("--use_reward_info_gain", type=boolcheck, default=False)
    parser.add_argument("--collect_trajectories",type=boolcheck, default=False)
    parser.add_argument("--trajectory_savedir", type=str,default="trajectories/")
    parser.add_argument("--use_epsilon_greedy", type=boolcheck, default=False)
    parser.add_argument("--epsilon_greedy_value", type=float, default=0.0)
    args = parser.parse_args()
    print("args: ", args)
    print(type(args))
    print(args.logdir)
    print(args.learning_rate)
    config = utils.get_config(args)
    print("Config!", config)

    main(config)
