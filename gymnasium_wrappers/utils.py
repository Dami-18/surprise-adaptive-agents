import gymnasium as gym
# import gym as old_gym
import csv_logger
import logging
import matplotlib.pyplot as plt
import minatar
import torch
import numpy as np
import imageio
from pathlib import Path
from gymnasium_wrappers.base_surprise import BaseSurpriseWrapper
from gymnasium_wrappers.base_sadapt import BaseSurpriseAdaptWrapper
from gymnasium_wrappers.base_surprise_adapt_bandit import BaseSurpriseAdaptBanditWrapper
# from gymnasium_wrappers.gym_to_gymnasium import GymToGymnasium
from gymnasium_wrappers.obs_resize import ResizeObservationWrapper
from gymnasium_wrappers.obs_history import ObsHistoryWrapper
from gymnasium_wrappers.rendering_wrapper import RenderObservationWrapper
from surprise.buffers.buffers import GaussianBufferIncremental, BernoulliBuffer, MultinoulliBuffer
from griddly import GymWrapperFactory, gd
import os
import ast

from IPython import embed
from gymnasium.envs.registration import register as gym_register

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )

def make_env(args):
    def thunk():
        theta_size = None
        grayscale = None
        threshold = 300
        ############ Create environment ############            
        if "tetris" in args.env_id:
            from surprise.envs.tetris.tetris import TetrisEnv
            env = TetrisEnv()
            max_steps = 500
            obs_size = (1,env.observation_space.shape[0])    
        elif "MinAtar" in args.env_id:
            env = gym.make(args.env_id+"-v1", render_mode='rgb_array', max_episode_steps=500)
            from gymnasium_wrappers.wrappers import ImageTranspose
            env = ImageTranspose(env)
            max_steps = 500
            o_, _ = env.reset()
            obs_size = o_.shape

        # elif "griddly" in args.env_id:
        #     # for instance griddly-MazeEnv
        #     register_griddly_envs()
        #     griddly_env_name = args.env_id.split('-')[-1]
        #     if "MazeEnv2" in griddly_env_name:
        #         max_steps = 100
        #     elif griddly_env_name in ["MazeEnvLarge", "MazeEnvLarge2"]:
        #         max_steps = 250
        #     else:
        #         max_steps = 500
        #     env = old_gym.make(f"GDY-{griddly_env_name}-v0", player_observer_type=gd.ObserverType.VECTOR, global_observer_type=gd.ObserverType.VECTOR)
        #     o_ = env.reset()
        #     obs_size = o_.shape
            
        #     if "MazeEnv" in griddly_env_name:
        #         from surprise.envs.maze.maze_env import MazeEnv
        #         env = MazeEnv(env)
        #     elif "ButterfliesEnv" in griddly_env_name:
        #         from surprise.envs.maze.butterflies_latest import ButterfliesEnv
        #         env = ButterfliesEnv(env)
        #         # discard spiders and cocoons and player
        #         obs_size = (obs_size[0] - 3, obs_size[1], obs_size[2])
        #     else:
        #         raise ValueError(f"Unknown griddly env {griddly_env_name}")
            
        #     env = GymToGymnasium(env, render_mode="rgb_array", max_steps=max_steps)

        elif "Atari" in args.env_id:
            atari_env_name = args.env_id.split('-')[-1]
            max_steps = 108_000
            env = gym.make(f"{atari_env_name}NoFrameskip-v4", render_mode='rgb_array', max_episode_steps=max_steps)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (64, 64))
            env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
            print(env.observation_space.sample().shape)
            grayscale = True
            channel_dim = 1 if grayscale else 3
            env = ObsHistoryWrapper(env, history_length=4, stack_channels=True, channel_dim=2)
            theta_size =  ast.literal_eval(args.theta_size)
            theta_size = (theta_size[0], theta_size[1], channel_dim) if grayscale else (theta_size[0], theta_size[1], channel_dim)
            obs_size = theta_size
            threshold = 80_000
            print("Atari theta size")
            print(obs_size)

        else:
            print(f"Making {args.env_id}")
            env = gym.make(args.env_id, render_mode='rgb_array', max_episode_steps = 500)
            max_steps = 500
            obs_size = env.observation_space.shape
        
        ############ Create buffer ############
        # if only 1 channel of info like tetris, we need to reshape the obs_size
        # e.g. for griddly envs, this will be (num_channels, channel_dim)

        if args.buffer_type == "gaussian":
            buffer = GaussianBufferIncremental(obs_size)
        elif args.buffer_type == "bernoulli":
            buffer = BernoulliBuffer(obs_size)
        elif args.buffer_type == "multinoulli":
            buffer = MultinoulliBuffer(obs_size)
        else:
            raise ValueError(f"Unknown buffer type {args.buffer_type}")
        
        if args.model == "smax":
            env = BaseSurpriseWrapper(
                env,
                buffer,
                add_true_rew=args.add_true_rew,
                minimize=False,
                int_rew_scale=1.0,
                max_steps=max_steps,
                theta_size = theta_size,
                grayscale = grayscale,
                soft_reset=args.soft_reset,
                death_cost = args.death_cost,
                exp_rew = args.exp_rew,
                threshold=threshold,
                add_random_obs=args.add_random_obs
            )
        
        elif args.model == "smin":
            env = BaseSurpriseWrapper(
                env,
                buffer,
                add_true_rew=args.add_true_rew,
                minimize=True,
                int_rew_scale=1.0,
                max_steps=max_steps,
                theta_size = theta_size,
                grayscale = grayscale,
                soft_reset=args.soft_reset,
                death_cost = args.death_cost,
                exp_rew = args.exp_rew,
                threshold=threshold,
                add_random_obs=args.add_random_obs
            )
        
        elif args.model == "sadapt":
            env = BaseSurpriseAdaptWrapper(
                env,
                buffer,
                surprise_window_len=args.surprise_window_len,
                surprise_change_threshold=args.surprise_change_threshold,
                momentum=True,
                add_true_rew=args.add_true_rew,
                int_rew_scale=1.0,
                max_steps=max_steps
            )
        
        elif args.model == "sadapt-inverse":
            env = BaseSurpriseAdaptWrapper(
                env,
                buffer,
                surprise_window_len=args.surprise_window_len,
                surprise_change_threshold=args.surprise_change_threshold,
                momentum=False,
                add_true_rew=args.add_true_rew,
                int_rew_scale=1.0,
                max_steps=max_steps
            )
        elif args.model == "sadapt-bandit":
            env = BaseSurpriseAdaptBanditWrapper(
                env, 
                buffer,
                add_true_rew=args.add_true_rew,
                int_rew_scale=args.int_rew_scale,
                max_steps = max_steps,
                theta_size = theta_size,
                grayscale = grayscale,
                soft_reset=args.soft_reset,
                ucb_coeff=args.ucb_coeff,
                death_cost = args.death_cost,
                exp_rew = args.exp_rew,
                use_surprise=args.use_surprise,
                threshold=threshold,
                add_random_obs=args.add_random_obs,
                bandit_step_size=args.bandit_step_size
            )
        elif args.model == "none":
            env = BaseSurpriseWrapper(
                env,
                buffer,
                add_true_rew=True,
                minimize=False,
                int_rew_scale=0.0,
                ext_only=True,
                max_steps=max_steps,
                theta_size = theta_size,
                grayscale = grayscale,
                soft_reset=args.soft_reset,
                survival_rew=args.survival_rew,
                death_cost = args.death_cost,
                threshold=threshold
            )
            
        else:
            raise ValueError(f"Unknown model {args.model}")

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(args.seed)

        return env
    return thunk

def make_csv_logger(csv_path):
    header = [
        'date',
        'env_steps',
        'ep_return',
        "ep_length",
        "ep_surprise",
        "ep_entropy",
        "rolling_alpha"
    ]
    log_level = ['logs_a']
    logger_ = csv_logger.CsvLogger(
        filename=csv_path,
        delimiter=',',
        level=logging.INFO,
        add_level_names=log_level,
        max_size=1e+9,
        add_level_nums=None,
        header=header,
    )
    return logger_


def log_heatmap(env, heatmap, ep_counter, writer, save_path):
    cmap = plt.get_cmap('Reds')
    cmap.set_under((0,0,0,0))
    cmap_args = dict(cmap=cmap)
    
    fig = plt.figure(num=1)
    #background_img = env.render()
    #background = cv2.resize(background_img, dsize=(env._env.width, env._env.height), interpolation=cv2.INTER_AREA)
    #plt.imshow(background.transpose(1,0,2) , alpha=0.75)
    plt.imshow(heatmap, **cmap_args, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.savefig(f"{save_path}/heatmap_{ep_counter}.png")
    writer.add_figure(f"trajectory/heatmap_{ep_counter}", fig, close=True)
    plt.clf()


# def _vector_env_frame(eval_envs):
#     """Safely fetch a rendered frame from a SyncVectorEnv."""
#     try:
#         frames = eval_envs.call("render")
#     except Exception:
#         return None
#     if isinstance(frames, (list, tuple)) and len(frames) > 0:
#         return frames[0]
#     return frames


# def _obs_to_device(obs, device):
#     if isinstance(obs, dict):
#         return {k: torch.as_tensor(v).to(device) for k, v in obs.items()}
#     return torch.as_tensor(obs).to(device)


# def _write_video(frames, video_dir, filename, fps=20):
#     valid_frames = []
#     for frame in frames:
#         if frame is None:
#             continue
#         frame_arr = np.asarray(frame)
#         if frame_arr.dtype != np.uint8:
#             frame_arr = np.clip(frame_arr, 0, 255).astype(np.uint8)
#         valid_frames.append(frame_arr)
#     if not valid_frames:
#         return None

#     video_path = Path(video_dir)
#     video_path.mkdir(parents=True, exist_ok=True)
#     video_file = video_path / filename
#     with imageio.get_writer(video_file, fps=fps) as writer:
#         for frame in valid_frames:
#             writer.append_data(frame)
#     return video_file


# def _log_wandb_video(video_path, metrics, global_step, video_key):
#     if video_path is None:
#         return
#     try:
#         import wandb
#     except ImportError:
#         logging.warning("wandb is not installed; skipping video upload.")
#         return

#     payload = dict(metrics)
#     payload[video_key] = wandb.Video(str(video_path), fps=20, format="mp4")
#     wandb.log(payload, step=global_step)


# def eval_episode_dqn(
#     q_network,
#     eval_envs,
#     device,
#     save_path,
#     global_step,
#     env_id,
#     track,
#     random=False,
#     max_steps=10000,
# ):
#     """Roll out one evaluation episode for DQN-style agents and save a video."""

#     if not random and q_network is None:
#         raise ValueError("q_network must be provided unless `random=True`.")

#     frames = []
#     obs, _ = eval_envs.reset()
#     frame = _vector_env_frame(eval_envs)
#     if frame is not None:
#         frames.append(frame)

#     total_reward = 0.0
#     steps = 0
#     done = False
#     episode_info = None

#     if q_network is not None:
#         was_training = q_network.training
#         q_network.eval()
#     else:
#         was_training = None

#     try:
#         while not done and steps < max_steps:
#             if random or q_network is None:
#                 action = np.array([eval_envs.single_action_space.sample()])
#             else:
#                 obs_tensor = _obs_to_device(obs, device)
#                 with torch.no_grad():
#                     logits = q_network(obs_tensor)
#                     action = torch.argmax(logits, dim=1).cpu().numpy()

#             next_obs, reward, terminated, truncated, infos = eval_envs.step(action)
#             total_reward += float(reward[0])
#             done = bool(terminated[0] or truncated[0])
#             steps += 1

#             if done and "final_info" in infos and infos["final_info"]:
#                 episode_info = infos["final_info"][0]

#             obs = next_obs
#             frame = _vector_env_frame(eval_envs)
#             if frame is not None:
#                 frames.append(frame)
#     finally:
#         if q_network is not None and was_training is not None:
#             q_network.train(was_training)

#     video_prefix = "random" if random else "policy"
#     video_dir = Path(save_path) / "videos" / env_id.replace("/", "_")
#     video_name = f"dqn_{video_prefix}_step_{global_step:09d}.mp4"
#     video_path = _write_video(frames, video_dir, video_name)

#     metrics = {
#         "eval/episode_reward": total_reward,
#         "eval/episode_length": steps,
#     }
#     if episode_info and "Average_task_return" in episode_info:
#         metrics["eval/average_task_return"] = episode_info["Average_task_return"]

#     logging.info(
#         "Eval DQN (%s) reward=%.2f length=%d video=%s",
#         "random" if random else "policy",
#         total_reward,
#         steps,
#         video_path,
#     )

#     if track:
#         _log_wandb_video(video_path, metrics, global_step, f"eval/{video_prefix}_video")

#     eval_envs.reset()
#     return metrics


# def eval_episode_ppo(
#     agent,
#     eval_envs,
#     device,
#     save_path,
#     global_step,
#     track=False,
#     max_steps=10000,
# ):
#     """Roll out one evaluation episode for PPO agents and save a video."""

#     if agent is None:
#         raise ValueError("agent must be provided for PPO evaluation.")

#     frames = []
#     obs, _ = eval_envs.reset()
#     frame = _vector_env_frame(eval_envs)
#     if frame is not None:
#         frames.append(frame)

#     total_reward = 0.0
#     steps = 0
#     done = False
#     episode_info = None

#     was_training = agent.training
#     agent.eval()

#     try:
#         while not done and steps < max_steps:
#             obs_tensor = _obs_to_device(obs, device)
#             with torch.no_grad():
#                 action, _, _, _ = agent.get_action_and_value(obs_tensor)
#             action_np = action.detach().cpu().numpy()

#             next_obs, reward, terminated, truncated, infos = eval_envs.step(action_np)
#             total_reward += float(reward[0])
#             done = bool(terminated[0] or truncated[0])
#             steps += 1

#             if done and "final_info" in infos and infos["final_info"]:
#                 episode_info = infos["final_info"][0]

#             obs = next_obs
#             frame = _vector_env_frame(eval_envs)
#             if frame is not None:
#                 frames.append(frame)
#     finally:
#         agent.train(was_training)

#     video_dir = Path(save_path) / "videos"
#     video_name = f"ppo_step_{global_step:09d}.mp4"
#     video_path = _write_video(frames, video_dir, video_name)

#     metrics = {
#         "eval/ppo_episode_reward": total_reward,
#         "eval/ppo_episode_length": steps,
#     }
#     if episode_info and "Average_task_return" in episode_info:
#         metrics["eval/ppo_average_task_return"] = episode_info["Average_task_return"]

#     logging.info(
#         "Eval PPO reward=%.2f length=%d video=%s",
#         total_reward,
#         steps,
#         video_path,
#     )

#     if track:
#         _log_wandb_video(video_path, metrics, global_step, "eval/ppo_video")

#     eval_envs.reset()
#     return metrics
 
    
# def register_griddly_envs():
#     env_dict = old_gym.envs.registration.registry.env_specs.copy()
#     for env in env_dict:
#         if 'ButterfliesEnv' in env or 'MazeEnv' in env:
#             print("Remove {} from registry".format(env))
#             del old_gym.envs.registration.registry.env_specs[env]

#     wrapper = GymWrapperFactory()
#     wrapper.build_gym_from_yaml('MazeEnv', f"{os.getcwd()}/surprise/envs/maze/maze_env.yaml")
#     wrapper.build_gym_from_yaml('MazeEnv2', f"{os.getcwd()}/surprise/envs/maze/maze_env2.yaml")
#     wrapper.build_gym_from_yaml('MazeEnvLarge', f"{os.getcwd()}/surprise/envs/maze/maze_env_large.yaml")
#     wrapper.build_gym_from_yaml('MazeEnvLarge2', f"{os.getcwd()}/surprise/envs/maze/maze_env_large2.yaml")
#     wrapper.build_gym_from_yaml('ButterfliesEnv', f"{os.getcwd()}/surprise/envs/maze/butterflies.yaml")
#     wrapper.build_gym_from_yaml('ButterfliesEnvLarge', f"{os.getcwd()}/surprise/envs/maze/butterflies_large.yaml")
#     wrapper.build_gym_from_yaml('ButterfliesEnvLarge2', f"{os.getcwd()}/surprise/envs/maze/butterflies_large2.yaml")
#     wrapper.build_gym_from_yaml('ButterfliesEnvLarge3', f"{os.getcwd()}/surprise/envs/maze/butterflies_large3.yaml")
#     wrapper.build_gym_from_yaml('ButterfliesEnvLarge4', f"{os.getcwd()}/surprise/envs/maze/butterflies_large4.yaml")
#     wrapper.build_gym_from_yaml('ButterfliesEnvLarge5', f"{os.getcwd()}/surprise/envs/maze/butterflies_large5.yaml")
#     wrapper.build_gym_from_yaml('ButterfliesEnvLarge6', f"{os.getcwd()}/surprise/envs/maze/butterflies_large6.yaml")
#     wrapper.build_gym_from_yaml('ButterfliesEnvLarge7', f"{os.getcwd()}/surprise/envs/maze/butterflies_large7.yaml")
#     wrapper.build_gym_from_yaml('ButterfliesEnvLarge8', f"{os.getcwd()}/surprise/envs/maze/butterflies_large8.yaml")
    