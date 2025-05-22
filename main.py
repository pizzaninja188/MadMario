import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import datetime
from pathlib import Path
import torch
import gym
import gym_super_mario_bros
from gym.vector import AsyncVectorEnv
from nes_py.wrappers import JoypadSpace

from wrappers import ResizeObservation, SkipFrame
from agent import Mario
from metrics import MetricLogger

from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

def make_env():
    def _thunk():
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        env = JoypadSpace(env, [['right'], ['right', 'A']])
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, shape=84)
        env = TransformObservation(env, f=lambda x: x / 255.)
        env = FrameStack(env, 4)
        return env
    return _thunk

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)  # Prevent PyTorch from hogging CPU

    num_envs = 2
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    def get_latest_checkpoint():
        all_checkpoints = sorted(Path("checkpoints").rglob("mario_net_*.chkpt"))
        return all_checkpoints[-1] if all_checkpoints else None

    checkpoint = get_latest_checkpoint()
    mario = Mario(state_dim=(4, 84, 84), action_dim=envs.single_action_space.n, save_dir=save_dir, checkpoint=checkpoint)
    start_episode = mario.episode if checkpoint else 0
    logger = MetricLogger(save_dir)

    episodes = 40000
    episode_rewards = [0.0] * num_envs
    last_recorded = mario.episode

    try:
        states = envs.reset()

        for episode in range(start_episode, episodes):
            actions = mario.act_batch(states)
            next_states, rewards, dones, infos = envs.step(actions)

            # cache and learn
            mario.cache(states, next_states, actions, rewards, dones)
            q_values, losses = mario.learn()

            avg_q = sum(q_values) / len(q_values) if q_values else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            # Log per step
            logger.log_step(sum(rewards), avg_loss, avg_q)

            for i, done in enumerate(dones):
                episode_rewards[i] += rewards[i]
                if done:
                    logger.log_episode()
                    print(f"Env {i} finished episode {mario.episode + 1} with reward {episode_rewards[i]}")
                    episode_rewards[i] = 0
                    mario.episode += 1

            states = next_states

            # Periodic logging every 20 total episodes
            if mario.episode - last_recorded >= 20:
                logger.record(
                    episode=mario.episode,
                    epsilon=mario.exploration_rate,
                    step=mario.curr_step
                )
                last_recorded = mario.episode
                pass
            
            mario.episode += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        mario.save()
        print("Checkpoint saved. Exiting gracefully.")
