import gym
from pipcs import Config
import torch
from torch import nn

from detorch import DE, Policy, default_config, Strategy


class Agent(Policy):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(4, 8),
            nn.Linear(8, 2))

    def rollout(self, env):
        done = False
        obs = env.reset()
        total_reward = 0
        while not done:
            obs_tensor = torch.from_numpy(obs).float()
            action = self.seq(obs_tensor).argmax().item()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        return total_reward


config = Config(default_config)


@config('environment')
class EnvironmentConfig():
    make_env = gym.make
    id: str = 'CartPole-v1'


@config('policy')
class PolicyConfig():
    policy = Agent


@config('de')
class DEConfig():
    n_rollout = 2
    n_step = 5
    population_size = 256
    strategy = Strategy.best1bin
    seed = 123123



if __name__ == '__main__':
    de = DE(config)

    @de.selection.add_hook()
    def after_selection(self, *args, **kwargs):
        print(f'Generation: {self.gen} Best Reward: {self.rewards[self.current_best]}')


    de.train()
