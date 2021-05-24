from typing import Callable, List, Optional
from dataclasses import field
from itertools import tee

import gym
from pipcs import Config
import torch
from torch import nn
import numpy as np

from detorch import DE, Policy, default_config, hook, Strategy


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Agent(Policy):
    def __init__(self, input_size, hidden_layers, output_size, output_func,
                 hidden_act = None, output_act=None, bias=True):
        super().__init__()
        layers: List[nn.Module] = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for idx, (layer1, layer2) in enumerate(pairwise(layer_sizes)):
            layers.append(nn.Linear(layer1, layer2, bias=bias))
            if hidden_act is not None:
                if idx < len(layer_sizes) - 2:
                    layers.append(hidden_act())
        if output_act is not None:
            layers.append(output_act())
        self.output_func = output_func
        self.seq = nn.Sequential(*layers)

    def rollout(self, env):
        done = False
        obs = env.reset()
        total_reward = 0
        while not done:
            obs_tensor = torch.from_numpy(obs).float()
            action = self.output_func(self.seq(obs_tensor))
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        return total_reward


config = Config(default_config)


def make_env(id):
    return gym.make(id)


@config('environment')
class EnvironmentConfig():
    make_env = make_env
    id: str = 'LunarLander-v2'


@config('policy')
class PolicyConfig():
    policy = Agent
    env = DE.make_env(**config.environment.to_dict())
    input_size: int = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        output_size: int = env.action_space.n
        output_func: Callable[[int], int] = lambda x: x.argmax().item()
    else:
        output_size: int = env.action_space.shape[0]
        output_func: Callable[[int], int] = lambda x: x.detach().numpy()
    hidden_layers: List[int] = field(default_factory=lambda: [])
    # hidden_act: Optional[nn.Module] = nn.ReLU
    bias: bool = True


@config('de')
class DEConfig():
    n_rollout = 5
    n_step = 100
    population_size = 256
    differential_weight = (0.7, 1.0)
    crossover_probability = None
    strategy = Strategy.best1bin
    seed = 123123


if __name__ == '__main__':
    de = DE(config)

    @de.selection.add_hook()
    def after_selection(self, *args, **kwargs):
        print(f'Generation: {self.gen} Best Reward: {self.rewards[self.current_best]}')

    de.train()

    # Evaluate and render the best policy.
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        de.env = gym.wrappers.Monitor(de.env, './videos', force = True)
        setattr(de, 'env.step', hook(de.env.step.__func__))
        @de.env.step.add_hook(after=True)
        def render(self, *args, **kwargs):
            self.render()
        reward = de.eval_policy(de.population[de.current_best])
        print(f'Reward: {reward}')
