from typing import Callable, List, Optional, cast
from dataclasses import field
from itertools import tee
import copy
import os
import sys

from mpi4py import MPI
import gym
from pipcs import Config
import torch
from torch import nn
import numpy as np

from detorch import DE, Policy, default_config, hook, Strategy
import nes  # pip install nes-torch


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Agent(Policy):
    def __init__(self, input_size, hidden_layers, output_size, output_func,
                 hidden_act=None, output_act=None, bias=True):
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


de_config = Config(default_config)
nes_config = Config(nes.default_config)


def make_env(id):
    return gym.make(id)


@de_config('environment')
@nes_config('environment')
class EnvironmentConfig():
    make_env = make_env
    id: str = 'LunarLander-v2'


@de_config('policy')
@nes_config('policy')
class PolicyConfig():
    policy = Agent
    env = DE.make_env(**de_config.environment.to_dict())
    input_size: int = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        output_size: int = env.action_space.n
        output_func: Callable[[int], int] = lambda x: x.argmax().item()
    else:
        output_size: int = env.action_space.shape[0]
        output_func: Callable[[int], int] = lambda x: x.detach().numpy()
    hidden_layers: List[int] = field(default_factory=lambda: [])
    # hidden_act: nn.Module = nn.ReLU
    bias: bool = True


@nes_config('policy')
class PolicyConfig():
    policy = type('Policy', (Agent, nes.Policy,), {})  # To supress assertion


@nes_config('optimizer')
class OptimizerConfig():
    lr = 0.02
    optim_type = torch.optim.Adam


@de_config('de', check=False)
@nes_config('nes', check=False)
class CommonConfig():
    n_rollout = 5
    seed = 123123


@nes_config('nes')
class NESConfig():
    n_step = 50
    population_size = 256
    l2_decay = 0.005
    sigma = 0.02

@de_config('de')
class DEConfig():
    n_step = 50
    population_size = 256
    differential_weight = (0.7, 1.0)
    crossover_probability = None
    strategy = Strategy.best1bin


if __name__ == '__main__':
    @nes.NES.__init__.add_hook()
    def init(self, *args, **kwargs):
        self.best_reward = -np.inf

    @DE.selection.add_hook()
    def after_selection(self, *args, **kwargs):
        print(f'Generation: {self.gen} Best Reward: {self.rewards[self.current_best]}')

    de = DE(de_config)

    @nes.NES.optimize.add_hook()
    def after_optimize(self, *args, **kwargs):
        reward = self.eval_policy(self.policy)
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_policy = copy.deepcopy(self.policy)
        print(f'Generation: {self.gen+de.gen} Best Reward: {self.best_reward} Eval Reward: {reward}')

    de.train()

    nes = nes.NES(nes_config)

    nes.policy = nes.make_policy(**nes_config.policy.to_dict())
    nes.policy.load_state_dict(de.population[de.current_best].state_dict())
    nes.optim = nes.make_optimizer(nes.policy, **nes_config.optimizer.to_dict())

    nes.train()

    # Evaluate and Render
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:

        nes.env = gym.wrappers.Monitor(nes.env, './videos', force = True)

        setattr(nes, 'env.step', hook(nes.env.step.__func__))
        @nes.env.step.add_hook(after=False)
        def render(self, *args, **kwargs):
            self.render()

        print(f'Reward: {nes.eval_policy(nes.best_policy)}')
