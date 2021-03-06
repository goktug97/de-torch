from typing import Callable, List, Optional, Type
from dataclasses import field
from itertools import tee

import gym
import torch
from torch import nn
import numpy as np

from detorch import DE, Policy, hook, Strategy
from detorch.config import default_config, Config

from modules import Agent


config = Config(default_config)


@config('de')
class DEConfig():
    n_step: int = 200
    population_size: int = 256
    differential_weight: float = 0.7
    crossover_probability: float = 0.05
    strategy: Strategy = Strategy.scaledbest1bin
    seed: int = 123123


@config('policy')
class PolicyConfig():
    policy: Type[Policy] = Agent

    # Below variables are user settings
    # they are passed to Agent as kwargs
    env_id: str = 'LunarLander-v2'
    n_rollout: int = 5
    hidden_layers: List[int] = field(default_factory=lambda: [])
    # hidden_act: Optional[nn.Module] = nn.ReLU
    # output_act: Optional[nn.Module] = nn.Tanh
    bias: bool = True
    seed: int = config.de.seed


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
        best_policy = de.population[de.current_best]
        best_policy.env = gym.wrappers.Monitor(best_policy.env, './videos', force = True)
        setattr(best_policy, 'env.step', hook(best_policy.env.step.__func__))
        @best_policy.env.step.add_hook(after=True)
        def render(self, *args, **kwargs):
            self.render()
        reward = best_policy.evaluate()
        print(f'Reward: {reward}')
