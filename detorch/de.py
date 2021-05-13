import os
import sys
import random
import copy
from abc import ABC, abstractmethod
from enum import IntEnum

import torch
from torch import nn
import numpy as np
from mpi4py import MPI
from pipcs import Config

from .utils import *


class Strategy(IntEnum):
    rand1bin = 0
    best1bin = 1


class Policy(nn.Module, ABC):
    """Abstract subclass of nn.Module."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def rollout(self, env):
        """This function should be implemented by the user and
        it should evaluate the model in the given environment and
        return the total reward."""
        pass


class DE():
    """
    :ivar int gen: Current generation
    :ivar Policy policy: Trained policy
    """
    @hook
    def __init__(self, config: Config):
        config.check_config()
        self.config = config

        self.env = self.make_env(**config.environment.to_dict())

        torch.manual_seed(config.de.seed)
        torch.cuda.manual_seed(config.de.seed)
        torch.cuda.manual_seed_all(config.de.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        np.random.seed(config.de.seed)
        random.seed(config.de.seed)
        self.env.seed(config.de.seed)
        self.env.action_space.seed(config.de.seed)

        self.rng = np.random.default_rng(config.de.seed)

        self.gen = 0

        self.population = []
        for _ in range(config.de.population_size):
            policy = self.make_policy(**config.policy.to_dict())
            self.population.append(policy)
        self.population = np.array(self.population)

    @staticmethod
    def make_env(make_env, **kwargs):
        """Helper function to create a gym environment."""
        return make_env(**kwargs)

    @staticmethod
    def make_policy(policy, **kwargs):
        """Helper function to create a policy."""
        assert issubclass(policy, Policy)
        return policy(**kwargs)

    @hook
    def eval_policy(self, policy):
        """Evaluate policy on the ``self.env`` for ``self.config.de.n_rollout times``"""
        total_reward = 0
        for _ in range(self.config.de.n_rollout):
            total_reward += policy.rollout(self.env)
        return total_reward / self.config.de.n_rollout

    @hook
    def sample(self):
        """Sample 3 policies from the population based on the strategy."""
        if self.config.de.strategy == Strategy.rand1bin:
            return self.population[self.rng.integers(0, self.config.de.population_size, 3)]
        elif self.config.de.strategy == Strategy.best1bin:
            return self.population[[self.current_best, *self.rng.integers(0, self.config.de.population_size, 2)]]
        else:
            raise NotImplementedError

    @hook
    def mutate(self, policy):
        """Mutate the given policy."""
        random_ps = self.sample()

        for parameter, p1_parameter, p2_parameter, p3_parameter in zip(
                policy.parameters(), *[p.parameters() for p in random_ps]):
            diff = (p2_parameter.data - p3_parameter.data)
            mutation = p1_parameter.data + self.config.de.differential_weight * diff
            noncross = torch.rand(parameter.shape) >= self.config.de.crossover_probability
            mutation[noncross] = parameter.data[noncross]
            parameter.data.copy_(mutation.data)

    @hook
    def eval_population(self, population):
        comm = MPI.COMM_WORLD
        rewards = []
        reward_array = np.zeros(self.config.de.population_size, dtype=np.float32)
        batch = np.array_split(population, comm.Get_size())[comm.Get_rank()]
        for policy in batch:
            rewards.append(self.eval_policy(policy))
        comm.Allgatherv([np.array(rewards, dtype=np.float32), MPI.FLOAT], [reward_array, MPI.FLOAT])
        return reward_array

    @hook
    def generate_candidates(self):
        """Generate new candidates by mutating the population."""
        population = copy.deepcopy(self.population)
        for policy in population:
            self.mutate(policy)
        return population

    @hook
    def selection(self, population, rewards):
        """Select policies from the given population that are better than their parents."""
        comp = rewards >= self.rewards
        self.population[comp] = population[comp]
        self.rewards[comp] = rewards[comp]
        self.current_best = np.argmax(self.rewards)

    def train(self):
        torch.set_grad_enabled(False)
        rank = MPI.COMM_WORLD.Get_rank()
        if not rank == 0:
            f = open(os.devnull, 'w')
            sys.stdout = f

        self.rewards = self.eval_population(self.population)
        self.current_best = np.argmax(self.rewards)

        for self.gen in range(self.config.de.n_step):
            population = self.generate_candidates()
            rewards = self.eval_population(population)
            self.selection(population, rewards)

        sys.stdout = sys.__stdout__
        torch.set_grad_enabled(True)
