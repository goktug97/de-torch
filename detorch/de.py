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
    rand2bin = 2
    best2bin = 3
    currenttobest1bin = 4
    randtobest1bin = 5


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
    :ivar List[Policy] population
    :ivar gym.Env Gym environment
    """
    @hook
    def __init__(self, config: Config):
        config.check_config()
        self.config = config

        self.env = self.make_env(**config.environment.to_dict())

        np.random.seed(config.de.seed)
        random.seed(config.de.seed)
        torch.manual_seed(config.de.seed)
        torch.cuda.manual_seed(config.de.seed)
        torch.cuda.manual_seed_all(config.de.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.env.seed(config.de.seed)
        self.env.action_space.seed(config.de.seed)

        self.differential_weight = self.config.de.differential_weight

        if self.config.de.crossover_probability is None:
            policy = self.make_policy(**config.policy.to_dict())
            size = len(torch.nn.utils.parameters_to_vector(policy.parameters()))
            self.crossover_probability = calculate_cr(size)
        else:
            self.crossover_probability = self.config.de.crossover_probability

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

    def sample_parameters(self, size):
        samples = self.population[self.rng.integers(0, len(self.population), size)]
        params = [torch.nn.utils.parameters_to_vector(sample.parameters()) for sample in samples]
        return params

    @hook
    def mutate(self, policy):
        """Mutate the given policy."""
        policy_params = torch.nn.utils.parameters_to_vector(policy.parameters())
        best_params = torch.nn.utils.parameters_to_vector(self.population[self.current_best].parameters())
        if self.config.de.strategy == Strategy.rand1bin:
            params = self.sample_parameters(3)
            diff = params[1] - params[2]
            p0 = params[0]
        elif self.config.de.strategy == Strategy.best1bin:
            params = self.sample_parameters(2)
            diff = params[0] - params[1]
            p0 = best_params
        elif self.config.de.strategy == Strategy.rand2bin:
            params = self.sample_parameters(5)
            diff = params[1] - params[2] + params[3] - params[4]
            p0 = params[0]
        elif self.config.de.strategy == Strategy.best2bin:
            params = self.sample_parameters(4)
            diff = params[0] - params[1] + params[2] - params[3]
            p0 = best_params
        elif self.config.de.strategy == Strategy.randtobest1bin:
            params = self.sample_parameters(3)
            diff = best_params - params[0] + params[1] - params[2]
            p0 = params[0]
        elif self.config.de.strategy == Strategy.currenttobest1bin:
            params = self.sample_parameters(2)
            diff = best_params - policy_params + params[0] - params[1]
            p0 = policy_params
        else:
            raise NotImplementedError

        mutation = p0 + self.differential_weight * diff
        cross = torch.rand(policy_params.shape) <= self.crossover_probability
        policy_params[cross] = mutation[cross]
        torch.nn.utils.vector_to_parameters(policy_params, policy.parameters())

    @hook
    def eval_population(self, population):
        comm = MPI.COMM_WORLD
        rewards = []
        reward_array = np.zeros(len(self.population), dtype=np.float32)
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

        while self.gen < self.config.de.n_step:
            if isinstance(self.config.de.differential_weight, tuple):
                self.differential_weight = self.rng.uniform(*self.config.de.differential_weight)
            population = self.generate_candidates()
            rewards = self.eval_population(population)
            self.selection(population, rewards)
            self.gen += 1

        sys.stdout = sys.__stdout__
        torch.set_grad_enabled(True)
