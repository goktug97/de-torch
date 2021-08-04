import os
import sys
import random
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
    scaledbest1bin = 6
    scaledrand1bin = 7
    rand1soft = 8
    best1soft = 9
    rand2soft = 10
    best2soft = 11
    currenttobest1soft = 12
    randtobest1soft = 13
    scaledbest1soft = 14
    scaledrand1soft = 15


class Policy(nn.Module, ABC):
    """Abstract subclass of nn.Module."""
    def __init__(self):
        self._age = 0
        super().__init__()

    @abstractmethod
    def evaluate(self):
        """This function should be implemented by the user and it
        should evaluate the model and return reward or negative loss."""
        pass


class DE():
    """
    :ivar int gen: Current generation
    :ivar List[Policy] population
    """
    @hook
    def __init__(self, config: Config):
        config.check_config()
        self.config = config

        np.random.seed(config.de.seed)
        random.seed(config.de.seed)
        torch.manual_seed(config.de.seed)
        torch.cuda.manual_seed(config.de.seed)
        torch.cuda.manual_seed_all(config.de.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.differential_weight = self.config.de.differential_weight

        if self.config.de.crossover_probability is None:
            policy = self.make_policy(**config.policy.to_dict())
            size = len(torch.nn.utils.parameters_to_vector(policy.parameters()))
            self.crossover_probability = calculate_cr(size)
        else:
            self.crossover_probability = self.config.de.crossover_probability

        self.rng = np.random.default_rng(config.de.seed)

        self.gen = 0

        self.population = self.create_population()

        self.rewards = None

    def create_population(self):
        population = []
        for _ in range(self.config.de.population_size):
            policy = self.make_policy(**self.config.policy.to_dict())
            population.append(policy)
        return np.array(population)

    @staticmethod
    def make_policy(policy, **kwargs):
        """Helper function to create a policy."""
        assert issubclass(policy, Policy)
        return policy(**kwargs)

    def sample_parameters(self, size):
        idxs = self.rng.integers(0, len(self.population), size)
        samples = self.population[idxs]
        params = [torch.nn.utils.parameters_to_vector(sample.parameters()) for sample in samples]
        return params, idxs

    @hook
    def mutate(self, policy):
        """Mutate the given policy."""
        policy_params = torch.nn.utils.parameters_to_vector(policy.parameters())
        best_params = torch.nn.utils.parameters_to_vector(self.population[self.current_best].parameters())
        if self.config.de.strategy % 8 == Strategy.rand1bin:
            params, _ = self.sample_parameters(3)
            diff = params[1] - params[2]
            p0 = params[0]
        elif self.config.de.strategy % 8 == Strategy.best1bin:
            params, _ = self.sample_parameters(2)
            diff = params[0] - params[1]
            p0 = best_params
        elif self.config.de.strategy % 8 == Strategy.rand2bin:
            params, _ = self.sample_parameters(5)
            diff = params[1] - params[2] + params[3] - params[4]
            p0 = params[0]
        elif self.config.de.strategy % 8 == Strategy.best2bin:
            params, _ = self.sample_parameters(4)
            diff = params[0] - params[1] + params[2] - params[3]
            p0 = best_params
        elif self.config.de.strategy % 8 == Strategy.randtobest1bin:
            params, _ = self.sample_parameters(3)
            diff = best_params - params[0] + params[1] - params[2]
            p0 = params[0]
        elif self.config.de.strategy % 8 == Strategy.currenttobest1bin:
            params, _ = self.sample_parameters(2)
            diff = best_params - policy_params + params[0] - params[1]
            p0 = policy_params
        elif self.config.de.strategy % 8 == Strategy.scaledbest1bin:
            params, idxs = self.sample_parameters(2)
            rewards = rank_transformation(self.rewards)[idxs]
            distances = torch.stack(params) - policy_params
            diff = torch.from_numpy(rewards) @ distances
            p0 = best_params
        elif self.config.de.strategy % 8 == Strategy.scaledrand1bin:
            params, idxs = self.sample_parameters(3)
            rewards = rank_transformation(self.rewards)[idxs[1:]]
            distances = torch.stack(params[1:]) - policy_params
            diff = torch.from_numpy(rewards) @ distances
            p0 = params[0]
        else:
            raise NotImplementedError

        mutation = p0 + self.differential_weight * diff

        if self.config.de.strategy <= 7:
            cross = torch.rand(policy_params.shape) <= self.crossover_probability
            policy_params[cross] = mutation[cross]
        else:
            policy_params = self.crossover_probability * policy_params + (1.0 - self.crossover_probability) * mutation

        torch.nn.utils.vector_to_parameters(policy_params, policy.parameters())

    @hook
    def eval_population(self, population):
        comm = MPI.COMM_WORLD
        rewards = []
        reward_array = np.zeros(len(self.population), dtype=np.float32)
        batch = np.array_split(population, comm.Get_size())[comm.Get_rank()]
        for policy in batch:
            rewards.append(policy.evaluate())
        comm.Allgatherv([np.array(rewards, dtype=np.float32), MPI.FLOAT], [reward_array, MPI.FLOAT])
        return reward_array

    @hook
    def generate_candidates(self):
        """Generate new candidates by mutating the population."""
        population = self.create_population()
        for idx, policy in enumerate(population):
            policy.load_state_dict(self.population[idx].state_dict())
            self.mutate(policy)
        return population

    @hook
    def selection(self, population, rewards):
        """Select policies from the given population that are better than their parents."""
        comp = rewards >= self.rewards
        self.population[comp] = population[comp]
        self.rewards[comp] = rewards[comp]
        self.current_best = np.argmax(self.rewards)

    @hook
    def step(self):
        if self.rewards is None:
            self.rewards = self.eval_population(self.population)
            self.current_best = np.argmax(self.rewards)
        if isinstance(self.config.de.differential_weight, tuple):
            self.differential_weight = self.rng.uniform(*self.config.de.differential_weight)
        population = self.generate_candidates()
        rewards = self.eval_population(population)
        self.selection(population, rewards)
        if self.config.de.max_age is not None:
            for idx, policy in enumerate(self.population):
                policy._age += 1
                if policy._age > self.config.de.max_age:
                    self.population[idx] = self.make_policy(**self.config.policy.to_dict())
        self.gen += 1

    def train(self):
        torch.set_grad_enabled(False)
        rank = MPI.COMM_WORLD.Get_rank()
        if not rank == 0:
            f = open(os.devnull, 'w')
            sys.stdout = f

        while self.gen < self.config.de.n_step:
            self.step()

        sys.stdout = sys.__stdout__
        torch.set_grad_enabled(True)
