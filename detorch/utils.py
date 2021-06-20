import math
import numpy as np
from functools import wraps, lru_cache


def hook(func):
    func.before_register = []
    func.after_register = []
    def add_hook(after=True):
        def _add_hook(hook):
            if after:
                func.after_register.append(hook)
            else:
                func.before_register.append(hook)
        return _add_hook

    func.add_hook = add_hook

    @wraps(func)
    def wrapped(*args, **kwargs):
        for hook in func.before_register:
            hook(*args, **kwargs)
        ret = func(*args, **kwargs)
        for hook in func.after_register:
            hook(*args, **kwargs, ret=ret)
        return ret

    return wrapped


def calculate_cr(size):
    return np.exp(-0.5) / (np.log(size) * np.sqrt(2 * np.pi) + np.exp(-0.5))


@lru_cache(maxsize=1)
def _center_function(population_size):
    centers = np.arange(0, population_size, dtype=np.float32)
    centers = centers / (population_size - 1)
    centers -= 0.5
    centers *= 2.0
    return centers


def _compute_ranks(rewards):
    rewards = np.array(rewards)
    ranks = np.empty(rewards.size, dtype=int)
    ranks[rewards.argsort()] = np.arange(rewards.size)
    return ranks


def rank_transformation(rewards):
    ranks = _compute_ranks(rewards)
    values = _center_function(rewards.size)
    return values[ranks]
