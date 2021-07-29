from pipcs import Config
import torch
from torch import nn
import numpy as np

from detorch import DE, Policy, default_config, Strategy


config = Config(default_config)


class Ackley(Policy):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.rand(1), requires_grad=False)
        self.y = nn.Parameter(torch.rand(1), requires_grad=False)

    def evaluate(self):
        x = self.x
        y = self.y
        first_term = -20 * torch.exp(-0.2*torch.sqrt(0.5*(x**2+y**2)))
        second_term = -torch.exp(0.5*(torch.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.e + 20
        return -(second_term + first_term).item()


@config('policy')
class PolicyConfig():
    policy = Ackley


@config('de')
class DEConfig():
    n_step = 20
    population_size = 256
    differential_weight = 0.7
    crossover_probability = 0.5
    strategy = Strategy.scaledbest1soft
    seed = 123123


if __name__ == '__main__':
    de = DE(config)

    @de.selection.add_hook()
    def after_selection(self, *args, **kwargs):
        print(f'Generation: {self.gen} Best Reward: {self.rewards[self.current_best]}')

    de.train()
