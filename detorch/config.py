from typing import Optional, Type, Callable, Any, Union, Tuple

import torch

from .de import Policy, DE, Strategy
from .dataclass_config import *


default_config = Config()


@default_config('policy')
class PolicyConfig():
    """**name: policy**

    :ivar Required[Type[Policy]] policy: torch.nn.Module with a rollout method
    """
    policy: Required[Type[Policy]] = Required()


@default_config('de')
class DEConfig():
    """**name: de**

    :ivar int population_size
    :ivar int n_step: Number of training steps
    :ivar Union[Tuple[float, float], float] differential_weight: The mutation constant.
    :ivar float crossover_probability
    :ivar Strategy strategy: Recombination strategy
    :ivar Optional[int] seed: Random seed
    """
    n_step: Required[int] = Required()
    population_size: int = 32
    differential_weight: Union[Tuple[float, float], float] = 0.02
    beta: float = 0.0
    crossover_probability: float = None
    strategy: Choice[Strategy] = Choice(list(Strategy), default=Strategy.best1bin)
    seed: int = 123
