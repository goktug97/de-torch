from typing import Optional, Type, Callable, Any, Union, Tuple

import torch
from pipcs import Config, Required, required, Choices

from .de import Policy, DE, Strategy


default_config = Config()


@default_config('policy')
class PolicyConfig():
    """**name: policy**

    :ivar Required[Type[Policy]] policy: torch.nn.Module with a rollout method
    """
    policy: Required[Type[Policy]] = required


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
    n_step: Required[int] = required
    population_size: int = 32
    differential_weight: Union[Tuple[float, float], float] = 0.02
    beta: float = 0.0
    crossover_probability: float = None
    strategy: Choices[Strategy] = Choices(list(Strategy), default=Strategy.best1bin)
    max_age: Optional[int] = None
    seed: int = 123
