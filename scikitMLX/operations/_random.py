import mlx.core as mx
import random


class generator:
    distributions = [
        "bernoulli",
        "categorical",
        "gumbel",
        "normal",
        "randint",
        "uniform",
        "truncated_normal",
    ]

    def __init__(self, seed=None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            self._key = mx.random.key(seed)

        elif isinstance(seed, int):
            self._key = mx.random.key(seed)

        elif isinstance(seed, mx.array):
            self._key = seed

        else:
            raise ValueError(
                f"'{seed}' cannot be used to seed a 'skML.operations.generator' instance"
            )

    def _wrap_generator(self, generator):
        def wrapped_generator(*args, **kwargs):
            return generator(*args, **kwargs, key=self._key)

        self._key = mx.random.split(self._key, 1)[0]
        return wrapped_generator

    def __getattr__(self, name):
        if name in self.distributions:
            generator = getattr(mx.random, name)
            return self._wrap_generator(generator)

        raise AttributeError(
            f"module 'skML.operations.generator' has no method '{name}'"
        )
