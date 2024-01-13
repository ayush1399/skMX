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

    def __choice(self, arr, size=None, replace=True, p=None, key=None):
        if not replace and size is not None and size > len(arr):
            raise ValueError(
                "Cannot take a larger sample than population when 'replace=False'"
            )

        if p is None:
            p = mx.ones(len(arr)) / len(arr)
        p = mx.array(p)

        if replace:
            logits = p.reshape(-1, 1)
            sampled_indices = mx.random.categorical(logits, num_samples=size, key=key)
            return mx.array(arr)[sampled_indices]
        else:
            sampled_indices = []
            for _ in range(size):
                logits = p.reshape(-1, 1)
                index = mx.random.categorical(logits, num_samples=1, key=key)
                sampled_indices.append(index[0])
                p[index] = 0
                p /= p.sum()

            return mx.array(arr)[sampled_indices]

    def choice(self, a, size=None, replace=True, p=None):
        if isinstance(a, int):
            a = list(range(a))
        choices = self.__choice(a, size=size, replace=replace, p=p, key=self._key)
        self._key = mx.random.split(self._key, 1)[0]
        return choices

    def __getattr__(self, name):
        if name in self.distributions:
            generator = getattr(mx.random, name)
            return self._wrap_generator(generator)

        raise AttributeError(
            f"module 'skML.operations.generator' has no method '{name}'"
        )
