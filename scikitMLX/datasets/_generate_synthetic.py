import mlx.core as mx
import numpy as np

from .. import operations as ops


def gen_swiss_roll(n_samples=100, *, noise=0.0, random_state=None, hole=False):
    generator = ops.generator(random_state)

    if not hole:
        t = 1.5 * np.pi * (1 + 2 * generator.uniform(size=n_samples))
        y = 21 * generator.uniform(size=n_samples)
    else:
        corners = mx.array(
            [[np.pi * (1.5 + i), j * 7] for i in range(3) for j in range(3)]
        )
        corners = ops.arr.delete(corners, 4, axis=0)
        corner_index = generator.choice(8, n_samples)
        parameters = generator.uniform(size=(2, n_samples)) * mx.array([[np.pi], [7]])
        t, y = corners[corner_index].T + parameters

    x = t * mx.cos(t)
    z = t * mx.sin(t)

    X = mx.stack((x, y, z), axis=0)
    X += noise * generator.normal(size=(3, n_samples))
    X = X.T
    t = mx.squeeze(t)

    return X, t
