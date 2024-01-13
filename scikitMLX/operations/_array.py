import numpy as np


class array:
    @staticmethod
    def delete(arr, obj, axis=0):
        if axis < 0:
            axis += arr.ndim

        if axis >= arr.ndim:
            raise ValueError("Axis is out of bounds for array")

        if np.isscalar(obj):
            obj = [obj]

        # Convert negative indices to positive ones
        obj = [index + arr.shape[axis] if index < 0 else index for index in obj]

        # Create an index mask excluding the indices in obj
        mask = [i for i in range(arr.shape[axis]) if i not in obj]

        # Use advanced indexing to select elements
        indices = [slice(None)] * arr.ndim
        indices[axis] = mask
        return arr[tuple(indices)]
