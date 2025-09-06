import numpy as np


def min_max_norm_np(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)


def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def nested_list_shape(lst):
    if not lst:
        return (0,)

    shape = [len(lst)]
    while isinstance(lst[0], list):
        shape.append(len(lst[0]))
        lst = lst[0]

    return tuple(shape)
