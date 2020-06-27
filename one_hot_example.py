import os
import glob
import numpy as np


def one_hot0(mask, num_classes=3):
    """
    mask.shape = (H, W)

    return one_hot (shape = (num_classes, H, W))
    """
    one_hot = mask == np.arange(num_classes)[:, None, None]

    return one_hot


def one_hot1(mask, num_classes=3):
    """
    mask.shape = (H, W)

    """
    mask_flatten = mask.ravel()
    one_hot = np.zeros((*mask_flatten.shape, num_classes), dtype=int)
    one_hot[np.arange(len(mask_flatten)), mask_flatten] = 1
    # import ipdb; ipdb.set_trace()
    return one_hot

if __name__ == "__main__":
    print("Start")

    mask = np.random.randint(0, high=3, size=(512, 512))
    out = one_hot1(mask)
    # usage: %timeit one_hot0(mask)  # 286 µs ± 299 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    # usage: %timeit one_hot1(mask)  # 1.75 ms ± 19.6 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

