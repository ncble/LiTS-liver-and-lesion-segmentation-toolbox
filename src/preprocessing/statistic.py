"""
Copyright (C) 2020  Lu Lin
Warning: The code is released under the license GNU GPL 3.0



"""
import os
import glob
import numpy as np
import h5py
from medpy.io import load as med_load
# from medpy.io import save as med_save
import medpy


class Statistic(object):

    def __init__(self, filepath):
        super(Statistic, self).__init__()
        self.h5py_dataset = h5py.File(filepath, "r")
        # with h5py.File(filepath, "r") as file:
        self.volume_start_index = self.h5py_dataset['volume_start_index']
        self.num_vol = len(self.volume_start_index)

    def compute_intensity(self):
        # direct computation or medpy.intensity
        # 

        return

    def close(self):
        self.h5py_dataset.close()


if __name__ == "__main__":
    print("Start")
