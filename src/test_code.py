import os
import glob
import numpy as np
import h5py
np.random.seed(0)
frame = np.random.random((32,32, 17))

with h5py.File("test.h5", "w") as file:
    dset = file.create_dataset('name1', data=frame,
                            maxshape=(32, 32, None), chunks=True, dtype='float32')
    dset.attrs['spacing'] = (0, 0.5, 1.0)
    dset.attrs['direction'] = (1., 1., 1.)

    dset = file.create_dataset("2", data=frame,
                               maxshape=(32, 32, None), chunks=True)
    dset.attrs['spacing'] = (0, 0.5, 1.0)
    dset.attrs['direction'] = (1., 1., 1.)


with h5py.File("test.h5", "r") as file:
    print(file.keys())
    A = file['name1'][:]
    print(A.dtype)
    print(file['name1'].attrs['spacing'])
    print(file['name1'].attrs['direction'])
    print(file.name)
    print(file['name1'].name)
    
    print(np.allclose(A, frame))

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    print("Start")
