"""
Copyright (C) 2020  Lu Lin
Warning: The code is released under the license GNU GPL 3.0


"""
import os
import glob
import argparse
import numpy as np
import time
import cv2
import torch
from torch.multiprocessing import Queue, Process
import torch.utils.data
from torchvision import transforms
import h5py
from PIL import Image, ImageEnhance

class LiTSDataset(torch.utils.data.Dataset):
    r"""LiTS dataset loader for training (Pytorch natively supported)

    Work with hdf5 or npy/npz file format. 
    Issue: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643


    """

    def __init__(self, filepath,  # sample_indices,
                 transform=None,
                 dtype=np.float32):
        super(LiTSDataset, self).__init__()
        self.filepath = filepath
        self.dataset = None
        self.transform = transform
        self.dtype = dtype
        with h5py.File(self.filepath, "r") as file:
            self.num_samples = file['volumes'].shape[0]

    def __len__(self):
        # the total number of samples
        return self.num_samples

    def __getitem__(self, index):
        # Generates one sample of data
        if self.dataset is None:
            self.dataset = h5py.File(self.filepath, "r")

        img = self.dataset['volumes'][index]  # shape (512, 512)
        msk = self.dataset['segmentations'][index] #.astype(self.dtype)  # shape (512, 512)
        # print(img.shape, msk.shape)
        # print(img.dtype, msk.dtype)
        if self.transform:
            # convert to 
            img = img[..., None] # new shape (512,512,1) for PIL Image transform
            img = self.transform(img)
            # msk = self.transform(msk) ## TODO TODO TODO
            
            # img = img[None, ...] # new shape (1, 512,512)
            # img = torch.from_numpy(img)  # (sharing storage without copying)
            msk = torch.from_numpy(msk)  # cf. torch.tensor: copy data
            # print(img.shape, msk.shape)
        return img, msk


def infinite_dataloader(dataloader):
    while True:
        for X in dataloader:
            yield X


if __name__ == "__main__":
    print("Start")
    # usage
    # python dataloader.py -f "./data/train_LiTS_db.h5" --show
    parser = argparse.ArgumentParser(description="Visualize (static) dataloader results (for debug purpose)")
    parser.add_argument('-f', "--filepath", type=str, default=None, required=True,
                        help="dataset filepath.")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--num-cpu", type=int, default=8,
                        help="Number of CPUs to use in parallel for dataloader.")
    parser.add_argument("--shuffle", action="store_true", default=False,
                        help="Shuffle the dataset")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument('--img-shape', type=str, default="(1,512,512)",
                        help='Image shape (default "(1,512,512)"')
    args = parser.parse_args()
    img_shape = tuple(map(int, args.img_shape.strip()[1:-1].split(",")))

    transform_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                               std=[0.5, 0.5, 0.5])

    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(64),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        # transform_normalize # TODO TODO enable it
    ])

    data_loader_params = {'batch_size': args.batch_size,
                          'shuffle': args.shuffle,
                          'num_workers': args.num_cpu,
                          #   'sampler': balanced_sampler,
                          'drop_last': True,
                          'pin_memory': False
                          }

    train_set = LiTSDataset(args.filepath,
                            dtype=np.float32,
                            transform=data_transform,
                            )

    dataloader = torch.utils.data.DataLoader(train_set, **data_loader_params)
    if args.show:  # for debug purpose
        count = 0
        # infinite dataloader
        dataloader = infinite_dataloader(dataloader)
        img = next(dataloader)[0]
        print(img.shape)
        img = img[0].cpu().detach().numpy()
        img = (img*255).astype(np.uint8)

        window_name = "Press any key to continue; 'q'/Esc to quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 512, 512)
        cv2.imshow(window_name, img)
        while True:
            # print('\r Batch: {:3d} Timestep: {:3d}'.format(count//251 + 1, count %
            #                                                251), end="")
            k = cv2.waitKey(0)
            if k == 27 or k == ord("q"):  # press 'Esc' or 'q' to quit
                break
            else:
                count += 1
                img = next(dataloader)[0][0].cpu().detach().numpy()
                img = (img*255).astype(np.uint8)
                # img = de_normalize(img)
                cv2.imshow("Press any key to continue; 'q'/Esc to quit", img)
