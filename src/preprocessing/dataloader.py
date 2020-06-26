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
                 inference_mode=False,
                 dtype=np.float32,
                 ):
        """
        :param inference_mode (bool), return img only
        """
        super(LiTSDataset, self).__init__()
        self.filepath = filepath
        self.dataset = None
        self.transform = transform
        self.inference_mode = inference_mode
        self.dtype = dtype
        with h5py.File(self.filepath, "r") as file:
            self.num_samples = file['volumes'].shape[0]
            self.volume_start_index = file['volume_start_index'][:]
            self.spacing = file['spacing'][:]
            self.direction = file['direction'][:]
            self.offset = file['offset'][:]

    def __len__(self):
        # the total number of samples
        return self.num_samples

    def __getitem__(self, index):
        # Generates one sample of data
        if self.dataset is None:
            self.dataset = h5py.File(self.filepath, "r")

        img = self.dataset['volumes'][index]  # shape (512, 512)
        if not self.inference_mode:
            msk = self.dataset['segmentations'][index]  # .astype(self.dtype)  # shape (512, 512)
        # print(img.shape, msk.shape)
        # print(img.dtype, msk.dtype)
        if self.transform:
            # convert to
            img = img[..., None]  # new shape (512,512,1) for PIL Image transform
            img = self.transform(img)
            # msk = self.transform(msk) ## TODO TODO TODO

            # img = img[None, ...] # new shape (1, 512,512)
            # img = torch.from_numpy(img)  # (sharing storage without copying)
            if not self.inference_mode:
                msk = torch.from_numpy(msk)  # cf. torch.tensor: copy data
            # print(img.shape, msk.shape)

        if not self.inference_mode:
            return img, msk
        else:
            return img


def infinite_dataloader(dataloader):
    """ transform a dataloader to infinite-dataloader
    This is important for special model.

    usage: next(dataloader)
    """
    while True:
        for X in dataloader:
            yield X


def display_img_and_msk(img, msk):
    """Helper function for OpenCV display [OTC]
    img (N, 1, H, W) torch.tensor (float in range [0, 1) )
    msk (N, H, W) torch.tensor (int, in range [0, 1, 2])

    return img with liver (blue mask) and tumor (red mask)
    """
    img = img[0][0].cpu().detach().numpy()  # shape (H, W)
    msk = msk[0].cpu().detach().numpy()  # shape (H, W)
    # import ipdb
    # ipdb.set_trace()
    msk_liver = msk == 1
    msk_tumor = msk == 2

    img = img[..., None]  # shape (H, W, 1)
    msk_liver = msk_liver[..., None]
    msk_tumor = msk_tumor[..., None]
    img = np.repeat(img, 3, axis=-1)  # shape (H, W, 3)
    zeros = np.zeros_like(msk_liver, dtype=int)  # shape (H, W, 1)
    msk_liver = np.concatenate([msk_liver.astype(int), zeros, zeros], axis=-1)  # BGR blue (255, 0, 0)
    msk_tumor = np.concatenate([zeros, zeros, msk_tumor.astype(int)], axis=-1)  # BGR red (0, 0, 255)
    img = img*0.6 + msk_liver*0.2 + msk_tumor*0.2

    return img


if __name__ == "__main__":
    print("Start")
    # usage
    # python ./src/preprocessing/dataloader.py -f "./data/train_LiTS_db.h5" --shuffle
    parser = argparse.ArgumentParser(description="Visualize (static) dataloader results (for debug purpose)")
    parser.add_argument('-f', "--filepath", type=str, default=None, required=True,
                        help="dataset filepath.")
    parser.add_argument("--num-cpu", type=int, default=8,
                        help="Number of CPUs to use in parallel for dataloader.")
    parser.add_argument("--shuffle", action="store_true", default=False,
                        help="Shuffle the dataset")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--test-set", action="store_true", default=False,
                        help="i.e no segmentation")
    args = parser.parse_args()

    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_loader_params = {'batch_size': args.batch_size,
                          'shuffle': args.shuffle,
                          'num_workers': args.num_cpu,
                          #   'sampler': balanced_sampler,
                          'drop_last': True,
                          'pin_memory': False
                          }

    train_set = LiTSDataset(args.filepath,
                            transform=data_transform,
                            inference_mode=args.test_set,
                            dtype=np.float32,
                            )

    dataloader = torch.utils.data.DataLoader(train_set, **data_loader_params)
    count = 0
    # infinite dataloader
    dataloader = infinite_dataloader(dataloader)
    if args.test_set:
        img = next(dataloader)
        img = img[0][0].cpu().detach().numpy()

    else:
        img, msk = next(dataloader)
        img = display_img_and_msk(img, msk)

    print(img.shape)
    img = (img*255).astype(np.uint8)

    window_name = "Press any key to continue; 'q'/Esc to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 512, 512)
    cv2.imshow(window_name, img)
    while True:
        k = cv2.waitKey(0)
        if k == 27 or k == ord("q"):  # press 'Esc' or 'q' to quit
            break
        else:
            count += 1
            if args.test_set:
                img = next(dataloader)[0][0].cpu().detach().numpy()
            else:
                img, msk = next(dataloader)
                img = display_img_and_msk(img, msk)

            img = (img*255).astype(np.uint8)
            # img = de_normalize(img)
            cv2.imshow("Press any key to continue; 'q'/Esc to quit", img)
