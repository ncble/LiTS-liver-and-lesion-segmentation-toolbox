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


class GeoCompose(object):
    """Composes several transforms together.

    A work-around for the issue: (pytorch) 
        - #5059 https://github.com/pytorch/pytorch/issues/5059

    Warning there are two type of transformation
        1. geometry_transform: rotation, warpAffine, translation, etc
        2. pixelwise_transform: color jitter, brightness, saturation, ToTensor(), normalize() etc

    For (1.), we need both the image and the mask
    For (2.), only need the image

    This class is designed for the first type.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> GeoCompose([
        >>>     OpenCVRotation(...)
        >>>     
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, msk):
        for t in self.transforms:
            img, msk = t(img, msk)
        return img, msk

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class LiTSDataset(torch.utils.data.Dataset):
    r"""LiTS dataset loader for training (Pytorch natively supported)

    Work with hdf5 or npy/npz file format. 
    Issue: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643

    inference_mode: whether to return mask as well
    geometry_transform: perform on both image and the mask
    pixelwise_transform: only for image

    return img, msk (if not inference_mode)
    img.shape = (1, H, W)
    msk.shape = (H, W)

    Warning Issue (fixed): (numpy random doesn't work with pytorch dataloader under linux)
        - (pytorch) #5059: https://github.com/pytorch/pytorch/issues/5059

        - numpy doesn't properly handle RNG states when fork subprocesses. 
          It's numpy's issue with multiprocessing tracked at numpy/numpy#9248


    """

    def __init__(self, filepath,  # sample_indices,
                 geometry_transform=None,
                 pixelwise_transform=None,
                 inference_mode=False,
                 num_classes=3,
                 dtype=np.float32,
                 ):
        """
        :param inference_mode (bool), return img only
        """
        super(LiTSDataset, self).__init__()
        self.filepath = filepath
        self.dataset = None
        self.geometry_transform = geometry_transform
        self.pixelwise_transform = pixelwise_transform
        self.inference_mode = inference_mode
        self.num_classes = num_classes
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
            # this is a work-around of an issue
            self.dataset = h5py.File(self.filepath, "r")

        img = self.dataset['volumes'][index]  # shape (512, 512)
        if not self.inference_mode:
            msk = self.dataset['segmentations'][index]  # .astype(self.dtype)  # shape (512, 512)
        # print(img.shape, msk.shape)
        # print(img.dtype, msk.dtype)
        if self.geometry_transform:
            # img = img[..., None]  # new shape (512,512,1) for PIL Image transform

            # [IMIMIM] before geometry transform, need one-hot encoding for mask
            one_hot_msk = msk == np.arange(self.num_classes)[:, None, None]  # shape (num_classes, H, W)
            one_hot_msk = np.transpose(one_hot_msk.astype(np.uint8), (1, 2, 0))  # shape (H, W, num_classes)
            img, one_hot_msk = self.geometry_transform(img, one_hot_msk)
            msk = np.argmax(one_hot_msk, axis=-1)  # decode one-hot
        
        if self.pixelwise_transform:  # include ToTensor() and normalized()
            
            img = self.pixelwise_transform(img)
            # manually ToTensor()
            # img = torch.from_numpy(img)
            if not self.inference_mode:
                # manually ToTensor()
                # torch.from_numpy (sharing storage without copying)
                msk = torch.from_numpy(msk)  # cf. torch.tensor: copy data

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
    from data_augmentation import OpenCVRotation

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

    geo_transform = GeoCompose([
        OpenCVRotation((-45, 45)),
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    def worker_init_fn(worker_id):
        # WARNING spawn start method is used,
        # worker_init_fn cannot be an unpicklable object, e.g., a lambda function.
        # np.random.seed(np.random.get_state()[1][0] + worker_id)
        np.random.seed()
        # torch.initial_seed()
        # np.random.seed(int(torch.initial_seed()) % (2**32-1))

    data_loader_params = {'batch_size': args.batch_size,
                          'shuffle': args.shuffle,
                          'num_workers': args.num_cpu,
                          #   'sampler': balanced_sampler,
                          'drop_last': True,
                          'pin_memory': False,
                          'worker_init_fn': worker_init_fn,  # lambda _: np.random.seed() NO!!
                          }

    train_set = LiTSDataset(args.filepath,
                            geometry_transform=geo_transform,
                            pixelwise_transform=data_transform,
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
        elif k == ord("s"):
            # press 's' to save image
            N = len(glob.glob("./demo/*.jpg"))
            cv2.imwrite(f"./demo/{N+1}.jpg", img)
        else:
            count += 1
            if args.test_set:
                img = next(dataloader)[0][0].cpu().detach().numpy()
            else:
                img, msk = next(dataloader)
                img = display_img_and_msk(img, msk)

            img = (img*255).astype(np.uint8)

            cv2.imshow("Press any key to continue; 'q'/Esc to quit", img)
