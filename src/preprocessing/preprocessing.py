"""
Copyright (C) 2020  Lu Lin
Warning: The code is released under the license GNU GPL 3.0


"""
import os
import argparse
from medpy.io import load as med_load
from medpy.io import save as med_save
import numpy as np
from tqdm import tqdm
import h5py
import cv2  # resize image


def get_filelist(dirpath):
    """[OTC]
    """
    filelist = []
    for root, subdir, name_list in os.walk(dirpath):
        for filename in name_list:
            if "volume" in filename:
                filelist.append(os.path.join(root, filename))
    return filelist


def parse_filepath(filepath):
    """[OTC]

    """
    dirpath, filename = os.path.split(filepath)
    index = filename.split(".")[0].split("-")[-1]
    return dirpath, filename, index


def hu_clipping(img, hu_max=250, hu_min=-200):
    """
    Clip HU value and normalize pixel value into [0, 1)

    """
    img[img > hu_max] = hu_max
    img[img < hu_min] = hu_min
    img = (img-hu_min)/(hu_max-hu_min)
    return img


def adapt_hist(clip_limit=2.0, tile_grid=(8, 8)):
    """[OPT]
    TODO 
    adaptive histogram transform
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def apply_transform(img):
        # cv2 accept only dtype=uint8
        lower = np.min(img)
        img = img - lower
        upper = np.max(img)
        img /= upper
        img *= 255.
        img = img.astype(np.uint8)
        cl1 = clahe.apply(img)
        cl1 = (cl1/255.)
        return cl1
    return apply_transform


def extract_one_volume(filepath,
                       fix_direction=True,
                       rotation_90=True,
                       test_set=False):
    """
    read one *.nii file.

    :param fix_direction (bool), flip volume according the 'direction' (3*3 matrix)
    :param rotation_90 (bool), rotate volume 90 degree clockwise
    :param test_set (bool), if true, no mask available


    return 
        - img with shape = (x, y, z)
        - msk with shape = (x, y, z) values in [0, 1, 2], return None if test_set
        - spacing (array), shape (3,), # how many mm represents a pixel 
        - direction (array), shape (3,3), 
        - offset (array), shape (3,), 
        - liver_ratio (float) return None if test_set
    """
    img, img_header = med_load(filepath)
    dirpath, filename, index = parse_filepath(filepath)
    if not test_set:
        msk_filepath = os.path.join(dirpath, "segmentation-{}.nii".format(index))
        msk, _ = med_load(msk_filepath)

    offset = img_header.get_offset()  # tuple of 3
    direction = img_header.get_direction()  # shape = (3, 3) array
    spacing = img_header.get_voxel_spacing()  # tuple of 3
    if not test_set:
        liver_ratio = np.sum(msk >= 1)/np.product(img.shape)
    else:
        liver_ratio = None

    if fix_direction:
        # correct the direction
        if direction[0, 0] == -1:
            img = img[::-1, ...]
            if not test_set:
                msk = msk[::-1, ...]
        if direction[1, 1] == -1:
            img = img[:, ::-1, ...]
            if not test_set:
                msk = msk[:, ::-1, ...]
        if direction[2, 2] == -1:
            img = img[..., ::-1]
            if not test_set:
                msk = msk[..., ::-1]

    if rotation_90:
        # rotation 90 degree
        img = np.transpose(img, axes=(1, 0, 2))
        if not test_set:
            msk = np.transpose(msk, axes=(1, 0, 2))

    if test_set:
        msk = None

    return img, msk, np.array(spacing), direction, np.array(offset), liver_ratio


def find_liver_bbox(msk):
    """[OPT]
    msk.shape = (x_slices, y_slices, z_slices)
    (x, y, z)

    mask description:
        0: background
        1: liver (not include tumor)
        2: tumor 


    """
    len_x, len_y, len_z = msk.shape
    index0 = np.sum(msk, axis=(1, 2)) > 0
    index1 = np.sum(msk, axis=(0, 2)) > 0
    index2 = np.sum(msk, axis=(0, 1)) > 0
    bbmin = np.zeros(3, dtype=int)
    bbmax = np.zeros(3, dtype=int)
    bbmin[0] = np.argmax(index0)
    bbmin[1] = np.argmax(index1)
    bbmin[2] = np.argmax(index2)
    bbmax[0] = len_x - np.argmax(index0[::-1])
    bbmax[1] = len_y - np.argmax(index1[::-1])
    bbmax[2] = len_z - np.argmax(index2[::-1])

    return bbmin, bbmax


def resize_volume(volume, img_shape=None):
    """resize one volume (z, x, y) to (z, img_shape[0], img_shape[1])

    :param img_shape (2-tuple or None)

    Warning: 
        - assume volume's voxel in range [0, 1], 
        - resizing might affect the performance due to float-int rounding.

    return resized volume, shape = (z, img_shape[0], img_shape[1])

    Warning: this is a 2D per-slice-resize algorithm
    """
    if img_shape is None:
        # do noting
        return volume
    else:
        volume = (volume*255).astype(np.uint8)
        vol_buffer = []
        for z_slice in volume:
            new_z = cv2.resize(z_slice, (img_shape[1], img_shape[0]))
            # new_z.shape == img_shape (e.g (224, 224))
            vol_buffer.append(new_z)
        resized_vol = np.array(vol_buffer)
        resized_vol = resized_vol.astype("float32")  # [OPT] could be optimized
        return resized_vol/255.


def resize_mask(mask_vol, img_shape=None, num_classes=3):
    """resize one mask-volume (z, H, W) to (z, img_shape[0], img_shape[1])
    [OTC][OPT] 

    :param img_shape (2-tuple or None)

    First, we transform mask to one-hot encoding:
        A simple solution is to:
            1. flatten() (or reshape())
            2. apply the usual one-hot encoding function
            3. reshape()

        Here, we do much better ! (use fancy python slicing trick): 
            (only one step)

            - one_hot = mask == np.arange(num_classes)[:, None, None]

            mask.shape = (H, W)
            one_hot.shape = (num_classes, H, W)

            [OPT]: analyze the time cost


    return resized mask volume, shape = (z, img_shape[0], img_shape[1])

    Warning: this is a 2D per-slice-resize algorithm
    """
    if img_shape is None:
        # do noting
        return mask_vol
    else:
        vol_buffer = []
        for mask in mask_vol: # [OPT] avoid for-loop
            # mask.shape = (H, W)
            one_hot = mask == np.arange(num_classes)[:, None, None]
            # one_hot.shape = (num_classes, H, W)
            # [OPT] avoid transpose ?
            one_hot = np.transpose(one_hot, axes=(1, 2, 0))  # (H, W, num_classes=3)

            # [OTC]: only work here, since num_classes=3, otherwise do it per channels
            assert num_classes == 3, "Modify the code: resize slice per channels (classes)"
            resized_msk = cv2.resize(one_hot.astype(
                np.uint8), (img_shape[1], img_shape[0]), interpolation=cv2.INTER_AREA)
            # new.shape = (H, W, num_classes=3)
            # TODO need better post-processing for resized_msk
            vol_buffer.append(resized_msk)

        vol_buffer = np.array(vol_buffer)  # shape (z, H, W, 3)
        # decode one-hot
        # TODO also need better post-processing here: classes priority
        vol_buffer = np.argmax(vol_buffer, axis=-1)  # shape (z, H, W)

        return vol_buffer.astype("int")  # dtype uint8 to int


def preprocessing(filelist,
                  save2file,
                  hu_max=250,
                  hu_min=-200,
                  fix_direction=True,
                  rotation_90=True,
                  fix_spacing=False,
                  file_format="h5",
                  test_set=False,
                  img_shape=None,
                  ):
    """Preprocessing: collect and save all *.nii file into a .h5 file.

    Perform the following operations (in order):
    - fix HU shift (NotImplemented, TODO): using linear/adaptive histogram equalization (adapt_hist)
    - fix direction (3, 3) matrix.
    - rotate 90 degree
    - HU clipping: (hu_min, hu_max)
    - fix spacing factor (NotImplemented, TODO)
    - transpose (x, y, z) to (z, x, y)
    - remove irrelevant slices (along z-axis): only keep 20% of slices without labels
    - resize volume/segmentation if needed (i.e. img_shape is not None)


    :param hu_max/hu_min: HU-values clipping range
    :param img_shape (2-tuple or None)
    :param file_format (str): save data to a file. (choices: npz, h5)



    """
    assert file_format in ['npz', 'h5'], "file format should be in ['npz', 'h5']"
    save2dir = os.path.split(save2file)[0]
    os.makedirs(save2dir, exist_ok=True)

    if file_format == 'h5':
        h5_file = h5py.File(save2file, "a")
    elif file_format == 'npz':
        img_buffer = []
        msk_buffer = []
        spacing_buffer = []
        offset_buffer = []
        liver_ratio_buffer = []
        bbox_buffer = []

    volume_start_index = [0]  # record the start_index of each volume

    for ind, filepath in tqdm(enumerate(filelist)):
        img, msk, spacing, direction, offset, liver_ratio = extract_one_volume(
            filepath, fix_direction=fix_direction, rotation_90=rotation_90, test_set=test_set)
        img = hu_clipping(img, hu_max=hu_max, hu_min=hu_min).astype('float32')
        if not test_set:
            bbmin, bbmax = find_liver_bbox(msk)
        # Several strategies:
        # If the memory is small: dump each volume to a file, then use 'generator' to load them
        # else: save the whole dataset into a file, load them into the memory.
        # A better way is to use h5py file format. (or npy, npz format)

        # (x, y, z) to (z, x, y)
        img = np.transpose(img, axes=(2, 0, 1))
        if not test_set:
            msk = np.transpose(msk, axes=(2, 0, 1))

        if not test_set:
            # [IMIMIM] TODO
            # remove/reduce slices without liver: extend +/- 10% of bbox
            num_z_slices = len(img)
            bbmin_z, bbmax_z = bbmin[-1], bbmax[-1]
            extend_z = int((bbmax_z - bbmin_z)*0.1)
            img = img[max(0, bbmin_z-extend_z):min(num_z_slices-1, bbmax_z+extend_z)]
            msk = msk[max(0, bbmin_z-extend_z):min(num_z_slices-1, bbmax_z+extend_z)]

        # resize the volume here if (img_shape is not None)
        img = resize_volume(img, img_shape=img_shape)
        if not test_set:
            # TODO: make sure the quality of labels !!!
            msk = resize_mask(msk, img_shape=img_shape)

        # import ipdb; ipdb.set_trace()

        if file_format == 'h5':
            shape = img.shape
            # TODO encounter some issue: h5 file too large...
            if ind == 0:
                h5_file.create_dataset('volumes', data=img,
                                       maxshape=(None, shape[1], shape[2]), chunks=True, dtype='float32')
                h5_file.create_dataset('spacing', data=spacing[None, :],
                                       maxshape=(None, 3), dtype='float32')
                h5_file.create_dataset('offset', data=offset[None, :],
                                       maxshape=(None, 3), dtype='float32')
                h5_file.create_dataset('direction', data=np.diag(direction)[None, :],
                                       maxshape=(None, 3), dtype='int')
                if not test_set:
                    h5_file.create_dataset('segmentations', data=msk,
                                           maxshape=(None, shape[1], shape[2]), chunks=True, dtype='int')
                    h5_file.create_dataset('liver_ratio', data=np.array([liver_ratio])[None, :],
                                           maxshape=(None, 1), dtype='float32')
                    h5_file.create_dataset('bbox', data=np.hstack([bbmin, bbmax])[None, :],
                                           maxshape=(None, 6), dtype='float32')
                h5_file.create_dataset('volume_start_index', data=np.array([0]),
                                       maxshape=(None,), dtype='int')
            else:
                h5_file['volumes'].resize(h5_file['volumes'].shape[0] + img.shape[0], axis=0)
                h5_file['volumes'][-img.shape[0]:] = img
                h5_file['spacing'].resize(h5_file['spacing'].shape[0] + 1, axis=0)
                h5_file['spacing'][-1:] = spacing[None, :]
                h5_file['offset'].resize(h5_file['offset'].shape[0] + 1, axis=0)
                h5_file['offset'][-1:] = offset[None, :]
                h5_file['direction'].resize(h5_file['direction'].shape[0] + 1, axis=0)
                h5_file['direction'][-1:] = np.diag(direction)[None, :]
                if not test_set:
                    h5_file['segmentations'].resize(h5_file['segmentations'].shape[0] + msk.shape[0], axis=0)
                    h5_file['segmentations'][-msk.shape[0]:] = msk
                    h5_file['liver_ratio'].resize(h5_file['liver_ratio'].shape[0] + 1, axis=0)
                    h5_file['liver_ratio'][-1:] = np.array([liver_ratio])[None, :]
                    h5_file['bbox'].resize(h5_file['bbox'].shape[0] + 1, axis=0)
                    h5_file['bbox'][-1:] = np.hstack([bbmin, bbmax])[None, :]
                h5_file['volume_start_index'].resize(h5_file['volume_start_index'].shape[0] + 1, axis=0)
                h5_file['volume_start_index'][-1:] = np.array([img.shape[0]])

        elif file_format == 'npz':
            img_buffer.append(img)
            spacing_buffer.append(spacing)
            offset_buffer.append(offset)
            if not test_set:
                msk_buffer.append(msk)
                liver_ratio_buffer.append(liver_ratio)
                bbox_buffer.append(np.hstack([bbmin, bbmax]))
            volume_start_index.append(img.shape[0])  # z-axis slices

    if file_format == "h5":
        h5_file['volume_start_index'][:] = np.cumsum(h5_file['volume_start_index'])
        h5_file.close()

    elif file_format in ['npz']:  # 'npy',
        dataset = {'volumes': np.vstack(img_buffer),
                   'segmentations': np.vstack(msk_buffer),
                   'spacing': np.vstack(spacing_buffer),
                   'offset': np.vstack(offset_buffer),
                   'liver_ratio': np.vstack(liver_ratio_buffer),
                   'bbox': np.vstack(bbox_buffer),
                   'volume_start_index': np.cumsum(np.array(volume_start_index[:-1]))}  # remove the last one
        np.savez(save2file, **dataset)


def main():
    parser = argparse.ArgumentParser(description="Dataset preprocessing")
    parser.add_argument('-dir', "--dirpath", type=str, default=None, required=True,
                        help="dataset dirpath (of *.nii).")
    parser.add_argument("--save2file", type=str, default=None,
                        help="save data to a file (filepath will have prefix: train_ & valid_).")
    parser.add_argument("--valid-split", type=float, default=0.2,
                        help="validation set split rate [0, 1). (default to 0.2)")
    parser.add_argument("--format", type=str, default='h5', choices=['h5', 'npz'],
                        help="save dataset to file format (default: 'h5'). choices=['h5', 'npz']")
    parser.add_argument('--img-shape', type=str, default=None,
                        help='Image shape 2-tuple (default None, i.e "(512, 512)" )')
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--test-set", action="store_true", default=False,
                        help="preprocessing test-set (valid_split will be set to 0)")

    args = parser.parse_args()

    if args.img_shape:
        # convert str to tuple
        args.img_shape = tuple(map(int, args.img_shape.strip()[1:-1].split(",")))

    filelist = get_filelist(args.dirpath)
    # sort list according to the volume index [OTC]
    filelist = sorted(filelist, key=lambda x: int(os.path.split(x)[-1].split(".")[0].split("-")[-1]))
    # import ipdb; ipdb.set_trace()
    save2dir, filename = os.path.split(args.save2file)
    os.makedirs(save2dir, exist_ok=True)

    if args.test_set:
        args.valid_split = 0
        train_filepath = os.path.join(save2dir, "test_{}".format(filename))
        train_filelist = filelist
    else:
        train_filepath = os.path.join(save2dir, "train_{}".format(filename))

        train_filelist = filelist[:-int(len(filelist)*args.valid_split)]

    preprocessing(train_filelist,
                  train_filepath,
                  hu_max=250,
                  hu_min=-200,
                  fix_direction=True,
                  rotation_90=True,
                  fix_spacing=False,
                  file_format=args.format,
                  test_set=args.test_set,
                  img_shape=args.img_shape,
                  )
    if args.valid_split > 0:
        valid_filepath = os.path.join(save2dir, "valid_{}".format(filename))
        valid_filelist = filelist[-int(len(filelist)*args.valid_split):]

        preprocessing(valid_filelist,
                      valid_filepath,
                      hu_max=250,
                      hu_min=-200,
                      fix_direction=True,
                      rotation_90=True,
                      fix_spacing=False,
                      file_format=args.format,
                      test_set=args.test_set,
                      img_shape=args.img_shape,
                      )

    return


if __name__ == "__main__":
    print("Start")
    # filelist = []
    # preprocessing(filelist, "./data/LiTS_db.h5", file_format="h5")
    # usage: python ./src/preprocessing/preprocessing.py -dir "./data" --save2file "./data/LiTS_db.h5"
    # usage (resize): python ./src/preprocessing/preprocessing.py -dir "./data" --save2file "./data/LiTS_db_224.h5" --img-shape "(224,224)"
    # usage (test set): python ./src/preprocessing/preprocessing.py -dir "./data" --save2file "./data/LiTS_db.h5" --test-set
    main()
