"""
Copyright (C) 2020  Lu Lin
Warning: The code is released under the license GNU GPL 3.0


"""
import os
from medpy.io import load as med_load
from medpy.io import save as med_save
import numpy as np
from tqdm import tqdm
import h5py
# import cv2  # unnecessary


def get_filelist(dirpath):
    """[OTC]
    """
    filelist = []
    for root, subdir, name_list in os.walk(dirpath):
        for filename in name_list:
            if filename.startswith("volume"):
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
    """
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


def extract_one_volume(filepath, fix_direction=True, rotation_90=True):
    """
    read one *.nii file.

    return 
        - img with shape = (x, y, z)
        - msk
        - spacing, 
        - direction, 
        - offset, 
        - liver_ratio
    """
    img, img_header = med_load(filepath)
    dirpath, filename, index = parse_filepath(filepath)
    msk_filepath = os.path.join(dirpath, f"segmentation-{index}.nii")
    msk, _ = med_load(msk_filepath)

    offset = img_header.get_offset()  # tuple of 3
    direction = img_header.get_direction()  # shape = (3, 3) array
    spacing = img_header.get_voxel_spacing()  # tuple of 3
    liver_ratio = np.sum(msk >= 1)/np.product(img.shape)

    if fix_direction:
        # correct the direction
        if direction[0, 0] == -1:
            img = img[::-1, ...]
            msk = msk[::-1, ...]
        if direction[1, 1] == -1:
            img = img[:, ::-1, ...]
            msk = msk[:, ::-1, ...]
        if direction[2, 2] == -1:
            img = img[..., ::-1]
            msk = msk[..., ::-1]

    if rotation_90:
        # rotation 90 degree
        img = np.transpose(img, axes=(1, 0, 2))
        msk = np.transpose(msk, axes=(1, 0, 2))

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


def preprocessing(data_dir,
                  save2file,
                  hu_max=250,
                  hu_min=-200,
                  fix_direction=True,
                  rotation_90=True,
                  fix_spacing=False,
                  file_format="npz"):
    """
    Perform the following operations:

    - fix HU shift (?, TODO): using linear/adaptive histogram equalization (adapt_hist)
    - HU clipping: (hu_min, hu_max)
    - fix direction (3, 3) matrix.
    - rotate 90 degree
    - fix spacing factor (?, TODO)

    - file_format: save data to a file. (Options: npz, h5)
    """
    assert file_format in ['npz', 'h5'], "file format should be in ['npz', 'h5']"
    save2dir = os.path.split(save2file)[0]
    os.makedirs(save2dir, exist_ok=True)
    filelist = get_filelist(data_dir)

    # sort list according to the volume index [OTC]
    filelist = sorted(filelist, key=lambda x: int(os.path.split(x)[-1].split(".")[0].split("-")[-1]))

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
            filepath, fix_direction=fix_direction, rotation_90=rotation_90)
        img = hu_clipping(img, hu_max=hu_max, hu_min=hu_min).astype('float32')
        bbmin, bbmax = find_liver_bbox(msk)
        # Several strategies:
        # If the memory is small: dump each volume to a file, then use 'generator' to load them
        # else: save the whole dataset into a file, load them into the memory.
        # A better way is to use h5py file format. (or npy, npz format)

        # (x, y, z) to (z, x, y)
        img = np.transpose(img, axes=(2, 0, 1))
        msk = np.transpose(msk, axes=(2, 0, 1))
        shape = img.shape

        # dset_vol = h5_file.create_dataset(f'volume-{ind}', data=img,
        #                                maxshape=(None, shape[1], shape[2]), chunks=True, dtype='float32')
        # dset_vol.attrs['spacing'] = spacing
        # dset_vol.attrs['direction'] = direction
        # dset_vol.attrs['offset'] = offset
        # dset_vol.attrs['liver_ratio'] = liver_ratio
        # dset_seg = h5_file.create_dataset(f'segment-{ind}', data=msk,
        #                                maxshape=(None, shape[1], shape[2]), chunks=True, dtype='int')
        # dset_seg.attrs['bbmin'] = bbmin
        # dset_seg.attrs['bbmax'] = bbmax
        if file_format == 'h5':
            # TODO encounter some issue: h5 file too large...
            if ind == 0:
                h5_file.create_dataset('volumes', data=img,
                                       maxshape=(None, shape[1], shape[2]), chunks=True, dtype='float32')
                h5_file.create_dataset('segmentations', data=msk,
                                       maxshape=(None, shape[1], shape[2]), chunks=True, dtype='int')
                h5_file.create_dataset('spacing', data=spacing[None, :],
                                       maxshape=(None, 3), dtype='float32')
                h5_file.create_dataset('offset', data=offset[None, :],
                                       maxshape=(None, 3), dtype='float32')
                h5_file.create_dataset('liver_ratio', data=np.array([liver_ratio])[None, :],
                                       maxshape=(None, 1), dtype='float32')
                h5_file.create_dataset('bbox', data=np.hstack([bbmin, bbmax])[None, :],
                                       maxshape=(None, 6), dtype='float32')
                h5_file.create_dataset('volume_start_index', data=np.array([0]),
                                       maxshape=(None,), dtype='int')
            else:
                h5_file['volumes'].resize(h5_file['volumes'].shape[0] + img.shape[0], axis=0)
                h5_file['volumes'][-img.shape[0]:] = img
                h5_file['segmentations'].resize(h5_file['segmentations'].shape[0] + msk.shape[0], axis=0)
                h5_file['segmentations'][-msk.shape[0]:] = msk
                h5_file['spacing'].resize(h5_file['spacing'].shape[0] + 1, axis=0)
                h5_file['spacing'][-1:] = spacing[None, :]
                h5_file['offset'].resize(h5_file['offset'].shape[0] + 1, axis=0)
                h5_file['offset'][-1:] = offset[None, :]
                h5_file['liver_ratio'].resize(h5_file['liver_ratio'].shape[0] + 1, axis=0)
                h5_file['liver_ratio'][-1:] = np.array([liver_ratio])[None, :]
                h5_file['bbox'].resize(h5_file['bbox'].shape[0] + 1, axis=0)
                h5_file['bbox'][-1:] = np.hstack([bbmin, bbmax])[None, :]
                h5_file['volume_start_index'].resize(h5_file['volume_start_index'].shape[0] + 1, axis=0)
                h5_file['volume_start_index'][-1:] = np.array([img.shape[0]])

        elif file_format == 'npz':
            img_buffer.append(img)
            msk_buffer.append(msk)
            spacing_buffer.append(spacing)
            offset_buffer.append(offset)
            liver_ratio_buffer.append(liver_ratio)
            bbox_buffer.append(np.hstack([bbmin, bbmax]))
            volume_start_index.append(img.shape[0])  # z-axis slices

        # dset_vol.attrs['spacing'] = spacing
        # dset_vol.attrs['direction'] = direction
        # dset_vol.attrs['offset'] = offset
        # dset_vol.attrs['liver_ratio'] = liver_ratio
        # dset_seg.attrs['bbmin'] = bbmin
        # dset_seg.attrs['bbmax'] = bbmax
        ## dset.resize(dset.shape[0]+10**4, axis=0)
    if file_format == "h5":
        h5_file['volume_start_index'][:] = np.cumsum(h5_file['volume_start_index'])
        h5_file.close()

    elif file_format in ['npy', 'npz']:
        dataset = {'volumes': np.vstack(img_buffer),
                   'segmentations': np.vstack(msk_buffer),
                   'spacing': np.vstack(spacing_buffer),
                   'offset': np.vstack(offset_buffer),
                   'liver_ratio': np.vstack(liver_ratio_buffer),
                   'bbox': np.vstack(bbox_buffer),
                   'volume_start_index': np.cumsum(np.array(volume_start_index[:-1]))}  # remove the last one
        np.savez(save2file, **dataset)


if __name__ == "__main__":
    print("Start")
    preprocessing('./debug/batch', "./debug/LiTS_db8.npz")
