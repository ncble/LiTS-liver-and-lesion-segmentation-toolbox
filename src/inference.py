"""
Copyright (C) 2020  Lu Lin
Warning: The code is released under the license GNU GPL 3.0

In order to follow the submission procedure of LiTS challenge, please refer to the instruction here:
    - https://github.com/PatrickChrist/LITS-CHALLENGE

"""
import os
import glob
import numpy as np
import argparse
import time

import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

# Official way to calculate the metric
from medpy import metric
# from surface import Surface
import nibabel as nb
# =====================

try:
    from models.unet import UNet
    from preprocessing.dataloader import LiTSDataset, infinite_dataloader
    from utils import printGreen, printRed, printBlue, printYellow
    from postprocessing.surface import Surface  # Official way to calculate the metric
except:
    pass


def get_scores(pred, label, vxlspacing):
    """
    pred: HxWxZ (x, y, z) of boolean
    label: HxWxZ (e.g. (512,512,75))
    vxlspacing: 3-tuple of float (spacing)

    """
    volscores = {}

    volscores['dice'] = metric.dc(pred, label)
    try:
        jaccard = metric.binary.jc(pred, label)
    except ZeroDivisionError:
        jaccard = 0.0
    volscores['jaccard'] = jaccard
    volscores['voe'] = 1. - volscores['jaccard']
    try:
        rvd = metric.ravd(label, pred)
    except:
        rvd = None
    volscores['rvd'] = rvd

    if np.count_nonzero(pred) == 0 or np.count_nonzero(label) == 0:
        volscores['assd'] = 0
        volscores['msd'] = 0
    else:
        evalsurf = Surface(pred, label, physical_voxel_spacing=vxlspacing,
                           mask_offset=[0., 0., 0.], reference_offset=[0., 0., 0.])
        volscores['assd'] = evalsurf.get_average_symmetric_surface_distance()

        volscores['msd'] = metric.hd(label, pred, voxelspacing=vxlspacing)

    return volscores


def inference():
    """Support two mode: evaluation (on valid set) or inference mode (on test-set for submission)

    """
    parser = argparse.ArgumentParser(description="Inference mode")
    parser.add_argument('-testf', "--test-filepath", type=str, default=None, required=True,
                        help="testing dataset filepath.")
    parser.add_argument("-eval", "--evaluate", action="store_true", default=False,
                        help="Evaluation mode")
    parser.add_argument("--load-weights", type=str, default=None,
                        help="Load pretrained weights, torch state_dict() (filepath, default: None)")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Load pretrained model, entire model (filepath, default: None)")

    parser.add_argument("--save2dir", type=str, default=None,
                        help="save the prediction labels to the directory (default: None)")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")

    parser.add_argument("--num-cpu", type=int, default=10,
                        help="Number of CPUs to use in parallel for dataloader.")
    parser.add_argument('--cuda', type=int, default=0,
                        help='CUDA visible device (use CPU if -1, default: 0)')
    args = parser.parse_args()

    printYellow("="*10 + " Inference mode. "+"="*10)
    if args.save2dir:
        os.makedirs(args.save2dir, exist_ok=True)

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available()
                          and (args.cuda >= 0) else "cpu")

    transform_normalize = transforms.Normalize(mean=[0.5],
                                               std=[0.5])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transform_normalize
    ])

    data_loader_params = {'batch_size': args.batch_size,
                          'shuffle': False,
                          'num_workers': args.num_cpu,
                          'drop_last': False,
                          'pin_memory': False
                          }

    test_set = LiTSDataset(args.test_filepath,
                           dtype=np.float32,
                           pixelwise_transform=data_transform,
                           inference_mode=(not args.evaluate),
                           )
    dataloader_test = torch.utils.data.DataLoader(test_set, **data_loader_params)
    # =================== Build model ===================
    if args.load_weights:
        model = UNet(in_ch=1,
                     out_ch=3,  # there are 3 classes: 0: background, 1: liver, 2: tumor
                     depth=4,
                     start_ch=64,
                     inc_rate=2,
                     kernel_size=3,
                     padding=True,
                     batch_norm=True,
                     spec_norm=False,
                     dropout=0.5,
                     up_mode='upconv',
                     include_top=True,
                     include_last_act=False,
                     )
        model.load_state_dict(torch.load(args.load_weights))
        printYellow("Successfully loaded pretrained weights.")
    elif args.load_model:
        # load entire model
        model = torch.load(args.load_model)
        printYellow("Successfully loaded pretrained model.")
    model.eval()
    model.to(device)

    # n_batch_per_epoch = len(dataloader_test)

    sigmoid_act = torch.nn.Sigmoid()
    st = time.time()

    volume_start_index = test_set.volume_start_index
    spacing = test_set.spacing
    direction = test_set.direction  # use it for the submission

    msk_pred_buffer = []
    if args.evaluate:
        msk_gt_buffer = []

    for data_batch in tqdm(dataloader_test):
        # import ipdb
        # ipdb.set_trace()
        if args.evaluate:
            img, msk_gt = data_batch
            msk_gt_buffer.append(msk_gt.cpu().detach().numpy())
        else:
            img = data_batch
        img = img.to(device)
        with torch.no_grad():
            msk_pred = model(img)  # shape (N, 3, H, W)
            msk_pred = sigmoid_act(msk_pred)
        msk_pred_buffer.append(msk_pred.cpu().detach().numpy())

        # TODO TODO: remember to correct 'direction' and np.transpose before the submission !!!
    msk_pred_buffer = np.vstack(msk_pred_buffer)  # shape (N, 3, H, W)
    if args.evaluate:
        msk_gt_buffer = np.vstack(msk_gt_buffer)

        results = []
    for vol_ind, vol_start_ind in enumerate(volume_start_index):
        if vol_ind == len(volume_start_index) - 1:
            volume_msk = msk_pred_buffer[vol_start_ind:]  # shape (N, 3, H, W)
            if args.evaluate:
                volume_msk_gt = msk_gt_buffer[vol_start_ind:]
        else:
            vol_end_ind = volume_start_index[vol_ind+1]
            volume_msk = msk_pred_buffer[vol_start_ind:vol_end_ind]  # shape (N, 3, H, W)

            if args.evaluate:
                volume_msk_gt = msk_gt_buffer[vol_start_ind:vol_end_ind]
        # liver
        liver_scores = get_scores(volume_msk[:, 1] >= 0.5, volume_msk_gt >= 1, spacing[vol_ind])
        # tumor
        lesion_scores = get_scores(volume_msk[:, 2] >= 0.5, volume_msk_gt == 2, spacing[vol_ind])
        print("Liver dice", liver_scores['dice'], "Lesion dice", lesion_scores['dice'])
        results.append([vol_ind, liver_scores, lesion_scores])

        # ===========================
        if args.save2dir:
            outpath = os.path.join(args.save2dir, "results.csv")
        # ======== code from official metric ========
        # create line for csv file
        outstr = str(vol_ind) + ','
        for l in [liver_scores, lesion_scores]:
            for k, v in l.items():
                outstr += str(v) + ','
                outstr += '\n'
        # create header for csv file if necessary
        if not os.path.isfile(outpath):
            headerstr = 'Volume,'
            for k, v in liver_scores.items():
                headerstr += 'Liver_' + k + ','
            for k, v in liver_scores.items():
                headerstr += 'Lesion_' + k + ','
            headerstr += '\n'
            outstr = headerstr + outstr
        # write to file
        f = open(outpath, 'a+')
        f.write(outstr)
        f.close()
        # ===========================
    # import ipdb; ipdb.set_trace()
    printGreen(f"Total elapsed time: {time.time()-st}")
    return results


if __name__ == "__main__":
    print("Start")
    # usage (evaluation mode)
    # python ./src/inference.py -eval --batch-size 32 --num-cpu 32 -testf ./data/valid_LiTS_db_224.h5 --load-weights "./weights/Exp_002/model_weights.pth"
    #
    inference()
