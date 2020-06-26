"""
Copyright (C) 2020  Lu Lin
Warning: The code is released under the license GNU GPL 3.0


"""
import os
import glob
import argparse
import time
import numpy as np

import cv2
import torch
from torchvision import transforms

try:
    from models.unet import UNet
    from models.losses import CrossEntropyLoss, DiceLoss, FocalLoss, IoULoss
    from preprocessing.dataloader import LiTSDataset, infinite_dataloader
    from utils import printGreen, printRed, printBlue, printYellow
except:
    # TODO: should support both relative/absolute import
    pass


def main():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument('-trainf', "--train-filepath", type=str, default=None, required=True,
                        help="training dataset filepath.")
    parser.add_argument('-validf', "--val-filepath", type=str, default=None,
                        help="validation dataset filepath.")
    parser.add_argument("--shuffle", action="store_true", default=False,
                        help="Shuffle the dataset")
    parser.add_argument("--load-weights", type=str, default=None,
                        help="load pretrained weights")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")

    parser.add_argument('--img-shape', type=str, default="(1,512,512)",
                        help='Image shape (default "(1,512,512)"')

    parser.add_argument("--num-cpu", type=int, default=10,
                        help="Number of CPUs to use in parallel for dataloader.")
    parser.add_argument('--cuda', type=int, default=0,
                        help='CUDA visible device (use CPU if -1, default: 0)')
    parser.add_argument('--cuda-non-deterministic', action='store_true', default=False,
                        help="sets flags for non-determinism when using CUDA (potentially fast)")

    parser.add_argument('-lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed (numpy and cuda if GPU is used.).')

    parser.add_argument('--log-dir', type=str, default=None,
                        help='Save the results/model weights/logs under the directory.')

    args = parser.parse_args()

    # TODO: support image reshape
    img_shape = tuple(map(int, args.img_shape.strip()[1:-1].split(",")))

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        best_model_path = os.path.join(args.log_dir, "model_weights.pth")
    else:
        best_model_path = None

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda >= 0:
            if args.cuda_non_deterministic:
                printBlue("Warning: using CUDA non-deterministc. Could be faster but results might not be reproducible.")
            else:
                printBlue("Using CUDA deterministc. Use --cuda-non-deterministic might accelerate the training a bit.")
            # Make CuDNN Determinist
            torch.backends.cudnn.deterministic = not args.cuda_non_deterministic

            # torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    # TODO [OPT] enable multi-GPUs ?
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available()
                          and (args.cuda >= 0) else "cpu")

    # ================= Build dataloader =================
    # DataLoader
    # transform_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                            std=[0.5, 0.5, 0.5])
    transform_normalize = transforms.Normalize(mean=[0.5],
                                               std=[0.5])

    # Warning: DO NOT use geometry transform (do it in the dataloader instead)
    data_transform = transforms.Compose([
        # transforms.ToPILImage(mode='F'), # mode='F' for one-channel image
        # transforms.Resize((256, 256)) # NO
        # transforms.RandomResizedCrop(256), # NO
        # transforms.RandomHorizontalFlip(p=0.5), # NO
        # WARNING, ISSUE: transforms.ColorJitter doesn't work with ToPILImage(mode='F').
        # Need custom data augmentation functions: TODO
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # TODO TODO
        transforms.ToTensor(),  # already done in the dataloader
        transform_normalize
    ])

    data_loader_params = {'batch_size': args.batch_size,
                          'shuffle': args.shuffle,
                          'num_workers': args.num_cpu,
                          #   'sampler': balanced_sampler,
                          'drop_last': True,
                          'pin_memory': False
                          }
    train_set = LiTSDataset(args.train_filepath,
                            dtype=np.float32,
                            transform=data_transform,
                            )
    valid_set = LiTSDataset(args.val_filepath,
                            dtype=np.float32,
                            transform=data_transform,
                            )

    dataloader_train = torch.utils.data.DataLoader(train_set, **data_loader_params)
    dataloader_valid = torch.utils.data.DataLoader(valid_set, **data_loader_params)
    # =================== Build model ===================
    model = UNet(in_ch=1,
                 out_ch=3,  # there are 3 classes: 0: background, 1: liver, 2: tumor
                 depth=4,
                 start_ch=64,
                 inc_rate=2,
                 padding=True,
                 batch_norm=True,
                 spec_norm=False,
                 dropout=0.5,
                 up_mode='upconv',
                 include_top=True,
                 include_last_act=False,
                 )
    if args.load_weights:
        printYellow(f"Loading pretrained weights from: {args.load_weights}...")
        model.load_state_dict(torch.load(args.load_weights))
        printYellow("+ Done.")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95))  # TODO
    best_valid_loss = float('inf')

    for epoch in range(args.epochs):
        for valid_mode, dataloader in enumerate([dataloader_train, dataloader_valid]):
            n_batch_per_epoch = len(dataloader)
            if args.debug:
                n_batch_per_epoch = 1

            # infinite dataloader allows several update per iteration (for special models e.g. GAN)
            dataloader = infinite_dataloader(dataloader)
            if valid_mode:
                printYellow("Switch to validation mode.")
                model.eval()
                prev_grad_mode = torch.is_grad_enabled()
                torch.set_grad_enabled(False)
            else:
                model.train()

            st = time.time()
            cum_loss = 0
            for iter_ind in range(n_batch_per_epoch):
                supplement_logs = ""
                # reset cumulated losses at the begining of each batch
                # loss_manager.reset_losses() # TODO TODO
                optimizer.zero_grad()

                img, msk = next(dataloader)
                img, msk = img.to(device), msk.to(device)

                # TODO this is ugly: convert dtype and convert the shape from (N, 1, 512, 512) to (N, 512, 512)
                msk = msk.to(torch.long).squeeze(1)

                msk_pred = model(img)  # shape (N, 3, 512, 512)

                # label_weights is determined according the liver_ratio & tumor_ratio
                # loss = CrossEntropyLoss(msk_pred, msk, label_weights=[1., 10., 100.], device=device)
                # loss = DiceLoss(msk_pred, msk, label_weights=[1., 20., 50.], device=device)
                loss = DiceLoss(msk_pred, msk, label_weights=[1., 20., 500.], device=device)

                if valid_mode:
                    pass
                else:
                    loss.backward()
                    optimizer.step()

                loss = loss.item()  # release
                cum_loss += loss
                if valid_mode:
                    print("\r--------(valid) {:.2%} Loss: {:.3f} (time: {:.1f}s) |supp: {}".format(
                        (iter_ind+1)/n_batch_per_epoch, cum_loss/(iter_ind+1), time.time()-st, supplement_logs), end="")
                else:
                    print("\rEpoch: {:3}/{} {:.2%} Loss: {:.3f} (time: {:.1f}s) |supp: {}".format(
                        (epoch+1), args.epochs, (iter_ind+1)/n_batch_per_epoch, cum_loss/(iter_ind+1), time.time()-st, supplement_logs), end="")
            print()
            if valid_mode:
                torch.set_grad_enabled(prev_grad_mode)

        valid_mean_loss = cum_loss/(iter_ind+1)  # validation (mean) loss of the current epoch

        if best_model_path and (valid_mean_loss < best_valid_loss):
            printGreen("Valid loss decreases from {:.5f} to {:.5f}, saving best model.".format(
                best_valid_loss, valid_mean_loss))
            best_valid_loss = valid_mean_loss
            # Only need to save the weights
            # torch.save(model.state_dict(), best_model_path)
            # save the entire model
            torch.save(model, best_model_path)

    return best_valid_loss


if __name__ == "__main__":
    print("Start")
    # usage: python train.py --epochs 50 -lr 0.0001 --log-dir "./weights/Exp_000" -trainf "./data/train_LiTS_db.h5" -validf "./data/valid_LiTS_db.h5" --batch-size 32  --num-cpu 32 --shuffle
    main()
