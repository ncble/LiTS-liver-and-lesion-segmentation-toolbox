"""
Copyright (C) 2020  Lu Lin
Warning: The code is released under the license GNU GPL 3.0



"""
import os
import glob
import numpy as np
import pickle


def read_inference_results(filepath):
    with open(filepath, "rb") as file:
        results = pickle.load(file)
    
    liver_dice = np.array(results['liver']['dice'])
    tumor_dice = np.array(results['tumor']['dice'])
    tumor_dice_bis = np.mean(tumor_dice[tumor_dice>0])
    print(f"Liver mean dice: {np.mean(liver_dice):.4f}")
    print(f"Tumor mean dice: {np.mean(tumor_dice):.4f}")
    print(f"Tumor mean dice bis: {tumor_dice_bis:.4f}")
    # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    print("Start")
    read_inference_results("./results/Exp_301_bis/results.pkl")
