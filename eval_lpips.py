#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from argparse import ArgumentParser
from glob import glob
from torch.utils.data import Dataset, DataLoader

def readImage(path):
    img = Image.open(path)
    #return tf.to_tensor(img).unsqueeze(0)[:, :3, :, :].cuda()
    return tf.to_tensor(img)[ :3, :, :] #batch

class ImageDataset(Dataset):
    def __init__(self, pred_image_paths, gt_image_paths, image_names):
        self.pred_image_paths = pred_image_paths
        self.gt_image_paths = gt_image_paths
        self.image_names = image_names

    def __len__(self):
        return len(self.pred_image_paths)

    def __getitem__(self, idx):
        pred_img = readImage(self.pred_image_paths[idx])
        gt_img = readImage(self.gt_image_paths[idx])
        return pred_img, gt_img, self.image_names[idx]

    # Usage example:

def evaluate(pred_images_paths, gt_images_paths, image_names, output_name):

    img2score = {}
    metrics = []
    # dataset = ImageDataset(pred_images_paths, gt_images_paths, image_names)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    for pred_img_path, gt_img_path, img_name in tqdm(zip(pred_images_paths, gt_images_paths, image_names),
                                           desc="Metric evaluation progress", total=len(pred_images_paths)):
        pred_img = readImage(pred_img_path)
        gt_img = readImage(gt_img_path)
        lpips_loss = lpips(pred_img, gt_img, net_type='vgg')
        img2score[img_name] = lpips_loss.item()
        metrics.append(lpips_loss.item())

    # for pred_img_batch, gt_img_batch, img_name_batch in tqdm(dataloader,
    #                                                          desc="Metric evaluation progress"):
    #     with torch.no_grad():
    #         lpips_loss = lpips(pred_img_batch.cuda(), gt_img_batch.cuda(), net_type='vgg')
    #         for img_name, lpips_loss_item in zip(img_name_batch, lpips_loss):
    #             img2score[img_name] = lpips_loss_item.item()
    #             metrics.append(lpips_loss_item.item())

    with open(f'{output_name}_per_img.json', 'w') as f:
        json.dump(img2score, f)
    with open(f'{output_name}.txt', 'w') as f:
        f.write(f'{sum(metrics) / len(metrics)}')
    print(f'{output_name} mean: {sum(metrics) / len(metrics)}')

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gt_dir', '-g', required=True, type=str)
    parser.add_argument('--pred_dir', '-p', required=True, type=str)
    parser.add_argument('--blender_split_file', '-s', required=True, type=str)
    args = parser.parse_args()

    images_info = json.load(open(os.path.join(args.gt_dir, args.blender_split_file), 'r'))['frames']

    gt_images_paths = [f'{args.gt_dir}/{ii["file_path"]}' for ii in images_info]
    image_names = [ii["image_name"] for ii in images_info]

    pred_images_paths = [f'{args.pred_dir}/color_{ii["image_name"]}' for ii in images_info]
    output_name = os.path.join(args.pred_dir, 'metric_lpips')
    evaluate(pred_images_paths, gt_images_paths, image_names, output_name)

    pred_images_paths = [f'{args.pred_dir}/color_cc_{ii["image_name"]}' for ii in images_info]
    output_name = os.path.join(args.pred_dir, 'metric_cc_lpips')
    evaluate(pred_images_paths, gt_images_paths, image_names, output_name)