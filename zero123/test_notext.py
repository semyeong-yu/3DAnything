import argparse, os
from omegaconf import OmegaConf
import math
from tqdm import tqdm
from PIL import Image
import cv2
import imageio
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import torch
import torch.nn as nn
import random
from ldm.util import *
from ldm.data.control import ObjaverseData, ObjaverseDataNormal
from torch.utils.data import DataLoader
from einops import rearrange

def minus_one_to_one(x):
    return 2 * x - 1.0

def zero_to_one(x):
    return (x + 1.0) / 2.0

def save_tensor(img, path, norm_z=False, dest='png'):
    if not norm_z:
        img = torch.clip(img, -1, 1)
        img = zero_to_one(img)
    else:
        img = torch.clip(img, 0, 1)
    
    if len(img.shape) == 4:
        img = img.cpu().numpy().transpose(0,2,3,1)[0]
    else:
        img = img.cpu().numpy()[0]
    if dest == 'png':
        imageio.imwrite(path+'.png', (img*255).astype(np.uint8))
    elif dest == 'exr':
        imageio.imwrite(path+'.exr', img)

def cartesian_to_spherical(self, xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    return np.array([theta, azimuth, z])

def get_T(self, target_RT, cond_RT):
    R, T = target_RT[:3, :3], target_RT[:3, -1]
    T_target = -R.T @ T

    R, T = cond_RT[:3, :3], cond_RT[:3, -1]
    T_cond = -R.T @ T

    theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
    
    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond
    
    d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
    return d_T
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200
    )
    parser.add_argument(
        "--cond_type",
        type=str,
        nargs="?",
    )
    
    opt = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    
    config = OmegaConf.load(opt.config)
    
    exp_path = os.path.join("results", opt.exp_name)
    os.makedirs(exp_path, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(os.path.join("logs", opt.ckpt), map_location="cpu")["state_dict"], strict=False)
    model = model.to(device)
    OmegaConf.save(config=config, f=os.path.join(exp_path, 'config.yaml'))
    
    dataset = instantiate_from_config(config=config.data.params.validation)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size)
    
    with torch.no_grad():
        with model.ema_scope():
            for index, batch in enumerate(dataloader):
                cond = {}
                target_img = batch["image_target"].to(device)
                cond_img = batch["image_cond"].to(device)
                T = batch["T"].to(device)
                target_img = rearrange(target_img, 'b h w c -> b c h w')
                cond_img = rearrange(cond_img, 'b h w c -> b c h w')
                
                clip_emb = model.get_learned_conditioning(cond_img)
                cond["text"] = clip_emb[:, None, :]
                cond["latent"] = model.encode_first_stage((cond_img)).mode()
                cond["control"] = cond_img
                cond["pose"] = model.cc_projection(torch.cat([clip_emb[:, None, :], T[:, None, :]], dim=-1))
                
                diffusion_samples, _ = model.sample_log(cond=cond,
                                              batch_size=opt.batch_size,
                                              ddim=True,
                                              ddim_steps=opt.ddim_steps,
                                              eta=0.)
                
                samples = model.decode_first_stage(diffusion_samples)
                for i in range(samples.shape[0]):
                    save_tensor(samples[i], os.path.join(exp_path, f'{index*opt.batch_size+i}_sample'))
                    save_tensor(target_img[i], os.path.join(exp_path, f'{index*opt.batch_size+i}_target'))
                    save_tensor(cond_img[i], os.path.join(exp_path, f'{index*opt.batch_size+i}_cond'))