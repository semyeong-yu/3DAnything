import numpy as np
from omegaconf import ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random
import matplotlib.pyplot as plt
import os
import math
from torchvision import transforms
from einops import rearrange

class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='/mnt/datassd/seeha/data/3D/train',
        default_trans=torch.zeros(3),
        return_paths=False,
        total_view=12,
        validation=False,
        patch_size=256
        ):
        
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        self.total_view = total_view
        # self.tform = image_transforms
        self.patch_size = patch_size
        if self.patch_size > 0:
            image_transforms = [transforms.Resize(self.patch_size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        # image_transforms.extend([transforms.ToTensor()])
        self.image_transforms = transforms.Compose(image_transforms)
        
        with open('/mnt/datassd/seeha/data/3D/paths.txt') as f:
            self.paths = f.read().splitlines()
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        
    def __len__(self):
        return len(self.paths)
    
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
    
    def process_im(self, im):
        im = im.convert("RGB")
        return self.image_transforms(im)
    
    # def load_im(self, path, color):
    def load_im(self, path):
        img = plt.imread(path)
        # img[img[:,:,-1] == 0.] = color # fix empty pixels
        img = Image.fromarray(np.uint8(img[:,:,:3]*255.))
        return img
    
    def __getitem__(self, index):
        data = {}
        total_view = self.total_view
        index_target, index_cond = random.sample(range(total_view), 2)
        filename = os.path.join(self.root_dir, self.paths[index])
        
        if self.return_paths:
            data["path"] = str(filename)
            
        color = [1., 1., 1., 1.]
        target_im = self.process_im(self.load_im(os.path.join(filename, 'image_render', '%03d.png' % index_target)))
        cond_im = self.process_im(self.load_im(os.path.join(filename, 'cannyedge_render', '%03d.png' % index_cond)))
        # target_im = self.process_im(self.load_im(os.path.join(filename, 'image_render', '%03d.png' % index_target), color))
        # cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
        target_RT = np.load(os.path.join(filename, 'annotation', '%03d.npy' % index_target), allow_pickle=True).item()['matrix_world']
        cond_RT = np.load(os.path.join(filename, 'annotation', '%03d.npy' % index_cond), allow_pickle=True).item()['matrix_world']

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        return data
    