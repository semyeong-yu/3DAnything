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
import pandas as pd 

DIRECTORY_MAP = {
    "rgb": "image_render",
    "canny": "cannyedge_render",
    "normal": "normal_render",
    "depth": "depth_render",
}

# NOTE: 여기가 현재 사용하는 data의 정의 부분
class ObjaverseData(Dataset):
    def __init__(self,
        root_dir,
        txt_path,
        caption_path=None,
        default_trans=torch.zeros(3),
        return_paths=False,
        total_view=12,
        patch_size=256,
        spatial_key=None,
        verbose:bool=False,
        determin_view:bool=False,
        ):
        
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        
        self.caption_path = caption_path
        if caption_path is not None:
            self.caption = pd.read_csv(caption_path)
        
        if spatial_key is not None:
            self.spatial_key: str = spatial_key
        assert spatial_key is not None, "Spatial key must be provided for Multimodal variant network."
        
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
        
        # with open('/mnt/datassd/seeha/data/3D/paths.txt') as f:
        with open(txt_path) as f:
            self.paths = f.read().splitlines()
        
        self.verbose = verbose
        self.determin_view = determin_view
            
        # print('============= length of dataset %d =============' % len(self.paths))
        
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
    
    # TODO: coordinate가 적절한가?
    # 카메라의 view direction에 대한 정보가 소실되는 느낌인데 괜찮으려나, 원본 zero123 paper에서 확인한 바로는 괜찮다.
    # 현재의 pipeline이 blender cam의 위치 정보에 대해서 받게 되고, camera를 object를 기준으로 spherical coordinate로 투영하는 방식인데, 이에 correspond하게 변환을 하므로 일단은 적절하다.
    # 이거 실제 coordinate 상에서 plot을 해봐야 겠다.
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
    
    # TODO: 다만, 현재로는 입력으로 사용되는 정보가 data["image_cond"]이거 하나인데, 이게 실은 canny edge이다. 그러면 걱정되는 점은 원래 controlnet 입력으로 사용되는 정보는 T2I에 spatial information이 있는 control signal을 넣는 것인데, 현재의 형태로는 사실상 zero123의 finetune에 불과한 상황이다.

    def __getitem__(self, index):
        data = {}
        filename = os.path.join(self.root_dir, self.paths[index])
        # view_num = len(os.listdir(f'{filename}/image_render')) # 이렇게 하면 안될 듯, dataframe에서 긁는게 더 나을 것이다.
        
        meta_class_name = filename.split('/')[-2]
        class_name = filename.split('/')[-1]
        base_name_list = self.caption[(self.caption["meta_class"] == meta_class_name) & (self.caption["class"] == class_name)]["base_name"].values
        
        if self.determin_view:
            index_target, index_cond = ("001", "000")
        else:    
            index_target, index_cond = np.random.choice(base_name_list, size=2, replace=False)
            index_target = index_target.split('.')[0]
            index_cond = index_cond.split('.')[0]
        
        if self.return_paths:
            data["path"] = str(filename)
            
        color = [1., 1., 1., 1.]
        target_img_path = os.path.join(filename, 'image_render', index_target + '.png')
        target_im = self.process_im(self.load_im(target_img_path))
        
        source_rgb = self.process_im(self.load_im(os.path.join(filename, DIRECTORY_MAP[self.spatial_key], index_cond + '.png')))
        data["rgb"] = source_rgb
        
        cond_im = self.process_im(self.load_im(os.path.join(filename, DIRECTORY_MAP[self.spatial_key], index_cond + '.png')))
        data[self.spatial_key] = cond_im
        
        target_RT = np.load(os.path.join(filename, 'annotation', index_target + ".npy"), allow_pickle=True).item()['modelview_matrix']
        cond_RT = np.load(os.path.join(filename, 'annotation', index_cond + ".npy"), allow_pickle=True).item()['modelview_matrix']

        data["image_target"] = target_im

        data["T"] = self.get_T(target_RT, cond_RT)
        
        if self.verbose:
            data["object_id"] = f"{meta_class_name}/{class_name}"
            data["viewpoint_ids"] = {
                "cond": index_cond,
                "target": index_target
            }
        
        if self.caption_path is not None:
            split_path = target_img_path.split('/')
            data["txt"] = self.caption[
                (self.caption["base_name"] == split_path[-1]) &
                (self.caption["class"] == split_path[-3])
            ]["caption"].values[0]

        return data
    