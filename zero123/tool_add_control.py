import sys
import os
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

assert len(sys.argv) == 3, 'Args are wrong.'

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
# from share import *

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='/mnt/datassd/seeha/3DAnything/zero123/configs/canny-edge.yaml')

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()
        # orig_conv_weight = sd[f"{key_prefix}.weight"]
        # orig_conv_bias = sd[f"{key_prefix}.bias"]
        # modif_conv_weight = orig_conv_weight[:, 0:self.unet_in_channels, :, :]
        # sd_new = sd
        # for name, param in sd_new:
        #     if name == f"{key_prefix}.weight":
        #         sd_new[name] = modif_conv_weight
target_dict = {}
key_prefix = "input_blocks.0.0.weight"
unet_in_channels = 4

for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        if key_prefix in copy_k:
            print(pretrained_weights[copy_k].clone().shape)
            target_dict[k] = pretrained_weights[copy_k].clone()[:, 0:unet_in_channels, :, :]
        else:
            target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
