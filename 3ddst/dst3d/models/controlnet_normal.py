import cv2
# from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
import numpy as np
from PIL import Image
import torch

from .base_model import BaseModel
from annotator.normalbae import NormalBaeDetector
from annotator.util import resize_image, HWC3

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
class ControlNetNormalBae(BaseModel):
    def __init__(self, model_name='Flux.1-dev-Controlnet-Surface-Normals/control_normalbae', img_resolution=512, controlnet_conditioning_scale=1.0, device='cpu', **kwargs):
        super().__init__(model_name, device)
        # self.canny_lower = canny_lower
        # self.canny_upper = canny_upper
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.img_resolution = img_resolution

        self.apply_normal = NormalBaeDetector()

        # Build ControlNet pipeline
        # controlnet = ControlNetModel.from_pretrained(
        #     "diffusers/controlnet-canny-sdxl-1.0",
        #     torch_dtype=torch.float16).to(device)
        controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Surface-Normals",
            torch_dtype=torch.float16
        )
        # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(device)
        # self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-base-1.0",
        #     controlnet=controlnet,
        #     vae=vae,
        #     torch_dtype=torch.float16).to(device)
        self.pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        )
        # self.pipe.enable_model_cpu_offload() # modified
        # self.pipe.enable_attention_slicing()
        # self.pipe.enable_sequential_cpu_offload()
        # print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        # print(f"Reserved memory: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        # print(f"Peak memory: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
        # torch.cuda.empty_cache()
        # torch.cuda.reset_peak_memory_stats()
        self.pipe.to("cuda")
        

    def get_condition(self, image):
        image = resize_image(HWC3(image), self.img_resolution)
        normal_map = self.apply_normal(image)
        normal_map = HWC3(normal_map)
        return normal_map

    @torch.no_grad()
    def forward(self, image=None, image_name=None, visual_prompt=None, prompt=None, negative_prompt=None, strength=None, ddim_steps=None, scale=None, normal_path=None):
        assert prompt is not None and negative_prompt is not None
        assert image is not None or visual_prompt is not None

        if strength is None:
            strength = self.controlnet_conditioning_scale

        if visual_prompt is None:
            visual_prompt = [self.get_condition(im) for im in image]
            # visual_prompt = [(self.get_condition(im), im_name) for im, im_name in zip(image, image_name)]
        
        # ### to only save the normal map ###
        # for i in range(len(visual_prompt)):
        #     img = visual_prompt[i][0]
        #     img_name = visual_prompt[i][1]
        #     if isinstance(img, np.ndarray):
        #         img = Image.fromarray(img)
        #     img.save(os.path.join(normal_path, img_name))

        # return 0
    
        return self.pipe(prompt, negative_prompt=negative_prompt, control_image=visual_prompt, controlnet_conditioning_scale=strength, num_inference_steps=ddim_steps, guidance_scale=scale).images
