import einops
import torch
import torch.nn as nn

from omegaconf import ListConfig, DictConfig
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from ldm.modules.diffusionmodules.controlnetmodel import ControlNet

from typing import List

class MultiModalControlNetV3(LatentDiffusion):
    # 전체 network l
    # T2I가 main network이고, I2I가 control signal을 주는 형태
    def __init__(self, control_stage_config, control_key, only_mid_control, uncond=None, opt_mode=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 이 부분을 개조 하면 될 것으로 보인다.
        # 처음이 pose에 대한 network, 두번째가 canny edge에 대한 network
        self.control_key: str = control_key
        # self.control_stage_config = control_stage_config
        self.control_model = instantiate_from_config(control_stage_config)
        
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        
        self.uncond = uncond
        if self.uncond is not None:
            print(f"Using uncond prob: {self.uncond}")
        
        # TODO 현재 보기에는 control_scales이 다 똑같은데 spatial resolution별로 다르게 조절하는 것이 좋을 것이다.

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, cond_key=None, return_original_cond=False, bs=None, uncond=0.05):
        
        if uncond != 0.05 and getattr(self, "uncond", None) is not None:
            uncond = self.uncond 
        
        inputs = super().get_input(batch, k, return_first_stage_outputs, cond_key, return_original_cond, bs, uncond, spatial_key=self.control_key) # c : {c_concat : pimg], c_crossattn : [text + pose]}
        z = inputs[0]
        c = inputs[1]
        out = [z, c]
        if return_first_stage_outputs:
            out.extend([inputs[2], inputs[3]])
        if return_original_cond:
            out.append(inputs[4])
        return out  # 여기 override해버렸다.

    # NOTE 여기에서 두 종의 control signal을 넣어주도록 한다.
    # 그리고 control에 대해서도 완전히 noise를 적용하지 않는 
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model # free from wrapper
        
        if "spatial" not in cond:
            _latent_hint = cond["c_latent"]
            _x = torch.cat([x_noisy, _latent_hint], dim=1)
            
            eps = diffusion_model(x=_x, timesteps=t, context=cond["c_pose"], control=None, only_mid_control=self.only_mid_control)
        
        else:
            cond_txt = cond['c_text']
            
            _latent_hint = cond["c_latent"]
            _x = torch.cat([x_noisy, _latent_hint], dim=1)
            
            # hint로서 주는 것은 model을 또 intialize해줘야 하니까 빼는 것으로 하자.
            control = self.control_model(x=x_noisy, hint=None, timesteps=t, context=cond["c_text"])
            
            for ctr_idx, ctr_scale in enumerate(self.control_scales):
                control[ctr_idx] = control[ctr_idx] * ctr_scale

            eps = diffusion_model(x=_x, timesteps=t, context=cond["c_pose"], control=control, only_mid_control=self.only_mid_control)

        return eps

    # NOTE T2I image generation이므로 여기에 쓸 unconditional signal은 null text가 되어야 한다.
    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=[""], image_size=512):
        cond = {}
        cond["c_pose"] = self.get_learned_conditioning(null_label, modality="txt").repeat(batch_size, 1, 1)
        cond["c_latent"] = torch.zeros([batch_size, 4, image_size // 8, image_size // 8]).to(self.device)
        
        return cond

    @torch.no_grad()
    # TODO 여기 뜯어 고쳐야 함
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False, plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True, return_original_cond=True, bs=N, uncond=0.0)  # control net에 적합한 data generation done
        N = min(z.shape[0], N)
        # c_control = c["c_concat"][0][:N]
        c_control = c["spatial"][:N]
        
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["control"] = c_control
        
        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            samples, z_denoise_row = self.sample_log(cond=c,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_full = self.get_unconditional_conditioning(batch_size=N, null_label=[""], image_size=x.shape[-1])
            
            samples_cfg, _ = self.sample_log(cond=c,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        
        # import ipdb; ipdb.set_trace()

        return log
    
    @torch.no_grad()
    def log_val_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False, plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True, return_original_cond=True, bs=N, uncond=0.0)  # control net에 적합한 data generation done
        N = min(z.shape[0], N)
        # c_control = c["c_concat"][0][:N]
        c_control = c["spatial"][:N]
        
        n_row = min(x.shape[0], n_row)
        log["object_id"] = batch["object_id"][:N]
        
        _viewpoint_dict = batch["viewpoint_ids"]
        _viewpoint_dict["cond"] = _viewpoint_dict["cond"][:N]
        _viewpoint_dict["target"] = _viewpoint_dict["target"][:N]
        log["viewpoint_ids"] = _viewpoint_dict
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["control"] = c_control
        
        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            samples, z_denoise_row = self.sample_log(cond=c,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_full = self.get_unconditional_conditioning(batch_size=N, null_label=[""], image_size=x.shape[-1])
            
            samples_cfg, _ = self.sample_log(cond=c,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        
        if batch.get("txt", None) is not None:
            log["txt"] = batch["txt"][:N]
        
        # import ipdb; ipdb.set_trace()

        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.diffusion_model.parameters())
        # diffusion network를 오히려 finetune
        if not self.sd_locked:
            # self.model.eval()
            params += list(self.control_model.parameters())
            # control network를 오히려 고정
        if self.cc_projection is not None:
            print('========== optimizing for cc projection weight ==========')
            opt = torch.optim.AdamW([{"params": params, "lr": lr},
                                    {"params": self.cc_projection.parameters(), "lr": 10. * lr}], lr=lr)
        else:
            opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def load_pre_control(self, pre_control_cfg: DictConfig, debug=False):
        for key, value in pre_control_cfg.path.items():
            if key == "pose":  # 이게 semantic control signal로서 동작을 해야 한다.
                print(f"Loading {key} control model weights")
                weight = torch.load(value, map_location="cpu")["state_dict"]
                ret = self.load_state_dict(weight, strict=False)
                if debug:
                    print(f"{key} Control model weights: {ret}")
            elif key == "sd":  # 이것을 main에 집어넣고
                print(f"Loading {key} control model weights")
                weight = torch.load(value, map_location="cpu")["state_dict"]
                new_weight = {}
                for k, v in weight.items():
                    if ("diffusion_model" in k and "model_ema" not in k) and ("input_blocks" in k or "middle_block" in k):
                        _k = k.replace("model.diffusion_model.", "")
                        
                        if "zero_convs" in _k:
                            v = torch.zeros_like(v)
                        new_weight[_k] = v
                    
                    elif ("model_ema" not in k) and ("time_embed" in k):
                        _k = k.replace("model.diffusion_model.", "")
                    
                        new_weight[_k] = v
                        
                ret = self.control_model.load_state_dict(new_weight, strict=False)
                if debug:
                    print(f"{key} Control model weights: {ret}")

class MultiModalControlNetV2(LatentDiffusion):
    # T2I가 main network이고, I2I가 control signal을 주는 형태
    def __init__(self, control_stage_config, control_key, only_mid_control, uncond=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 이 부분을 개조 하면 될 것으로 보인다.
        # 처음이 pose에 대한 network, 두번째가 canny edge에 대한 network
        self.control_key: str = control_key
        # self.control_stage_config = control_stage_config
        self.control_model = instantiate_from_config(control_stage_config)
        
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        
        self.uncond = uncond
        if self.uncond is not None:
            print(f"Using uncond prob: {self.uncond}")
        
        # TODO 현재 보기에는 control_scales이 다 똑같은데 spatial resolution별로 다르게 조절하는 것이 좋을 것이다.

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, cond_key=None, return_original_cond=False, bs=None, uncond=0.05):
        
        if uncond != 0.05 and getattr(self, "uncond", None) is not None:
            uncond = self.uncond 
        
        inputs = super().get_input(batch, k, return_first_stage_outputs, cond_key, return_original_cond, bs, uncond) # c : {c_concat : pimg], c_crossattn : [text + pose]}
        z = inputs[0]
        c = inputs[1]
        out = [z, c]
        if return_first_stage_outputs:
            out.extend([inputs[2], inputs[3]])
        if return_original_cond:
            out.append(inputs[4])
        return out  # 여기 override해버렸다.

    # NOTE 여기에서 두 종의 control signal을 넣어주도록 한다.
    # 그리고 control에 대해서도 완전히 noise를 적용하지 않는 
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model # free from wrapper
        
        if "c_null_txt" in cond:
            # null condition diffusion
            cond_txt = cond["c_null_txt"]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        
        else:
            cond_txt = cond['c_text']
            # control_list = []
            # for c_key, c_model in zip(self.control_key_list, self.control_model_list):
            #     # pose 정보를 어떻게 embedding할 지에 대해서
            #     # canny edge map을 어떻게 embedding할 지에 대해서
            #     # hint가 어떻게 쓰이는지, 무조건 있어야 하는가? 이러한 것들
            #     if c_key == 'pose':
            #         control_list.append(
            #             c_model(x=x_noisy, hint=None, timesteps=t, context=cond['c_text_pose'])
            #         )
            #     elif c_key == 'canny':
            #         canny_image_embeding = cond["canny"]
            #         control_list.append(
            #             c_model(x=x_noisy, hint=canny_image_embeding, timesteps=t, context=cond_txt)
            #         )
            
            # canny edge 정보도 original implementation을 따라서 first stage model로서 encode해줘보자.
            
            _latent_hint = self.encode_first_stage(cond["canny"]).mode().detach()
            _x = torch.cat([x_noisy, _latent_hint], dim=1)
            
            # hint로서 주는 것은 model을 또 intialize해줘야 하니까 빼는 것으로 하자.
            control = self.control_model(x=_x, hint=None, timesteps=t, context=cond["c_pose"])
            # conditioning model 하나더 필요함
            
            for ctr_idx, ctr_scale in enumerate(self.control_scales):
                control[ctr_idx] = control[ctr_idx] * ctr_scale

            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    # NOTE T2I image generation이므로 여기에 쓸 unconditional signal은 null text가 되어야 한다.
    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size):
        cond = {}
        cond["c_null_txt"] = self.get_learned_conditioning([""], modality="txt").repeat(batch_size, 1, 1)
        
        return cond

    @torch.no_grad()
    # TODO 여기 뜯어 고쳐야 함
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False, plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True, return_original_cond=True, bs=N, uncond=0.0)  # control net에 적합한 data generation done
        N = min(z.shape[0], N)
        # c_control = c["c_concat"][0][:N]
        c_control = c["canny"][:N]
        
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["control"] = c_control
        
        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            samples, z_denoise_row = self.sample_log(cond=c,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_full = self.get_unconditional_conditioning(batch_size=N)
            
            samples_cfg, _ = self.sample_log(cond=c,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        
        # import ipdb; ipdb.set_trace()

        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            # self.model.eval()
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        if self.cc_projection is not None:
            print('========== optimizing for cc projection weight ==========')
            opt = torch.optim.AdamW([{"params": params, "lr": lr},
                                    {"params": self.cc_projection.parameters(), "lr": 10. * lr}], lr=lr)
        else:
            opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def load_pre_control(self, pre_control_cfg: DictConfig, debug=False):
        for key, value in pre_control_cfg.path.items():
            if key == "sd":
                print(f"Loading {key} control model weights")
                weight = torch.load(value, map_location="cpu")["state_dict"]
                ret = self.load_state_dict(weight, strict=False)
                if debug:
                    print(f"{key} Control model weights: {ret}")
            elif key == "pose":
                print(f"Loading {key} control model weights")
                weight = torch.load(value, map_location="cpu")["state_dict"]
                new_weight = {}
                for k, v in weight.items():
                    if ("diffusion_model" in k and "model_ema" not in k) and ("input_blocks" in k or "middle_block" in k):
                        _k = k.replace("model.diffusion_model.", "")
                        
                        # if "input_blocks.0.0.weight" in _k:
                        #     # (320, 8, 3, 3) -> (320, 4, 2, 3, 3) -> (320, 4, 3, 3)
                        #     v_shape = v.shape
                        #     v = v.view(v_shape[0], v_shape[1] // 2, 2, v_shape[2], v_shape[3])
                        #     v = v.mean(dim=2)
                        
                        if "zero_convs" in _k:
                            v = torch.zeros_like(v)
                        new_weight[_k] = v
                    
                    elif ("model_ema" not in k) and ("time_embed" in k):
                        _k = k.replace("model.diffusion_model.", "")
                    
                        new_weight[_k] = v
                        
                ret = self.control_model.load_state_dict(new_weight, strict=False)
                if debug:
                    print(f"{key} Control model weights: {ret}")
    



class MultiModalControlNet(LatentDiffusion):
    def __init__(self, control_stage_config, control_key, only_mid_control, uncond=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 이 부분을 개조 하면 될 것으로 보인다.
        # 처음이 pose에 대한 network, 두번째가 canny edge에 대한 network
        self.control_key: str = control_key
        # self.control_stage_config = control_stage_config
        self.control_model = instantiate_from_config(control_stage_config)
        
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        
        self.uncond = uncond
        if self.uncond is not None:
            print(f"Using uncond prob: {self.uncond}")
        
        # TODO 현재 보기에는 control_scales이 다 똑같은데 spatial resolution별로 다르게 조절하는 것이 좋을 것이다.

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, cond_key=None, return_original_cond=False, bs=None, uncond=0.05):
        
        if uncond != 0.05 and getattr(self, "uncond", None) is not None:
            uncond = self.uncond 
        
        inputs = super().get_input(batch, k, return_first_stage_outputs, cond_key, return_original_cond, bs, uncond) # c : {c_concat : pimg], c_crossattn : [text + pose]}
        z = inputs[0]
        c = inputs[1]
        out = [z, c]
        if return_first_stage_outputs:
            out.extend([inputs[2], inputs[3]])
        if return_original_cond:
            out.append(inputs[4])
        return out  # 여기 override해버렸다.

    # NOTE 여기에서 두 종의 control signal을 넣어주도록 한다.
    # 그리고 control에 대해서도 완전히 noise를 적용하지 않는 
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model # free from wrapper
        
        if "c_null_txt" in cond:
            # null condition diffusion
            cond_txt = cond["c_null_txt"]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        
        else:
            cond_txt = cond['c_text']
            # control_list = []
            # for c_key, c_model in zip(self.control_key_list, self.control_model_list):
            #     # pose 정보를 어떻게 embedding할 지에 대해서
            #     # canny edge map을 어떻게 embedding할 지에 대해서
            #     # hint가 어떻게 쓰이는지, 무조건 있어야 하는가? 이러한 것들
            #     if c_key == 'pose':
            #         control_list.append(
            #             c_model(x=x_noisy, hint=None, timesteps=t, context=cond['c_text_pose'])
            #         )
            #     elif c_key == 'canny':
            #         canny_image_embeding = cond["canny"]
            #         control_list.append(
            #             c_model(x=x_noisy, hint=canny_image_embeding, timesteps=t, context=cond_txt)
            #         )
            
            # canny edge 정보도 original implementation을 따라서 first stage model로서 encode해줘보자.
            
            control = self.control_model(x=x_noisy, hint=cond["canny"], timesteps=t, context=cond["c_pose"])
            # conditioning model 하나더 필요함
            
            for ctr_idx, ctr_scale in enumerate(self.control_scales):
                control[ctr_idx] = control[ctr_idx] * ctr_scale

            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    # NOTE T2I image generation이므로 여기에 쓸 unconditional signal은 null text가 되어야 한다.
    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size):
        cond = {}
        cond["c_null_txt"] = self.get_learned_conditioning([""], modality="txt").repeat(batch_size, 1, 1)
        
        return cond

    @torch.no_grad()
    # TODO 여기 뜯어 고쳐야 함
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False, plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True, return_original_cond=True, bs=N, uncond=0.0)  # control net에 적합한 data generation done
        N = min(z.shape[0], N)
        # c_control = c["c_concat"][0][:N]
        c_control = c["canny"][:N]
        
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["control"] = c_control
        
        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            samples, z_denoise_row = self.sample_log(cond=c,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_full = self.get_unconditional_conditioning(batch_size=N)
            
            samples_cfg, _ = self.sample_log(cond=c,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        
        # import ipdb; ipdb.set_trace()

        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            # self.model.eval()
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        if self.cc_projection is not None:
            print('========== optimizing for cc projection weight ==========')
            opt = torch.optim.AdamW([{"params": params, "lr": lr},
                                    {"params": self.cc_projection.parameters(), "lr": 10. * lr}], lr=lr)
        else:
            opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def load_pre_control(self, pre_control_cfg: DictConfig, debug=False):
        for key, value in pre_control_cfg.path.items():
            if key == "sd":
                print(f"Loading {key} control model weights")
                weight = torch.load(value, map_location="cpu")["state_dict"]
                ret = self.load_state_dict(weight, strict=False)
                if debug:
                    print(f"{key} Control model weights: {ret}")
            elif key == "pose":
                print(f"Loading {key} control model weights")
                weight = torch.load(value, map_location="cpu")["state_dict"]
                new_weight = {}
                for k, v in weight.items():
                    if ("diffusion_model" in k and "model_ema" not in k) and ("input_blocks" in k or "middle_block" in k):
                        _k = k.replace("model.diffusion_model.", "")
                        
                        if "input_blocks.0.0.weight" in _k:
                            # (320, 8, 3, 3) -> (320, 4, 2, 3, 3) -> (320, 4, 3, 3)
                            v_shape = v.shape
                            v = v.view(v_shape[0], v_shape[1] // 2, 2, v_shape[2], v_shape[3])
                            v = v.mean(dim=2)
                        
                        if "zero_convs" in _k:
                            v = torch.zeros_like(v)
                        new_weight[_k] = v
                    
                    elif ("model_ema" not in k) and ("time_embed" in k):
                        _k = k.replace("model.diffusion_model.", "")
                    
                        new_weight[_k] = v
                        
                ret = self.control_model.load_state_dict(new_weight, strict=False)
                if debug:
                    print(f"{key} Control model weights: {ret}")
    

# NOTE conrolNET variant
# 증손자 class

class MultiControlNet(LatentDiffusion):
    def __init__(self, control_stage_config, control_key, only_mid_control, uncond=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 이 부분을 개조 하면 될 것으로 보인다.
        # 처음이 pose에 대한 network, 두번째가 canny edge에 대한 network
        self.control_key_list: List[str] = control_key
        # self.control_stage_config = control_stage_config
        
        self.control_model_list = nn.ModuleList()
        if not isinstance(control_stage_config, list):
            assert isinstance(control_stage_config, ListConfig), f"control_stage_config should be a list of controlnet configs, but got {type(control_stage_config)} as {control_stage_config}"
        for i, control_model in enumerate(control_stage_config):
            self.control_model_list.append(instantiate_from_config(control_model))
        
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        
        self.uncond = uncond
        if self.uncond is not None:
            print(f"Using uncond prob: {self.uncond}")
        
        # TODO 현재 보기에는 control_scales이 다 똑같은데 spatial resolution별로 다르게 조절하는 것이 좋을 것이다.

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, cond_key=None, return_original_cond=False, bs=None, uncond=0.05):
        
        if uncond != 0.05 and getattr(self, "uncond", None) is not None:
            uncond = self.uncond 
        
        inputs = super().get_input(batch, k, return_first_stage_outputs, cond_key, return_original_cond, bs, uncond) # c : {c_concat : pimg], c_crossattn : [text + pose]}
        z = inputs[0]
        c = inputs[1]
        out = [z, c]
        if return_first_stage_outputs:
            out.extend([inputs[2], inputs[3]])
        if return_original_cond:
            out.append(inputs[4])
        return out  # 여기 override해버렸다.

    # NOTE 여기에서 두 종의 control signal을 넣어주도록 한다.
    # 그리고 control에 대해서도 완전히 noise를 적용하지 않는 
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model # free from wrapper
        
        if "c_null_txt" in cond:
            # null condition diffusion
            cond_txt = cond["c_null_txt"]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        
        else:
            cond_txt = cond['c_text']
            control_list = []
            for c_key, c_model in zip(self.control_key_list, self.control_model_list):
                # pose 정보를 어떻게 embedding할 지에 대해서
                # canny edge map을 어떻게 embedding할 지에 대해서
                # hint가 어떻게 쓰이는지, 무조건 있어야 하는가? 이러한 것들
                if c_key == 'pose':
                    control_list.append(
                        c_model(x=x_noisy, hint=None, timesteps=t, context=cond['c_text_pose'])
                    )
                elif c_key == 'canny':
                    canny_image_embeding = cond["canny"]
                    control_list.append(
                        c_model(x=x_noisy, hint=canny_image_embeding, timesteps=t, context=cond_txt)
                    ) 
            
            control = []
            for ctr_idx, ctr_scale in enumerate(self.control_scales):
                control.append(
                    control_list[0][ctr_idx] * ctr_scale + control_list[1][ctr_idx] * ctr_scale 
                )
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    # NOTE T2I image generation이므로 여기에 쓸 unconditional signal은 null text가 되어야 한다.
    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size):
        cond = {}
        cond["c_null_txt"] = self.get_learned_conditioning([""]).repeat(batch_size, 1, 1)
        
        return cond

    @torch.no_grad()
    # TODO 여기 뜯어 고쳐야 함
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False, plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True, return_original_cond=True, bs=N, uncond=0.0)  # control net에 적합한 data generation done
        N = min(z.shape[0], N)
        # c_control = c["c_concat"][0][:N]
        c_control = c["canny"][:N]
        
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["control"] = c_control
        
        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            samples, z_denoise_row = self.sample_log(cond=c,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_full = self.get_unconditional_conditioning(batch_size=N)
            
            samples_cfg, _ = self.sample_log(cond=c,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        
        # import ipdb; ipdb.set_trace()

        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model_list.parameters())
        if not self.sd_locked:
            # self.model.eval()
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        if self.cc_projection is not None:
            print('========== optimizing for cc projection weight ==========')
            opt = torch.optim.AdamW([{"params": params, "lr": lr},
                                    {"params": self.cc_projection.parameters(), "lr": 10. * lr}], lr=lr)
        else:
            opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def load_pre_control(self, pre_control_cfg: DictConfig, debug=False):
        for key, value in pre_control_cfg.path.items():
            if key == "sd":
                print(f"Loading {key} control model weights")
                weight = torch.load(value, map_location="cpu")["state_dict"]
                ret = self.load_state_dict(weight, strict=False)
                if debug:
                    print(f"{key} Control model weights: {ret}")
            elif key == "pose":
                print(f"Loading {key} control model weights")
                weight = torch.load(value, map_location="cpu")["state_dict"]
                new_weight = {}
                for k, v in weight.items():
                    if ("diffusion_model" in k and "model_ema" not in k) and ("input_blocks" in k or "middle_block" in k):
                        _k = k.replace("model.diffusion_model.", "")
                        
                        if "input_blocks.0.0.weight" in _k:
                            # (320, 8, 3, 3) -> (320, 4, 2, 3, 3) -> (320, 4, 3, 3)
                            v_shape = v.shape
                            v = v.view(v_shape[0], v_shape[1] // 2, 2, v_shape[2], v_shape[3])
                            v = v.mean(dim=2)
                        
                        if "zero_convs" in _k:
                            v = torch.zeros_like(v)
                        new_weight[_k] = v
                    
                    elif ("model_ema" not in k) and ("time_embed" in k):
                        _k = k.replace("model.diffusion_model.", "")
                    
                        new_weight[_k] = v
                        
                ret = self.control_model_list[0].load_state_dict(new_weight, strict=False)
                if debug:
                    print(f"{key} Control model weights: {ret}")
            elif key == "canny":
                print(f"Loading {key} control model weights")
                weight = torch.load(value, map_location="cpu")
                new_weight = {}
                for k, v in weight.items():
                    if "control_model" in k:
                        _k = k.replace("control_model.", "")
                        if "zero_convs" in _k:
                            v = torch.zeros_like(v)
                        new_weight[_k] = v
                        
                ret = self.control_model_list[1].load_state_dict(new_weight, strict=False)
                if debug:
                    print(f"{key} Control model weights: {ret}")
    
class ControlLDM(LatentDiffusion):
    """
    1. DDPM model을 추가함
        a. 이는 zero123랑 비슷한 implementation을 보임 
    2. controlnet을 추가함
    """
    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        # 이 부분을 개조 하면 될 것으로 보인다.
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        # TODO 현재 보기에는 control_scales이 다 똑같은데 spatial resolution별로 다르게 조절하는 것이 좋을 것이다.

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, cond_key=None, return_original_cond=False, bs=None, uncond=0.05):
        inputs = super().get_input(batch, k, return_first_stage_outputs, cond_key, return_original_cond, bs, uncond) # c : {c_concat : pimg], c_crossattn : [text + pose]}
        z = inputs[0]
        c = inputs[1]
        out = [z, c]
        if return_first_stage_outputs:
            out.extend([inputs[2], inputs[3]])
        if return_original_cond:
            out.append(inputs[4])
        return out  # 여기 override해버렸다.

    # NOTE 여기에서 두 종의 control signal을 넣어주도록 한다.
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model # free from wrapper

        # control of langauge
        # c_crossattn이 어떻게 construct 되는 지 확인 필요
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        # control of edge
        # c_concat이 어떻게 construct 되는 지 확인 필요
        
        control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
                # control of pose + langauge
                
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        # eps = diffusion_model(x=torch.cat([x_noisy] + cond['c_concat'], dim=1), timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        
        # import ipdb; ipdb.set_trace()
        
        # control signal added 
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    # TODO: non-referenced function
    def get_unconditional_conditioning(self, batch_size, null_label=None, image_size=512):
        if null_label is not None:
            xc = null_label
            if isinstance(xc, ListConfig):
                xc = list(xc)
            if isinstance(xc, dict) or isinstance(xc, list):
                c = self.get_learned_conditioning(xc)
            else:
                if hasattr(xc, "to"):
                    xc = xc.to(self.device)
                c = self.get_learned_conditioning(xc)
        else:
            # todo: get null label from cond_stage_model
            raise NotImplementedError()
        c = repeat(c, '1 ... -> b ...', b=batch_size).to(self.device)
        cond = {}
        cond["c_crossattn"] = [c]
        cond["c_concat"] = [torch.zeros([batch_size, 4, image_size // 8, image_size // 8]).to(self.device)]
        # cond["c_control"] = [torch.zeros([batch_size, 4, image_size // 8, image_size // 8]).to(self.device)]
        return cond

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False, plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True, return_original_cond=True, bs=N)
        N = min(z.shape[0], N)
        c_control = c["c_concat"][0][:N]
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["control"] = c_control
        
        
        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            samples, z_denoise_row = self.sample_log(cond=c,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = torch.zeros_like(c["c_crossattn"][0])
            uc_cat = torch.zeros_like(c["c_concat"][0]) # c_control
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond=c,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    def configure_optimizers(self):
        """
        여기가 핵심이다.
        1. 오직 controlnet의 parameter만을 update를 해야한다.
        
        model.first_stage_model: FREEZE
        model.model.diffusion_model: FREEZE
        model.cond_stage_model: FREEZE
        
        model.control_model: learnable
        나는 control signal을 한번에 두개에서 부여 해야 한다.
        
        """
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        if self.cc_projection is not None:
            print('========== optimizing for cc projection weight ==========')
            opt = torch.optim.AdamW([{"params": params, "lr": lr},
                                    {"params": self.cc_projection.parameters(), "lr": 10. * lr}], lr=lr)
        else:
            opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
