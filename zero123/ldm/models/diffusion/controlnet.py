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

# NOTE conrolNET variant
# 증손자 class

class MultiControlNet(LatentDiffusion):
    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
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
    # 그리고 control에 대해서도 완전히 noise를 적용하지 않는 
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model # free from wrapper

        # control of langauge
        # c_crossattn이 어떻게 construct 되는 지 확인 필요
        # TODO DDPM class로 hack을 해야한다.
        # TODO -> DONE 입력 loader 또한 변형을 해주어야 한다.
        # cond_txt = torch.cat(cond['c_crossattn'], 1)
        # import ipdb; ipdb.set_trace()
        
        # pose controller에는 text와 pose 정보가 conditioning을 하되, hint는 None으로 한다.
        # 그리하여 initial noise와 vanilla text embedding with pose injection 정보를 embedding하는 network가 되도록 해야 한다.
        # 다만 이렇기 때문에 pose 정보를 sinusoidal expansion하고 non-linear encoding scheme까지 도입을 할 수는 있을 거 같다.
        
        # canny edge controller에는 hint로 canny edge map과 vanilla text embedding를 넣어주도록 한다.

        # 그리고 vanilla LDM에 대해서는 vanilla text만을 이용해서 stable diffusion의 능력을 그대로 활용하도록 한다.
        
        # zero convolution을 zero initialize를 꼭해서 diffusion이 망가지지 않도록 한다.
        
        # control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
        # control = [c * scale for c, scale in zip(control, self.control_scales)]
        
        # TODO 여기 바꿔주기
        # 어떻게 control을 composition할 지도 중요한 issue이다.
        
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
                    # 여기 positional embedding signal을 엄청 늘려 버려야 겠다.
                elif c_key == 'canny':
                    # 원본 controlnet에서는 convolution layer로 latent space로 mapping하는 데 여기 코드 구현 좀 봐야겠다.
                    canny_image_embeding = cond["canny"]
                    control_list.append(
                        c_model(x=x_noisy, hint=canny_image_embeding, timesteps=t, context=cond_txt)
                    ) 
            
            control = []
            for ctr_idx, ctr_scale in enumerate(self.control_scales):
                control.append(
                    control_list[0][ctr_idx] * ctr_scale + control_list[1][ctr_idx] * ctr_scale 
                )

            # control signal added
            # 여기다. 그런데, conntrol signamling을 할 때, 귀찮아서 DiffusionWrapper의 forward를 쓰지 않고 우회해서 하고 있다.
            # 따라서 input만 잘 정의해주면 이제 거의 문제가 끝나 가는 듯?
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    # NOTE T2I image generation이므로 여기에 쓸 unconditional signal은 null text가 되어야 한다.
    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size):
        # if null_label is not None:
        #     xc = null_label
        #     if isinstance(xc, ListConfig):
        #         xc = list(xc)
        #     if isinstance(xc, dict) or isinstance(xc, list):
        #         c = self.get_learned_conditioning(xc)
        #     else:
        #         if hasattr(xc, "to"):
        #             xc = xc.to(self.device)
        #         c = self.get_learned_conditioning(xc)
        # else:
        #     # todo: get null label from cond_stage_model
        #     raise NotImplementedError()
        # c = repeat(c, '1 ... -> b ...', b=batch_size).to(self.device)
        # cond = {}
        # cond["c_crossattn"] = [c]
        # cond["c_concat"] = [torch.zeros([batch_size, 4, image_size // 8, image_size // 8]).to(self.device)]
        # # cond["c_control"] = [torch.zeros([batch_size, 4, image_size // 8, image_size // 8]).to(self.device)]
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
            # T2I image generation이므로 여기에 쓸 unconditional signal은 null text가 되어야 한다.
            
            # cross attention시에 null text를 넣어주되, control net signal은 여전히 pose 정보는 들어가야만 한다.
            # 다만 걱정되는 것이 spatial control signal은 어떻게 해야 할 지에 대한 문제이다.
            
            # text pose만 null embedding으로 generation하면 될 것 같기는 한데
            # control signal은 어떻게 처리될 지 모르겠다.
            uc_full = self.get_unconditional_conditioning(batch_size=N)
            
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
