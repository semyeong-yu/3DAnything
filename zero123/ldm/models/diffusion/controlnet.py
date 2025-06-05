import einops
import torch
import torch.nn as nn

from omegaconf import ListConfig
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    # def get_input(self, batch, k, bs=None, *args, **kwargs):
    def get_input(self, batch, k, return_first_stage_outputs=False, cond_key=None, return_original_cond=False, bs=None, uncond=0.05):
        # x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs) # c : {c_concat : pimg], c_crossattn : [text + pose]}
        inputs = super().get_input(batch, k, return_first_stage_outputs, cond_key, return_original_cond, bs, uncond) # c : {c_concat : pimg], c_crossattn : [text + pose]}
        z = inputs[0]
        c = inputs[1]
        # control = c['c_concat']
        # control = batch[self.control_key]
        # if bs is not None:
        #     control = control[:bs]
        # control = control.to(self.device)
        # control = einops.rearrange(control, 'b h w c -> b c h w')
        # control = control.to(memory_format=torch.contiguous_format).float()
        # c["c_control"] = [control]
        out = [z, c]
        if return_first_stage_outputs:
            out.extend([inputs[2], inputs[3]])
        if return_original_cond:
            out.append(inputs[4])
        return out
        # return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model # free from wrapper

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        # if cond['c_concat'] is None:
        #     eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        # else:
        #     control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
        #     control = [c * scale for c, scale in zip(control, self.control_scales)]
        #     eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        # control = self.control_model(x=torch.cat([x_noisy] + cond['c_concat'], dim=1), hint=torch.cat(cond['c_control'], 1), timesteps=t, context=cond_txt)
        control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        # eps = diffusion_model(x=torch.cat([x_noisy] + cond['c_concat'], dim=1), timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    # @torch.no_grad()
    # def get_unconditional_conditioning(self, N):
    #     return self.get_learned_conditioning([""] * N)
    @torch.no_grad()
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
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=200, ddim_eta=1.0, plot_denoise_rows=False, plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True, return_original_cond=True, bs=N)
        N = min(z.shape[0], N)
        # c_control = c["c_control"][0][:N]
        c_control = c["c_concat"][0][:N]
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["cond"] = xc
        log["reconstruction"] = xrec
        log["control"] = c_control
        
        # log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        
        if plot_diffusion_rows:
            # get diffusion row
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
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond=c,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            # uc = self.get_unconditional_conditioning(N, unconditional_guidance_label, image_size=x.shape[-1])
            # uc_cross = self.get_unconditional_conditioning(N)
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

    # @torch.no_grad()
    # def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
    #     ddim_sampler = DDIMSampler(self)
    #     b, c, h, w = cond["c_concat"][0].shape
    #     shape = (self.channels, h // 8, w // 8)
    #     samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
    #     return samples, intermediates

    def configure_optimizers(self):
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
