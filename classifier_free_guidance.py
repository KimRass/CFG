import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import imageio
from tqdm import tqdm
import contextlib


class CFGDiffusion(nn.Module):
    def get_linear_beta_schdule(self):
        self.beta = torch.linspace(
            self.init_beta,
            self.fin_beta,
            self.n_diffusion_steps,
            device=self.device,
        )

    def __init__(
        self,
        model,
        img_size,
        device,
        min_lamb=-20, # "$\lambda_{\text{min}}$"
        max_lamb=20, # "$\lambda_{\text{max}}$"
        interpolation_coeff=0.3, # "$v$"
        uncond_prob=0.1,
        image_channels=3,
        n_diffusion_steps=1000,
        init_beta=0.0001,
        fin_beta=0.02,
    ):
        # "We sample $\lambda$ via \lambda = -2\log\tan(au + b) for uniformly distributed $u \in [0, 1]$,
        # where $b = \arctan(e^{-\lambda_{\text{max}} / 2})$
        # and $a = \arctan(e^{-\lambda_{\text{max}} / 2}) - b$."
        super().__init__()

        self.img_size = img_size
        self.device = device
        self.image_channels = image_channels
        self.n_diffusion_steps = n_diffusion_steps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.model = model.to(device)

        self.get_linear_beta_schdule()
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # min_lamb=-20
        # max_lamb=20
        self.b = torch.arctan(torch.exp(torch.tensor(-max_lamb / 2)))
        self.a = torch.arctan(torch.exp(torch.tensor(-min_lamb / 2))) - self.b

    def sample_noise(self, batch_size):
        """
        "$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$"
        """
        return torch.randn(
            size=(batch_size, self.image_channels, self.img_size, self.img_size),
            device=self.device,
        )

    def sample_lambda(self, batch_size):
        """
        "$\lambda \sim p(\lambda)$"
        """
        u = torch.rand(
            size=(batch_size, self.image_channels, self.img_size, self.img_size),
            device=self.device,
        )
        return -2 * torch.log(torch.tan(self.a * u + self.b))

    def lambda_to_signal_ratio(self, lamb):
        """
        "$\alpha^{2}_{\lambda} = 1 / (1 + e^{-\lambda})$"
        """
        return 1 / (1 + torch.exp(-lamb))

    def signal_ratio_to_noise_ratio(self, signal_ratio):
        """
        "$\sigma^{2}_{\lambda} = 1 - \alpha^{2}_{\lambda}$"
        """
        return 1 - signal_ratio

    def batchify_diffusion_steps(self, diffusion_step_idx, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=diffusion_step_idx,
            dtype=torch.long,
            device=self.device,
        )

    def perform_diffusion_process(self, ori_image, lamb, rand_noise=None):
        """
        "$\mathbf{z}_{\lambda} = \alpha_{\lambda}\mathbf{x} + \sigma_{\lambda}\mathbf{\epsilon}$"
        """
        signal_ratio = self.lambda_to_signal_ratio(lamb)
        noise_ratio = self.signal_ratio_to_noise_ratio(signal_ratio)
        mean = signal_ratio ** 0.5
        var = noise_ratio ** 0.5
        if rand_noise is None:
            rand_noise = self.sample_noise(batch_size=ori_image.size(0))
        return mean + (var ** 0.5) * rand_noise

    def forward(self, noisy_image, lamb):
        return self.model(noisy_image=noisy_image, lamb=lamb)

    def get_loss(self, ori_image):
        """
        "Take gradient step on $\nabla_{\theta}\Vert\mathbf{\epsilon}(\mathbf{z}_{\lambda}, c) - \mathbf{\epsilon}\Vert^{2}$"
        """
        # "Algorithm 1-4; $\lambda \sim p(\lambda)$"
        rand_lamb = self.sample_lambda(batch_size=ori_image.size(0))
        # "Algorithm 1-5; $\epsilon \sim \mathcal{N}(0, I)$"
        rand_noise = self.sample_noise(batch_size=ori_image.size(0))
        noisy_image = self.perform_diffusion_process(
            ori_image=ori_image, lamb=rand_lamb, rand_noise=rand_noise,
        )
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16,
        ) if self.device.type == "cuda" else contextlib.nullcontext():
            pred_noise = self(noisy_image=noisy_image, lamb=rand_lamb)
            return F.mse_loss(pred_noise, rand_noise, reduction="mean")

    @torch.inference_mode()
    def take_denoising_step(self, noisy_image, diffusion_step_idx):
        diffusion_step = self.batchify_diffusion_steps(
            diffusion_step_idx=diffusion_step_idx, batch_size=noisy_image.size(0),
        )
        alpha_t = self.index(self.alpha, diffusion_step=diffusion_step)
        beta_t = self.index(self.beta, diffusion_step=diffusion_step)
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        pred_noise = self(noisy_image=noisy_image.detach(), diffusion_step=diffusion_step)
        model_mean = (1 / (alpha_t ** 0.5)) * (
            noisy_image - ((beta_t / ((1 - alpha_bar_t) ** 0.5)) * pred_noise)
        )
        model_var = beta_t

        if diffusion_step_idx > 0:
            z_min_lamb = self.sample_noise(batch_size=noisy_image.size(0))
        else:
            z_min_lamb = torch.zeros(
                size=(noisy_image.size(0), self.image_channels, self.img_size, self.img_size),
                device=self.device,
            )
        return model_mean + (model_var ** 0.5) * z_min_lamb

    def perform_denoising_process(self, noisy_image, start_diffusion_step_idx, n_frames=None):
        """
        "$p_{\theta}(z_{\lambda'} \vert z_{\lambda})$"
        """
        if n_frames is not None:
            frames = list()

        x = noisy_image
        pbar = tqdm(range(start_diffusion_step_idx, -1, -1), leave=False)
        for diffusion_step_idx in pbar:
            pbar.set_description("Denoising...")

            x = self.take_denoising_step(x, diffusion_step_idx=diffusion_step_idx)

            if n_frames is not None and (
                diffusion_step_idx % (self.n_diffusion_steps // n_frames) == 0
            ):
                frames.append(self._get_frame(x))
        return frames if n_frames is not None else x

    def sample(self, batch_size):
        # "$p_{\theta}(z_{\lambda_{\text{min}}}) = \mathcal{N}(0, I)$"
        z_min_lamb = self.sample_noise(batch_size=batch_size)
        return self.perform_denoising_process(
            noisy_image=z_min_lamb,
            start_diffusion_step_idx=self.n_diffusion_steps - 1,
            n_frames=None,
        )
