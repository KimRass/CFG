import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import contextlib

from unet import UNet


class CFGDiffusion(nn.Module):
    def __init__(
        self,
        net,
        img_size,
        device,
        min_lamb=-20, # "$\lambda_{\text{min}}$"
        max_lamb=20, # "$\lambda_{\text{max}}$"
        interpol_coeff=0.3, # "$v$"
        uncond_prob=0.1,
        guidance_str=0.1,
        image_channels=3,
        n_diffusion_steps=1000,
    ):
        # "We sample $\lambda$ via \lambda = -2\log\tan(au + b) for uniformly distributed $u \in [0, 1]$,
        # where $b = \arctan(e^{-\lambda_{\text{max}} / 2})$
        # and $a = \arctan(e^{-\lambda_{\text{max}} / 2}) - b$."
        super().__init__()

        self.img_size = img_size
        self.device = device
        self.min_lambda = min_lamb
        self.max_lambda = max_lamb
        self.interpol_coeff = interpol_coeff
        self.uncond_prob = uncond_prob
        self.guidance_str = guidance_str
        self.image_channels = image_channels
        self.n_diffusion_steps = n_diffusion_steps

        self.net = net.to(device)

        self.diffusion_step = torch.linspace(
            min_lamb, max_lamb, n_diffusion_steps, device=self.device,
        )

        self.b = torch.arctan(
            torch.exp(torch.tensor(-max_lamb / 2, device=self.device))
        )
        self.a = torch.arctan(
            torch.exp(torch.tensor(-min_lamb / 2, device=self.device))
        ) - self.b

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
        "$q(\mathbf{z}_{\lambda} \vert \mathbf{x}) = \mathcal{N}(\alpha_{\lambda}\mathbf{x}, \sigma^{2}_{\lambda}\mathbf{I})$"
        "$\mathbf{z}_{\lambda} = \alpha_{\lambda}\mathbf{x} + \sigma_{\lambda}\mathbf{\epsilon}$"
        """
        signal_ratio = self.lambda_to_signal_ratio(lamb)
        noise_ratio = self.signal_ratio_to_noise_ratio(signal_ratio)
        mean = signal_ratio ** 0.5
        var = noise_ratio ** 0.5
        if rand_noise is None:
            rand_noise = self.sample_noise(batch_size=ori_image.size(0))
        return mean + (var ** 0.5) * rand_noise

    def batchify_diffusion_steps(self, diffusion_step_idx, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=diffusion_step_idx,
            dtype=torch.long,
            device=self.device,
        )

    def forward(self, noisy_image, diffusion_step, label):
        return self.unet(
            noisy_image=noisy_image, diffusion_step=diffusion_step, label=label,
        )

    def predict_noise(self, noisy_image, diffusion_step_idx, label):
        diffusion_step = self.batchify_diffusion_steps(
            diffusion_step_idx=diffusion_step_idx, batch_size=noisy_image.size(0),
        )
        pred_noise_cond = self(
            noisy_image=noisy_image, diffusion_step=diffusion_step, label=label,
        )
        pred_noise_uncond = self(
            noisy_image=noisy_image, diffusion_step=diffusion_step, label=,
        )
        return (1 + self.guidance_str) * pred_noise_cond - self.guidance_str * pred_noise_uncond

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
            pred_noise = self.predict_noise(noisy_image=noisy_image, lamb=rand_lamb)
            return F.mse_loss(pred_noise, rand_noise, reduction="mean")

    @torch.inference_mode()
    def predict_ori_image(self, noisy_image, noise, signal_ratio, noise_ratio):
        """
        "$\mathbf{x}_{\theta}(\mathbf{z}_{\lambda}) = (\mathbf{z}_{\lambda} - \sigma_{\lambda}\mathbf{\epsilon}_{\theta}(\mathbf{z}_{\lambda})) / \alpha_{\lambda}$"
        """
        return (noisy_image - (noise_ratio ** 0.5) * noise) / (signal_ratio ** 0.5)

    @torch.inference_mode()
    def take_denoising_step(self, noisy_image, diffusion_step_idx, label):
        """
        "$p_{\theta}(\mathbf{z}_{\lambda'} \vert z_{\lambda})$"
        """
        lamb1 = self.diffusion_step[diffusion_step_idx]
        lamb2 = self.diffusion_step[diffusion_step_idx - 1]

        signal_ratio1 = self.lambda_to_signal_ratio(lamb1)
        noise_ratio1 = self.signal_ratio_to_noise_ratio(signal_ratio1)
        pred_noise = self.predict_noise(
            noisy_image=noisy_image, diffusion_step_idx=diffusion_step_idx, label=label.to(self.device),
        )
        pred_ori_image  = self.predict_ori_image(
            noisy_image=noisy_image,
            noise=pred_noise,
            signal_ratio=signal_ratio1,
            noise_ratio=noise_ratio1,
        )

        signal_ratio2 = self.lambda_to_signal_ratio(lamb2)
        exp_lamb12 = torch.exp(lamb1 - lamb2)
        # "$\tilde{\mathbf{\mu}}_{\lambda' \vert \lambda}(\mathbf{z}_{\lambda}, \mathbf{x}) = e^{\lambda - \lambda'}(\alpha_{\lambda'}/\alpha_{\lambda})\mathbf{z}_{\lambda} + (1 - e^{\lambda - \lambda'})\alpha_{\lambda'}\mathbf{x}$"
        model_mean = exp_lamb12 * (
            (signal_ratio2 ** 0.5) / (signal_ratio1 ** 0.5)
        ) * noisy_image + (1 - exp_lamb12) * (signal_ratio2 ** 0.5) * pred_ori_image

        noise_ratio2 = self.signal_ratio_to_noise_ratio(signal_ratio2)
        sigma_square21 = (1 - exp_lamb12) * noise_ratio2
        sigma_square12 = (1 - exp_lamb12) * noise_ratio1
        model_var = (sigma_square21 ** (
            1 - self.interpol_coeff
        )) * (sigma_square12 ** self.interpol_coeff)
        return model_mean + (model_var ** 0.5) * noisy_image

    def perform_denoising_process(self, noisy_image, start_diffusion_step_idx, label):
        x = noisy_image
        pbar = tqdm(range(start_diffusion_step_idx, 0, -1), leave=False)
        for diffusion_step_idx in pbar:
            pbar.set_description("Denoising...")

            x = self.take_denoising_step(
                noisy_image=x, diffusion_step_idx=diffusion_step_idx, label=label,
            )
        return x

    def sample(self, label):
        # "$p_{\theta}(z_{\lambda_{\text{min}}}) = \mathcal{N}(0, I)$"
        z_min_lamb = self.sample_noise(batch_size=label.size(0))
        return self.perform_denoising_process(
            noisy_image=z_min_lamb,
            start_diffusion_step_idx=self.n_diffusion_steps - 1,
            label=label,
        )


if __name__ == "__main__":
    N_CLASSES = 10
    IMG_SIZE = 32
    DEVICE = torch.device("mps")
    net = UNet(n_classes=N_CLASSES)
    model = CFGDiffusion(
        net=net,
        img_size=IMG_SIZE,
        device=DEVICE,
    )

    BATCH_SIZE = 1
    LABEL = torch.randint(0, N_CLASSES, size=(BATCH_SIZE,))
    model.sample(label=LABEL)
