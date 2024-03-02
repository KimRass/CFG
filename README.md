# 1. Theoretical Background
$$\lambda = -2\log\tan(au + b)$$
$$b = \arctan(e^{-\lambda_{\text{max}} / 2})$$
$$a = \arctan(e^{-\lambda_{\text{max}} / 2}) - b$$
$$\alpha^{2}_{\lambda} = 1 / (1 + e^{-\lambda}), \sigma^{2}_{\lambda} = 1 - \alpha^{2}_{\lambda}$$
$$p_{\theta}(z_{\lambda_{\text{min}}}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$$
$$\lambda < \lambda'$$
$\lambda$가 커지는 방향이 Denoising process에 해당합니다.
## 1) Forward Diffusion Process
$$\mathbf{z}_{\lambda} = \alpha_{\lambda}\mathbf{x} + \sigma_{\lambda}\mathbf{\epsilon}$$
## 2) Backward Denoising Process
$$p_{\theta}(z_{\lambda'} \vert z_{\lambda}) = \mathcal{N}(\tilde{\mu}_{\lambda' \vert \lambda}$$
