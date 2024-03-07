- Auxiliary classifier 없이 Low-temperature sampling을 하고자 합니다.

# 기존 연구
- Classifier guidance

# Classifier-Free Guidance
- $z_{\lambda}$: Noisy image (Same as $x_{t}$)
## Forward (diffusion) process
- $\lambda$가 감소하는 방향 ($\lambda < \lambda'$)
- $q(z_{\lambda} \vert z_{\lambda'})$
## Backward (Denoising) Process
- $\lambda$가 증가하는 방향
- $p_{\theta}$
- $\epsilon_{\theta}(z_{\lambda})$: Model
