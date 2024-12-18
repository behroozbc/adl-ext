import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672.
    """
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
    alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(beta_start, beta_end, timesteps, slimit=6):
    """
    Sigmoidal beta schedule - following a sigmoid function.
    """
    steps = torch.linspace(-slimit, slimit, timesteps)
    betas = beta_start + torch.sigmoid(steps) * (beta_end - beta_start)
    return betas



class Diffusion:
    
        # TODO (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps
        self.img_size = img_size
        self.device = device

        # Define beta schedule
        self.betas = get_noise_schedule(self.timesteps).to(self.device)
        
        # Calculate alphas and alpha_cumprod (cumulative product)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        # Precompute square roots for forward and reverse diffusion
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.one_by_sqrt_alpha = 1.0 / torch.sqrt(self.alphas)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, class_labels=None, w=0.0):
        """
        Perform a single reverse step: p(x_{t-1} | x_t)
        Equation (8): xt-1 = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1-alpha_cumprod) * epsilon_theta) + sqrt(beta_t) * z
        """
        # Predict the noise using the model
        # epsilon_theta = model(x, t)
        
        eps_theta_c = model(x, t, class_labels)
        eps_theta_uc = model(x, t, None)
        eps_theta = (1 + w) * eps_theta_c - w * eps_theta_uc
        
        
        beta_t = self.betas[t_index]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t_index]
        one_by_sqrt_alpha_t = self.one_by_sqrt_alpha[t_index]

        # Compute the mean of the reverse process
        # mean = one_by_sqrt_alpha_t * (x - beta_t / sqrt_one_minus_alpha_cumprod_t * epsilon_theta)
        mean = one_by_sqrt_alpha_t * (x - beta_t / sqrt_one_minus_alpha_cumprod_t * eps_theta)
        
        # Add noise z ~ N(0, I) unless at timestep 0
        if t_index > 0:
            z = torch.randn_like(x)
            mean += torch.sqrt(beta_t) * z

        return mean

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        """
        Full sampling process: Start from random noise and iteratively denoise.
        """
        x = torch.randn(batch_size, channels, image_size, image_size).to(self.device)

        for t_index in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), t_index, dtype=torch.long, device=self.device)
            x = self.p_sample(model, x, t, t_index)
        
        return x

    def q_sample(self, x_zero, t, noise=None):
        """
        Forward diffusion process: Generate a noisy sample x_t at timestep t from x_0.
        Equation (4): x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1-alpha_cumprod_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_zero)
        
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]

        return sqrt_alpha_cumprod_t * x_zero + sqrt_one_minus_alpha_cumprod_t * noise

    def p_losses(self, denoise_model, x_zero, t,class_labels=None, noise=None, loss_type="l1"):
        """
        Compute the loss for the denoising process.
        """
        if noise is None:
            noise = torch.randn_like(x_zero)
        
        x_noisy = self.q_sample(x_zero, t, noise)
        predicted_noise = denoise_model(x_noisy, t, class_labels)
        
        if loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(predicted_noise, noise)
        else:
            raise NotImplementedError("Unknown loss type")
        
        return loss




# import torch
# import torch.nn.functional as F
# from ex02_helpers import extract
# from tqdm import tqdm

# def linear_beta_schedule(beta_start, beta_end, timesteps):
#     """
#     Standard linear beta/variance schedule as proposed in the original paper.
#     """
#     return torch.linspace(beta_start, beta_end, timesteps)

# def cosine_beta_schedule(timesteps, s=0.008):
#     """
#     Cosine schedule as proposed in https://arxiv.org/abs/2102.09672.
#     """
#     steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
#     alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return torch.clip(betas, 0, 0.999)

# def sigmoid_beta_schedule(beta_start, beta_end, timesteps, slimit=6):
#     """
#     Sigmoidal beta schedule - following a sigmoid function.
#     """
#     steps = torch.linspace(-slimit, slimit, timesteps)
#     betas = beta_start + torch.sigmoid(steps) * (beta_end - beta_start)
#     return betas

# class Diffusion:

#     def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
#         """
#         Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
#         """
#         self.timesteps = timesteps
#         self.img_size = img_size
#         self.device = device

#         # Define beta schedule
#         self.betas = get_noise_schedule(self.timesteps).to(self.device)
        
#         # Calculate alphas and alpha_cumprod (cumulative product)
#         self.alphas = 1.0 - self.betas
#         self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

#         # Precompute square roots for forward and reverse diffusion
#         self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
#         self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
#         self.one_by_sqrt_alpha = 1.0 / torch.sqrt(self.alphas)

#     @torch.no_grad()
#     def p_sample(self, model, x, t, t_index):
#         """
#         Perform a single reverse step: p(x_{t-1} | x_t).
#         Equation (8): xt-1 = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1-alpha_cumprod) * epsilon_theta) + sqrt(beta_t) * z
#         """
#         # Predict the noise using the model
#         epsilon_theta = model(x, t)

#         # Use `extract` for batch-compatible operations
#         beta_t = extract(self.betas, t, x.shape)
#         sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alpha_cumprod, t, x.shape)
#         one_by_sqrt_alpha_t = extract(self.one_by_sqrt_alpha, t, x.shape)

#         # Compute the mean of the reverse process
#         mean = one_by_sqrt_alpha_t * (x - beta_t / sqrt_one_minus_alpha_cumprod_t * epsilon_theta)

#         # Add noise z ~ N(0, I) unless at timestep 0
#         if t_index > 0:
#             z = torch.randn_like(x)
#             mean += torch.sqrt(beta_t) * z

#         return mean

#     @torch.no_grad()
#     def sample(self, model, image_size, batch_size=16, channels=3):
#         """
#         Full sampling process: Start from random noise and iteratively denoise.
#         """
#         x = torch.randn(batch_size, channels, image_size, image_size).to(self.device)

#         for t_index in reversed(range(0, self.timesteps)):
#             t = torch.full((batch_size,), t_index, dtype=torch.long, device=self.device)
#             x = self.p_sample(model, x, t, t_index)
        
#         return x

#     def q_sample(self, x_zero, t, noise=None):
#         """
#         Forward diffusion process: Generate a noisy sample x_t at timestep t from x_0.
#         Equation (4): x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1-alpha_cumprod_t) * noise
#         """
#         if noise is None:
#             noise = torch.randn_like(x_zero)

#         # Use `extract` for batch-compatible values
#         sqrt_alpha_cumprod_t = extract(self.sqrt_alpha_cumprod, t, x_zero.shape)
#         sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alpha_cumprod, t, x_zero.shape)

#         return sqrt_alpha_cumprod_t * x_zero + sqrt_one_minus_alpha_cumprod_t * noise

#     def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1"):
#         """
#         Compute the loss for the denoising process.
#         """
#         if noise is None:
#             noise = torch.randn_like(x_zero)

#         # Generate noisy input using q_sample
#         x_noisy = self.q_sample(x_zero, t, noise)
#         predicted_noise = denoise_model(x_noisy, t)

#         # Calculate loss
#         if loss_type == 'l1':
#             loss = F.l1_loss(predicted_noise, noise)
#         elif loss_type == 'l2':
#             loss = F.mse_loss(predicted_noise, noise)
#         else:
#             raise NotImplementedError("Unknown loss type")

#         return loss
