'''Author: Namas Bhandari
   Date: 29/01/25
   This script defines a Time Embedding module for a Transformer. A Time Embedding module takes in the patch embeddings generated via the Patch Embedding module
   and encodes time embeddings so that the model is able to understand which time step t of the diffusion process is being processed by the transformer - 
   allowing it to understand the intensity of denoising in the diffusion process. 
   This Time Embedding module will be a Hybrid approach that combines sinusoidal encodings along with a MLP. 
   In the diffusion process, for a single image (with n patches), we have a time step t that adds a certain amount of noise. Therefore, the Time Embedding module must generate a tensor of shape 
   (1, num_patches, embedding_dim) for each sample in the batch to encode that diffusion time step within that image for the transformer to understand which 
   time step t is being denoised.
   '''

import torch
import math
import numpy as np
import torch.nn as nn
from patch_embedding import PatchEmbedding

class HybridTimeEmbedding(nn.Module):
    def __init__(self, embedding_size=256, hidden_dim_multiplier=4):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_dim_size = embedding_size * hidden_dim_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_dim_size),
            nn.GELU(),
            nn.Linear(self.hidden_dim_size, self.embedding_size)
        )
        self.half_dim = self.embedding_size // 2 # We obtain the half-dim since half embedding values are sin values and half are cosine values.

    def generate_sinusoidal_embedding(self, t):
        
        # Convert the time steps which are integers to float32 value - time steps are same as the batch size (batch_size,)
        t = t.to(dtype=torch.float32).unsqueeze(1) # (batch_size,) -> (batch_size, 1)
        w_i = torch.exp(-math.log(10000.0) / (self.half_dim - 1) * torch.arange(self.half_dim, device=t.device, dtype=torch.float32)) # These are frequences for each of these half dim positions.
        t_w_i = t * w_i # (batch_size, half_dim) -> we multiply the time steps t with the frequencies
        sin_t_w_i = torch.sin(t_w_i) # (batch_size, half_dim) -> This generates unique multi-scale representation of time step t for half-dim positions using sin function.
        cos_t_w_i = torch.cos(t_w_i) # (batch_size, half_dim) -> This generates unique multi-scale representation of time step t for half-dim positions using cos function.
        sinusoidal_embedding = torch.cat([sin_t_w_i, cos_t_w_i], dim=1) # Concatenate the half dims to create the original embedding_dim (batch_size, embedding_dim).
        return sinusoidal_embedding


    def forward(self, patch_embeddings, t):
        
        # First, we obtain the number of patches for the patch embeddings.
        batch_size = patch_embeddings.shape[0]
        number_of_patches = patch_embeddings.shape[1]
        # Now, we generate sinusoidal embeddings from time steps t of size (batch_size, )
        sinusoidal_embeddings = self.generate_sinusoidal_embedding(t)
        # Repeat the sinusoidal embeddings across the number_of_patches dimensions to generate a vector with the same shape as the patch_embeddings
        sinusoidal_embeddings = sinusoidal_embeddings.unsqueeze(dim=1).expand((batch_size, number_of_patches, self.embedding_size))
        # Add the sinusoidal embeddings to the patch_embeddings to add the time step t information to the embeddings.
        patch_embeddings += sinusoidal_embeddings
        # Now, we pass these embeddings with encoded time step t information into the MLP to generate activations.
        activations = self.mlp(patch_embeddings)
        return activations
    
if __name__ == '__main__':
    example_input_image = torch.randn((4,3,256,256))
    patch_size = 32
    embedding_size = 512
    patch_embedding_module = PatchEmbedding(patch_size, embedding_size, image_height=256, image_width=256)
    patch_embeddings = patch_embedding_module(example_input_image)
    time_step = torch.from_numpy(np.asarray([3,50,4,10])).to(dtype=torch.int8)
    time_embedding_module = HybridTimeEmbedding(embedding_size=embedding_size)
    time_embeddings = time_embedding_module(patch_embeddings, time_step)
    print(f'Generated Time Embeddings of shape:{time_embeddings.shape} from Patch Embeddings of shape:{patch_embeddings.shape} with time-step array:{time_step}')