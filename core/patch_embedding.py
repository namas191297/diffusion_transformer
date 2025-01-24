'''Author: Namas Bhandari
   Date: 22/01/25
   This script defines a Patch Embedding module for a Transformer. A Patch Embedding module takes in an image of size (H,W,C) and divides the image into
   patches of size PxP which are treated as tokens. These tokens are then embedded into an embedding vector space and passed onto the Encoder within the
   transformer. 
   If Image is of size (H,W,C) and a Patch is of size (PxP) then the number of patches will be equal to (H//P) * (W//P).
   So for example, an image of 256,256,3 with Patch Size of 16x16 would result in 256//16 * 256//16 = 16 * 16 = 256 patches of 16x16 size (without multiplying the channels). 
   Each patch of size 16x16x3 (PxPxC) is flattened into a 16*16*3 = 768(P*P*C) sized 1-d vector and passed into a linear layer than projects into an
   higher dimensional embedding vector space of 1-d size E (ex. 256). 
   Since transformers are order-agnostic, we will also be applying positional encoding to the patches in order to preserve the information about their relative 
   position in the input, allowing the model to understand the order of the tokens. Unlike CNN, Transformers do not preserve the spatial relationship between
   pixels since they are not aware of the spacial relationship between the tokens, hence, they need to be fed this information.
   '''

import torch.nn as nn
import torch

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=32, embedding_size=256, image_height=512, image_width=512, image_channels=3):
        super().__init__()
        
        # Define the fixed parameters for the Patch Embedding Module
        self.patch_size = patch_size
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels
        if self.image_height == self.image_width:
            self.no_patches = (self.image_height // self.patch_size) ** 2 # H // P * W // P, since H and W is same, we take it's square.
        else:
            self.no_patches = (self.image_height // self.patch_size) * (self.image_width // self.patch_size)
        self.embedding_size = embedding_size

        # Define the module layers
        self.embedding_layer = nn.Linear(self.patch_size * self.patch_size * self.image_channels, self.embedding_size)

        # The position encoding is a learnable parameter intialized with normal values (between -1 and 1) of size (1,number_of_patches,embedding_size).
        # This is due to the fact that we have N patches and each patch will be of a certain embedding size, so for each patch embedding, we will have a positional
        # encoding of the same size - which is embedding_size. Moreover, the batch size is always 1, since positional encoding will be shared across batches.
        self.positional_encoding = nn.Parameter(torch.randn(1,self.no_patches, self.embedding_size))

    def divide_image_into_patches(self, input_image):
        # First, we take a slice of the input image (C,H,W) across the height dimension. A stride of patch_size must be taken.
        patches = input_image.unfold(2, self.patch_size, self.patch_size) # (B,C,H,W) -> (B,C,H//PATCH_SIZE,PATCH_SIZE,W)
        # Second, we follow up by taking a slice along the width dimension.
        patches = patches.unfold(4, self.patch_size, self.patch_size) # (B,C,H//PATCH_SIZE,PATCH_SIZE,W//PATCH_SIZE, PATCH_SIZE)
        # Then we flatten everything and reshape it into (BatchSize, Channels, NumberOfPatches, PatchSize * PatchSize)
        patches = patches.contiguous().view(input_image.shape[0], self.image_channels, self.no_patches, self.patch_size * self.patch_size) # (B,C,H//PATCH_SIZE,PATCH_SIZE,W//PATCH_SIZE, PATCH_SIZE) -> (B,C,NUMBER_PATCHES, PATCH_SIZE * PATCH_SIZE)
        # Then we flatten out even more by multiplying the channels with the PatchSize * PatchSize resulting in (BatchSize, NumberOfPatches, PatchSize ** 2 * ImageChannels)
        patches = patches.permute(0,2,3,1).contiguous().view(input_image.shape[0], self.no_patches, -1)
        return patches

    def forward(self, input_image):
        # First, divide the input image into patches
        patches = self.divide_image_into_patches(input_image)
        # Then, we project these patches into a higher dimensional embedding space
        embeddings = self.embedding_layer(patches)
        # Then, we add the learnable position encodings onto the embeddings.
        embeddings += self.positional_encoding

        return embeddings
    

if __name__ == '__main__':
    example_input_image = torch.randn((1,3,256,256))
    patch_size = 32
    embedding_size = 512
    patch_embedding_module = PatchEmbedding(patch_size, embedding_size, image_height=256, image_width=256)
    print(f'Testing the PatchEmbedding Module with Patch Size:{patch_size}, Embedding Size:{embedding_size} and Input Image Size:{example_input_image.shape}')
    embeddings = patch_embedding_module(example_input_image)
    print(f'Embeddings generated:{embeddings.shape} containing Batch Size:{embeddings.shape[0]}, No. of Patches:{embeddings.shape[1]}, Embedding Size:{embeddings.shape[2]}')