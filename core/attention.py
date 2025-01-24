'''Author: Namas Bhandari
   Date: 23/01/25
   This script defines the self-attention mechanism for the Encoder part of the Transformer. The attention mechanism allows the Encoder to assign importance to
   each patch of the input image and helps the model capture the relationship between different patches. The reason why multiple heads are used is because
   it allows the Encoder to learn different relationships simultaneously - it attends to multiple representation subspaces.
   Example:
   One head focuses on the relationship between the dog and the leash.
   Another head captures the interaction between the tree and the sky.
   A third head might focus on the relationship between the ball and the ground.
'''

import torch.nn as nn
import torch

class SingleHeadedAttention(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.embedding_size = embedding_size

        # Initialize the W_K, W_Q and W_V linear layers for the Key, Query and Value.
        self.W_K = nn.Linear(self.embedding_size, self.embedding_size)
        self.W_Q = nn.Linear(self.embedding_size, self.embedding_size)
        self.W_V = nn.Linear(self.embedding_size, self.embedding_size)

        # Initialize the final linear layer within the Attention Head
        self.linear_out = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, patch_embeddings):
        
        # First, the patch embeddings are used to generate the Key, Query and Values.
        key = self.W_K(patch_embeddings) #(BatchSize,NumberOfPatches,EmbeddingSize) -> (BatchSize,NumberOfPatches,EmbeddingSize)
        query = self.W_Q(patch_embeddings) #(BatchSize,NumberOfPatches,EmbeddingSize) -> (BatchSize,NumberOfPatches,EmbeddingSize)
        value = self.W_V(patch_embeddings) #(BatchSize,NumberOfPatches,EmbeddingSize) -> (BatchSize,NumberOfPatches,EmbeddingSize)

        # Implement self-attention mechanism
        key_transpose = key.transpose(1,2) # K.T (BatchSize,NumberOfPatches,EmbeddingSize) -> (BatchSize,EmbeddingSize,NumberOfPatches)
        query_key_product = query @ key_transpose # Q @ K.T (BatchSize, NumberOfPatches, EmbeddingSize) @ (BatchSize, Embedding Size, NumberOfPatches) -> (BatchSize, NumberOfPatches, NumberofPatches)
        query_key_product = query_key_product/(self.embedding_size ** 0.5) # (Q @ K.T)/sqrt(head_dimension) -> since single head attention, head_dimension = embedding_size -> (BatchSize, NumberOfPatches, NumberofPatches)
        attention_scores = torch.nn.functional.softmax(query_key_product, dim=2) # Apply softmax to the last dimension, over the keys for each query. This gives us the score/importance for each patch and all corresponding patches -> (BatchSize, NumberOfPatches, NumberofPatches)
        output = attention_scores @ value # (AttentionScores @ Value) -> (BatchSize,NumberOfPatches,NumberOfPatches) @ (BatchSize, NumberOfPatches, NumberofPatches) -> (BatchSize, NumberOfPatches, EmbeddingSize)
        output = self.linear_out(output) # Final linear layer. (BatchSize,NumberOfPatches,EmbeddingSize)
        
        return output
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, embedding_size=512, number_of_heads=8):
        super().__init__()
        self.embedding_size = embedding_size
        self.number_of_heads = number_of_heads
        self.head_dim = embedding_size // number_of_heads

        # Initialize the W_K, W_Q and W_V linear layers for the Key, Query and Value.
        self.W_K = nn.Linear(self.embedding_size, self.embedding_size)
        self.W_Q = nn.Linear(self.embedding_size, self.embedding_size)
        self.W_V = nn.Linear(self.embedding_size, self.embedding_size)

        # Initialize the final linear layer within the Attention Head
        self.linear_out = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, patch_embeddings):
        
        # First, the patch embeddings are used to generate the Key, Query and Values.
        key = self.W_K(patch_embeddings) #(BatchSize,NumberOfPatches,EmbeddingSize) -> (BatchSize,NumberOfPatches,EmbeddingSize)
        query = self.W_Q(patch_embeddings) #(BatchSize,NumberOfPatches,EmbeddingSize) -> (BatchSize,NumberOfPatches,EmbeddingSize)
        value = self.W_V(patch_embeddings) #(BatchSize,NumberOfPatches,EmbeddingSize) -> (BatchSize,NumberOfPatches,EmbeddingSize)

        # Now, we divide the key, query and values into different heads.
        key = key.contiguous().view(key.shape[0], key.shape[1], self.number_of_heads, self.head_dim) #(BatchSize,NumberOfPatches,EmbeddingSize) -> (BatchSize,NumberOfPatches,NumberOfHeads,HeadDim)
        query = query.contiguous().view(query.shape[0], query.shape[1], self.number_of_heads, self.head_dim) #(BatchSize,NumberOfPatches,EmbeddingSize) -> (BatchSize,NumberOfPatches,NumberOfHeads,HeadDim)
        value = value.contiguous().view(value.shape[0], value.shape[1], self.number_of_heads, self.head_dim) #(BatchSize,NumberOfPatches,EmbeddingSize) -> (BatchSize,NumberOfPatches,NumberOfHeads,HeadDim)

        # Now, we transpose the dimension for the NumberOfPatches and the NumberOfHeads so that we can obtain the attention scores.
        key = key.transpose(1,2) #(BatchSize,NumberOfPatches,NumberOfHeads,HeadDim) -> (BatchSize,NumberOfHeads,NumberOfPatches,HeadDim)
        query = query.transpose(1,2) #(BatchSize,NumberOfPatches,NumberOfHeads,HeadDim) -> (BatchSize,NumberOfHeads,NumberOfPatches,HeadDim)
        value = value.transpose(1,2) #(BatchSize,NumberOfPatches,NumberOfHeads,HeadDim) -> (BatchSize,NumberOfHeads,NumberOfPatches,HeadDim)

        # Now, same as the SingleHeadAttention block, we implement the self-attention mechanism across multiple heads.
        query_key_product = query @ key.transpose(2,3) # Q @ K.T (BatchSize,NumberOfHeads,NumberOfPatches,HeadDim) @ (BatchSize,NumberOfHeads,HeadDim,NumberOfPatches) -> (BatchSize,NumberOfHeads,NumberOfPatches,NumberOfPatches)
        query_key_product = query_key_product/(self.head_dim ** 0.5) # (Q @ K.T)/sqrt(head_dimension) (BatchSize,NumberOfHeads,NumberOfPatches,NumberOfPatches)
        attention_scores = torch.nn.functional.softmax(query_key_product, dim=3) #Apply softmax to the last dimension, over the keys for each query, for each respective head. This gives us the score/importance for each patch and all corresponding patches for each head -> (BatchSize,NumberOfHeads,NumberOfPatches,NumberOfPatches)
        output = attention_scores @ value # (AttentionScores @ Value) -> (BatchSize,NumberOfHeads,NumberOfPatches,NumberOfPatches) @ (BatchSize,NumberOfHeads,NumberOfPatches,HeadDim) -> (BatchSize,NumberOfHeads,NumberOfPatches,HeadDim)
        output = output.transpose(1,2) #(BatchSize,NumberOfHeads,NumberOfPatches,HeadDim) -> (BatchSize,NumberOfPatches,NumberOfHeads,HeadDim)

        # Now, we concatenate all the heads to to obtain the original embedding size.
        output = output.contiguous().view(output.shape[0], output.shape[1], self.embedding_size)
        output = self.linear_out(output)

        return output
    
if __name__ == '__main__':
    num_patches = 32
    embedding_size = 512
    num_heads = 8
    example_patch_embeddings = torch.randn((3,num_patches,embedding_size)) # (Batch, NumberOfPatches, EmbeddingSize)
    mha = MultiHeadedAttention(embedding_size, num_heads)
    print(f'Testing the MultiHeadAttention Module with Num Heads:{num_heads}, Embedding Size:{embedding_size} and Patch Embeddings Size:{example_patch_embeddings.shape}')
    features = mha(example_patch_embeddings)
    print(f'Attention output:{features.shape} containing Batch Size:{features.shape[0]}, No. of Patches:{features.shape[1]}, Embedding Size:{features.shape[2]}')



