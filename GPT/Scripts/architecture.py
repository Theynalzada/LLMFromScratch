# Importing Dependencies
import torch

# Making sure the GPU is being utilized
assert torch.cuda.is_available()

# Creating a custom Gaussian Error Linear Unit (GELU) activation function
class GELU(torch.nn.Module):
    # Defining instance attribute(s)
    def __init__(self):
        # Inheriting the methods and attributes
        super().__init__()
        
    # Defining the forward instance method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculating the activation function
        gelu=0.5*x*(1+torch.tanh(input=torch.sqrt(input=torch.tensor(2/torch.pi))*(x+(0.044715*torch.pow(input=x, exponent=3)))))
        
        # Returning the activation function
        return gelu
    
# Creating a custom layer normalization (LayerNorm)
class LayerNorm(torch.nn.Module):
    # Defining instance attribute(s)
    def __init__(self, emb_dim: int, eps: float=1e-5):
        # Inheriting the methods and attributes
        super().__init__()
        
        # Defining the shift parameter
        self.shift=torch.nn.Parameter(data=torch.zeros(emb_dim))
        
        # Defining the scale parameter
        self.scale=torch.nn.Parameter(data=torch.ones(emb_dim))
        
        # Defining the epsilon parameter
        self.eps=eps
        
    # Defining the forward instance method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculating the mean
        mean=x.mean(dim=-1, keepdim=True)
        
        # Calculating the variance
        var=x.var(dim=-1, unbiased=False, keepdim=True)
        
        # Calculating the normalized values
        norm_x=(x-mean)/torch.sqrt(input=var+self.eps)
        
        # Adding scale and shift parameters
        norm_x=(norm_x*self.scale)+self.shift
        
        # Returning the normalized values
        return norm_x
    
# Creating a custom feed forward neural network
class FeedForward(torch.nn.Module):
    # Defining instance attribute(s)
    def __init__(self, emb_dim: int):
        # Inheriting the methods and attributes
        super().__init__()
        
        # Defining the input layer
        input_layer=torch.nn.Linear(in_features=emb_dim, out_features=emb_dim*4)
        
        # Defining the output layer
        output_layer=torch.nn.Linear(in_features=emb_dim*4, out_features=emb_dim)
        
        # Instantiating the GELU activation function
        gelu=GELU()
        
        # Building sequential layers
        self.layers=torch.nn.Sequential(input_layer, gelu, output_layer)
        
    # Defining the forward instance method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculating the hidden states
        x=self.layers(x)
        
        # Returning the hidden states
        return x
    
# Creating a custom Multi Head Attention (MHA) mechanism
class MultiHeadAttention(torch.nn.Module):
    # Instantiating the attribute(s)
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 qkv_bias: bool,
                 context_length: int,
                 dropout_proba: float):
        # Inheriting the methods and attributes
        super().__init__()
        
        # Asserting the model dimension to be divisible by the number of attention heads
        assert d_model % n_heads == 0
        
        # Defining a causal masked attention mechanism
        self.register_buffer(name="mask", tensor=torch.triu(input=torch.ones(size=[context_length, context_length], dtype=torch.bool), diagonal=1))
        
        # Defining the query, key, and value vectors
        self.W_qkv=torch.nn.Linear(in_features=d_model, out_features=d_model*3, bias=qkv_bias)
        
        # Defining linear projection weights that are necessary for LLMs
        self.linear_proj=torch.nn.Linear(in_features=d_model, out_features=d_model)
        
        # Defining the regularization
        self.dropout=torch.nn.Dropout(p=dropout_proba)
        
        # Defining the number of dimensions based on attention heads
        self.head_dim=d_model // n_heads
        
        # Defining the model dimension
        self.d_model=d_model
        
        # Instantiating the number of attention heads
        self.n_heads=n_heads
        
    # Defining the forward instance method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unpacking the values
        n_batches, n_tokens, _ = x.shape
        
        # Calculating the weights
        qkv_weights=self.W_qkv(x)
        
        # Unpacking to extract the query, key, and value vectors separately
        queries, keys, values=torch.chunk(input=qkv_weights, chunks=3, dim=-1)
        
        # Reshaping the query vectors from [b, t, d] to [b, t, h, d]
        queries=queries.view(n_batches, n_tokens, self.n_heads, self.head_dim)
        
        # Reshaping the key vectors from [b, t, d] to [b, t, h, d]
        keys=keys.view(n_batches, n_tokens, self.n_heads, self.head_dim)
        
        # Reshaping the value vectors from [b, t, d] to [b, t, h, d]
        values=values.view(n_batches, n_tokens, self.n_heads, self.head_dim)
        
        # Transposing query vectors from [b, t, h, d] to [b, h, t, d]
        queries=queries.transpose(1, 2)
        
        # Transposing key vectors from [b, t, h, d] to [b, h, t, d]
        keys=keys.transpose(1, 2)
        
        # Transposing value vectors from [b, t, h, d] to [b, h, t, d]
        values=values.transpose(1, 2)
        
        # Calculating the attention scores
        attn_scores=queries @ keys.transpose(-2, -1)
        
        # Instantiating the masking
        masking=self.mask[:n_tokens, :n_tokens]
        
        # Masking the attention scores
        attn_scores_masked=attn_scores.masked_fill(mask=masking, value=-torch.inf)
        
        # Calculating the regularized attention weights
        attn_weights=self.dropout(torch.softmax(input=attn_scores_masked/self.head_dim**0.5, dim=-1))
        
        # Calculating the context vectors
        context_vectors=(attn_weights @ values).transpose(1, 2)
        
        # Reshaping the context vectors
        context_vectors=self.linear_proj(context_vectors.contiguous().view(n_batches, n_tokens, self.d_model))
        
        # Returning the context vectors
        return context_vectors
    
# Creating a custom Transformer Block
class TransformerBlock(torch.nn.Module):
    # Defining instance attribute(s)
    def __init__(self, config: dict):
        # Inheriting the methods and attributes
        super().__init__()
        
        # Creating a dictionary of keyword arguments
        llm_kwargs=dict(d_model=config["emb_dim"],
                        n_heads=config["n_heads"],
                        qkv_bias=config["qkv_bias"],
                        context_length=config["context_length"],
                        dropout_proba=config["drop_rate"])
        
        # Instantiating the Multi Head Attention (MHA) mechanism
        self.mha=MultiHeadAttention(**llm_kwargs)
        
        # Instantiating the Feed Forward Neural Network (FF)
        self.ff=FeedForward(emb_dim=config["emb_dim"])
        
        # Instantiating the layer normalization for MHA
        self.layer_norm_mha=LayerNorm(emb_dim=config["emb_dim"])
        
        # Instantiating the layer normalization for FF
        self.layer_norm_ff=LayerNorm(emb_dim=config["emb_dim"])
        
        # Defining the regularization
        self.dropout=torch.nn.Dropout(p=config["drop_rate"])
        
    # Defining the forward instance method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initializing the shortcut
        shortcut=x
        
        # Normalizing the values
        x=self.layer_norm_mha(x)
        
        # Calculating the context vectors
        x=self.mha(x)
        
        # Applying regularization
        x=self.dropout(x)
        
        # Updating the vectors
        x=x+shortcut
        
        # Reinitializing the shortcut
        shortcut=x
        
        # Normalizing the values
        x=self.layer_norm_ff(x)
        
        # Calculating the hidden states
        x=self.ff(x)
        
        # Applying regularization
        x=self.dropout(x)
        
        # Updating the vectors
        x=x+shortcut
        
        # Returning the hidden states
        return x
    
# Creating a custom GPT model
class GPTModel(torch.nn.Module):
    # Defining the instance attribute(s)
    def __init__(self, config: dict):
        # Inheriting the methods and attributes
        super().__init__()
        
        # Defining the token embedding layer
        self.tok_emb_layer=torch.nn.Embedding(num_embeddings=config["vocab_size"], embedding_dim=config["emb_dim"])
        
        # Defining the positional (absolute) embedding layer
        self.pos_emb_layer=torch.nn.Embedding(num_embeddings=config["context_length"], embedding_dim=config["emb_dim"])
        
        # Instantiating the transformer blocks based on the number of layers
        self.trf_blocks=torch.nn.Sequential(*[TransformerBlock(config=config) for _ in range(config["n_layers"])])
        
        # Defining the output head
        self.lm_head=torch.nn.Linear(in_features=config["emb_dim"], out_features=config["vocab_size"], bias=False)
        
        # Instantiating the final layer normalization
        self.final_layer_norm=LayerNorm(emb_dim=config["emb_dim"])
        
        # Defining the regularization
        self.dropout=torch.nn.Dropout(p=config["drop_rate"])
        
    # Defining the forward instance method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unpacking the values
        _, seq_length=x.shape
        
        # Calculating the token embeddings
        tok_embs=self.tok_emb_layer(x)
        
        # Calculating the positional (absolute) embeddings
        pos_embs=self.pos_emb_layer(torch.arange(seq_length, device=x.device))
        
        # Calculating the joint embeddings
        x=tok_embs+pos_embs
        
        # Applying regularization
        x=self.dropout(x)
        
        # Calculating the hidden states
        x=self.trf_blocks(x)
        
        # Applying normalization
        x=self.final_layer_norm(x)
        
        # Mapping the hidden states to vocabulary logits
        x=self.lm_head(x)
        
        # Returning the logits
        return x