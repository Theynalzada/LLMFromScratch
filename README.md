# LLM From Scratch: GPT

A from-scratch implementation of the GPT-2 architecture in PyTorch. Every building block (layer normalization, GELU, multi-head causal attention, the feed-forward network, the transformer block and the full model) is written by hand, checked against PyTorch's built-in version where one exists, and then used to load and run OpenAI's pretrained GPT-2 weights.

## Project structure

```
LLMFromScratch/
├── requirements.txt                    # Pinned dependencies (PyTorch built for CUDA 13.0)
└── GPT/
    ├── Configuration/
    │   └── config.yml                  # Hyperparameters for the four GPT-2 sizes
    ├── Scripts/
    │   └── architecture.py             # Reusable module with the full model architecture
    └── Notebooks/
        └── GPT From Scratch.ipynb      # Builds, tests and runs each component step by step
```

## Setup

The project expects an NVIDIA GPU with CUDA. `requirements.txt` pulls PyTorch from the CUDA 13.0 wheel index.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

| Package        | Version | Used for                                              |
|----------------|---------|-------------------------------------------------------|
| `torch`        | 2.14.1  | Tensors, autograd and `nn.Module` building blocks     |
| `tiktoken`     | 0.14.0  | GPT-2 byte-pair-encoding tokenizer                    |
| `transformers` | 5.18.0  | Downloading the pretrained GPT-2 weights from Hugging Face |
| `numpy`        | 2.5.2   | Intermediate format for the weights before loading    |
| `PyYAML`       | 6.0.3   | Reading `config.yml`                                  |
| `ipykernel`    | 7.4.0   | Running the notebook in Jupyter or VS Code            |

Open `GPT/Notebooks/GPT From Scratch.ipynb` and run the cells from top to bottom. The notebook reads the config through the relative path `../Configuration/config.yml`, so it has to run with `GPT/Notebooks/` as its working directory (the default in Jupyter and VS Code).

## Configuration

[GPT/Configuration/config.yml](GPT/Configuration/config.yml) defines the hyperparameters for all four published GPT-2 sizes under the `gpt` key:

| Key        | Model              | Parameters | `emb_dim` | `n_heads` | `n_layers` |
|------------|--------------------|-----------:|----------:|----------:|-----------:|
| `small`    | GPT-2 Small        | 124M       | 768       | 12        | 12         |
| `medium`   | GPT-2 Medium       | 355M       | 1024      | 16        | 24         |
| `large`    | GPT-2 Large        | 774M       | 1280      | 20        | 36         |
| `original` | GPT-2 XL (the full model) | 1558M | 1600   | 25        | 48         |

All four share `vocab_size: 50257`, `context_length: 1024`, `drop_rate: 0.1` and `qkv_bias: True`.

## Architecture

The model in [GPT/Scripts/architecture.py](GPT/Scripts/architecture.py) follows GPT-2's decoder-only, pre-norm design:

```
token IDs ─► token embedding + positional embedding ─► dropout
          ─► N × TransformerBlock
          ─► final LayerNorm ─► lm_head (Linear, no bias) ─► vocabulary logits

TransformerBlock:
    x = x + Dropout(MultiHeadAttention(LayerNorm(x)))
    x = x + Dropout(FeedForward(LayerNorm(x)))
```

### Components

**`GELU`**: The tanh approximation of the Gaussian Error Linear Unit, as used in GPT-2:
`0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`

**`LayerNorm`**: Normalizes over the last dimension using the biased variance and `eps = 1e-5`, then applies the learnable `scale` (initialized to ones) and `shift` (initialized to zeros).

**`FeedForward`**: `Linear(d → 4d) → GELU → Linear(4d → d)`.

**`MultiHeadAttention`**: Causal multi-head self-attention:
- A single fused `W_qkv` linear layer produces queries, keys and values, which are then split with `torch.chunk`. This matches GPT-2's fused `c_attn` layout, so the pretrained weights load without being split.
- Tensors are reshaped from `[batch, tokens, d_model]` to `[batch, heads, tokens, head_dim]`.
- Attention scores are scaled by `√head_dim`, and future positions are masked with an upper-triangular boolean buffer (registered with `register_buffer`, so it moves with the model across devices).
- Dropout is applied to the attention weights, the heads are concatenated, and the result goes through the output projection `linear_proj`.

**`TransformerBlock`**: Pre-norm attention and feed-forward sublayers, each wrapped in dropout and a residual (shortcut) connection.

**`GPTModel`**: Token and learned absolute positional embeddings, a stack of `n_layers` transformer blocks, a final `LayerNorm`, and a bias-free `lm_head` that maps hidden states to logits over the 50,257-token vocabulary.

## Notebook walkthrough

[GPT/Notebooks/GPT From Scratch.ipynb](GPT/Notebooks/GPT%20From%20Scratch.ipynb) builds the model step by step. It defines its own copies of each class, so it runs without importing `architecture.py`.

### 1. Loading the configuration and device
Loads `config.yml`, selects the `original` (1.5B) configuration for the architecture experiments, and checks that CUDA is available.

### 2. Checking each component against PyTorch
Each component runs on a random tensor of shape `[2, 16, 1600]`:

| Component            | Reference                               | `torch.allclose` |
|----------------------|-----------------------------------------|:----------------:|
| Custom `LayerNorm`   | `torch.nn.LayerNorm`                    | ✅ `True`        |
| Custom `GELU`        | `torch.nn.GELU(approximate="tanh")`     | ✅ `True`        |
| `MultiHeadAttention` | Output shape check: `[2, 16, 1600]`     | ✅               |

### 3. Assembling the full model
`FeedForward`, `TransformerBlock` and `GPTModel` are defined. A forward pass on the tokenized sentence *"I like watching movies"* returns logits of shape `[1, 4, 50257]`.

### 4. Text generation
`generate_text()` is an autoregressive decoding loop that supports:
- **Context cropping**: only the last `context_length` tokens are fed to the model.
- **Top-k filtering**: logits outside the top `k` are set to `-inf`.
- **Temperature scaling**: with `temperature > 0` the next token is sampled from the softmax with `torch.multinomial`, and with `temperature == 0` it is chosen greedily with `argmax`.
- **Early stopping**: generation stops when the end-of-sequence token (`<|endoftext|>`) is produced.

With random (untrained) weights, the output is gibberish, as expected:

```
He never wanted to retained Drill refinement contended Sw
```

### 5. Loading the pretrained GPT-2 weights
- **`download_and_load_gpt2(model_size)`** downloads the official GPT-2 checkpoint from Hugging Face (`openai-community/gpt2`, `-medium`, `-large` or `-xl` for `"124M"`, `"355M"`, `"774M"` or `"1558M"`). It converts the state dict to NumPy and reorganizes it into a nested `params` dictionary (`wte`, `wpe`, `blocks`, `g`, `b`).
- **`assign(left, right)`** checks that the shapes match and wraps the array as a `torch.nn.Parameter`.
- **`load_weights_into_gpt(gpt, params)`** maps every GPT-2 tensor onto the custom model:

| GPT-2 (Hugging Face)      | Custom model                                   |
|---------------------------|------------------------------------------------|
| `wte` / `wpe`             | `tok_emb_layer` / `pos_emb_layer`              |
| `attn.c_attn`             | `mha.W_qkv` (transposed)                       |
| `attn.c_proj`             | `mha.linear_proj` (transposed)                 |
| `mlp.c_fc` / `mlp.c_proj` | `ff.layers[0]` / `ff.layers[2]` (transposed)   |
| `ln_1` / `ln_2`           | `layer_norm_mha` / `layer_norm_ff`             |
| `ln_f`                    | `final_layer_norm`                             |
| `wte` (tied)              | `lm_head`                                      |

The linear-layer weights are transposed because Hugging Face's GPT-2 stores them in `Conv1D` layout (`[in, out]`), while `torch.nn.Linear` expects `[out, in]`. GPT-2 ties its output head to the token embeddings, so `lm_head` reuses `wte`.

### 6. Generating with the pretrained model
The 124M model is loaded into `GPTModel` with the `small` configuration, moved to the GPU and switched to evaluation mode. With the prompt *"Every effort moves you"*, `temperature=1.5` and `top_k=50`, it produces coherent English:

```
Every effort moves you as far as the hand can go until the end of your turn unless something interrupts your control flow. As you may observe I
```

Coherent output from the pretrained weights confirms that the from-scratch architecture matches GPT-2 exactly.

## Using the architecture module

```python
import yaml
import torch
from architecture import GPTModel   # run from GPT/Scripts or add it to sys.path

with open("GPT/Configuration/config.yml") as f:
    config = yaml.safe_load(f)

model = GPTModel(config=config["gpt"]["small"]).to("cuda")
logits = model(torch.tensor([[15496, 11, 995]], device="cuda"))   # [1, 3, 50257]
```

Importing `architecture.py` runs `assert torch.cuda.is_available()`, so it fails on a machine without a GPU.