# Importing Dependencies
from transformers import GPT2Model
import torch

# Defining a function to generate text
def generate_text(model,
                  token_ids,
                  max_new_tokens,
                  context_length,
                  temperature=0.0,
                  top_k=None,
                  eos_id=None):
    # Looping through the range of maximum new tokens
    for _ in range(max_new_tokens):
        # Cropping the sequence length to match the context length
        cropped_token_ids=token_ids[:, -context_length:]

        # Disabling gradients
        with torch.no_grad():
            # Making predictions
            logits=model(cropped_token_ids)

        # Extracting the embeddings for the last step
        logits=logits[:, -1, :]

        # Creating a condition based on top K
        if top_k is not None:
            # Extracting the top K logits
            top_logits, _ = torch.topk(input=logits, k=top_k)

            # Extracting the minimum value from the logits
            min_logit=top_logits[:, -1:]

            # Updating the logits
            logits=torch.where(condition=logits<min_logit, input=torch.tensor(data=float("-inf")).to(device=logits.device), other=logits)

        # Creating a condition based on the temperature scaling
        if temperature>0.0:
            # Dividing the logits by temperature
            logits=logits/temperature

            # Calculating the probabilities
            probas=torch.softmax(input=logits, dim=-1)

            # Extracting the next token ID
            next_token_id=torch.multinomial(input=probas, num_samples=1)
        else:
            # Extracting the next token ID with the highest probability
            next_token_id=torch.argmax(input=logits, dim=-1, keepdim=True)

        # Creating a condition based on the end of sequence token ID
        if next_token_id==eos_id:
            # Stopping text generation
            break
        
        # Concatenating the next token ID to the current token IDs
        token_ids=torch.cat(tensors=(token_ids, next_token_id), dim=1)

    # Returning the token IDs
    return token_ids

# Defining a function to download and load the weights
def download_and_load_gpt2(model_size, models_dir=None):
    # Mapping the model sizes to Hugging Face model IDs (models_dir is kept for book compatibility; unused)
    allowed_sizes={"124M": "openai-community/gpt2",
                   "355M": "openai-community/gpt2-medium",
                   "774M": "openai-community/gpt2-large",
                   "1558M": "openai-community/gpt2-xl"}

    # Asserting the model size to be supported
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {tuple(allowed_sizes)}")

    # Downloading the PyTorch model and converting its weights to NumPy
    hf_model=GPT2Model.from_pretrained(allowed_sizes[model_size])
    sd={k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

    # Extracting the hyperparameters from the Hugging Face configuration
    cfg=hf_model.config
    settings={"n_vocab": cfg.vocab_size, "n_ctx": cfg.n_positions,
              "n_embd": cfg.n_embd, "n_head": cfg.n_head, "n_layer": cfg.n_layer}

    # Mapping the root weights
    params={"wte": sd["wte.weight"], "wpe": sd["wpe.weight"],
            "g": sd["ln_f.weight"], "b": sd["ln_f.bias"], "blocks": []}

    # Mapping the layer blocks
    for i in range(settings["n_layer"]):
        p=f"h.{i}."
        params["blocks"].append({
            "attn": {"c_attn": {"w": sd[p+"attn.c_attn.weight"], "b": sd[p+"attn.c_attn.bias"]},
                     "c_proj": {"w": sd[p+"attn.c_proj.weight"], "b": sd[p+"attn.c_proj.bias"]}},
            "ln_1": {"g": sd[p+"ln_1.weight"], "b": sd[p+"ln_1.bias"]},
            "ln_2": {"g": sd[p+"ln_2.weight"], "b": sd[p+"ln_2.bias"]},
            "mlp": {"c_fc": {"w": sd[p+"mlp.c_fc.weight"], "b": sd[p+"mlp.c_fc.bias"]},
                    "c_proj": {"w": sd[p+"mlp.c_proj.weight"], "b": sd[p+"mlp.c_proj.bias"]}},
        })

    # Returning the settings and parameters
    return settings, params

# Defining a function to copy a NumPy array into a parameter after checking the shapes
def assign(left, right):
    # Asserting the shapes to match
    if left.shape!=right.shape:
        raise ValueError(f"Shape mismatch. Left: {tuple(left.shape)}, Right: {right.shape}")

    # Returning the new parameter
    return torch.nn.Parameter(torch.tensor(right, dtype=left.dtype))

# Defining a function to load the GPT-2 weights into the GPT model
def load_weights_into_gpt(gpt, params):
    # Loading the embedding weights
    gpt.pos_emb_layer.weight=assign(gpt.pos_emb_layer.weight, params["wpe"])
    gpt.tok_emb_layer.weight=assign(gpt.tok_emb_layer.weight, params["wte"])

    # Looping through the transformer blocks
    for block, p in zip(gpt.trf_blocks, params["blocks"]):
        # Loading the fused query, key, and value weights (Conv1D stores [in, out], Linear expects [out, in])
        block.mha.W_qkv.weight=assign(block.mha.W_qkv.weight, p["attn"]["c_attn"]["w"].T)
        block.mha.W_qkv.bias=assign(block.mha.W_qkv.bias, p["attn"]["c_attn"]["b"])

        # Loading the attention output projection weights
        block.mha.linear_proj.weight=assign(block.mha.linear_proj.weight, p["attn"]["c_proj"]["w"].T)
        block.mha.linear_proj.bias=assign(block.mha.linear_proj.bias, p["attn"]["c_proj"]["b"])

        # Loading the feed forward neural network weights
        block.ff.layers[0].weight=assign(block.ff.layers[0].weight, p["mlp"]["c_fc"]["w"].T)
        block.ff.layers[0].bias=assign(block.ff.layers[0].bias, p["mlp"]["c_fc"]["b"])
        block.ff.layers[2].weight=assign(block.ff.layers[2].weight, p["mlp"]["c_proj"]["w"].T)
        block.ff.layers[2].bias=assign(block.ff.layers[2].bias, p["mlp"]["c_proj"]["b"])

        # Loading the layer normalization weights
        block.layer_norm_mha.scale=assign(block.layer_norm_mha.scale, p["ln_1"]["g"])
        block.layer_norm_mha.shift=assign(block.layer_norm_mha.shift, p["ln_1"]["b"])
        block.layer_norm_ff.scale=assign(block.layer_norm_ff.scale, p["ln_2"]["g"])
        block.layer_norm_ff.shift=assign(block.layer_norm_ff.shift, p["ln_2"]["b"])

    # Loading the final layer normalization weights
    gpt.final_layer_norm.scale=assign(gpt.final_layer_norm.scale, params["g"])
    gpt.final_layer_norm.shift=assign(gpt.final_layer_norm.shift, params["b"])

    # Loading the output head weights (GPT-2 ties them to the token embeddings)
    gpt.lm_head.weight=assign(gpt.lm_head.weight, params["wte"])