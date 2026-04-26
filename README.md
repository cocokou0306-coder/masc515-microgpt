# masc515-microgpt
Implementation of GELU, LoRA, RoPE and MoE based on microGPT


## 1. GELU (Gaussian Error Linear Unit)
    
    ### What is GELU?
    
    GELU is a smooth activation function widely used in Transformer models such as GPT. Unlike ReLU, which sets all negative values to zero, GELU allows small negative values to pass through in a smooth manner.
    
    Mathematically:
    
    GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    
    ---
    
    ### Why replace ReLU?
    
    The original model uses ReLU:
    
    - ReLU(x) = max(0, x)
    - It removes all negative values
    - This may lose useful information
    
    GELU improves this by:
    
    - Providing smoother gradients
    - Preserving more information
    - Improving optimization stability
    
    ---
    
    ### Implementation
    
    We modified the activation function in the MLP layer:
    
    - Original:
      x = [xi.relu() for xi in x]
    
    - Modified:
      x = [xi.gelu() for xi in x]
    
    We also implemented a `gelu()` function inside the `Value` class.
    
    ---
    
    ### Observed Changes
    
    After replacing ReLU with GELU:
    
    - The model still trains successfully
    - The loss decreased slightly (e.g., from ~2.64 to ~2.51)
    - The generated outputs (names) changed
    
    This confirms that GELU was successfully integrated and affects model behavior.
    
    ---

## 2. RoPE (Rotary Positional Embedding)

    ### What is RoPE?
    
    RoPE is a positional encoding method that uses rotation to encode position information instead of simple addition. It enables the model to better capture relative positional relationships.
    
    ---
    
    ### Why replace positional embedding?
    
    The original model uses:
    
    x = tok_emb + pos_emb
    
    Limitations:
    
    - Only captures absolute positions
    - Limited generalization
    
    RoPE improves this by:
    
    - Encoding relative positions
    - Using sine/cosine rotation
    - Providing better structural representation
    
    ---
    
    ### Implementation
    
    We modified how positional information is applied:
    
    - Removed positional embedding addition
    - Introduced a function:
      apply_rope(x, pos)
    
    - Replaced:
    
      x = tok_emb + pos_emb
    
      with:
    
      x = apply_rope(tok_emb, pos_id)
    
    ---
    
    ### Observed Changes
    
    After applying RoPE:
    
    - The model still trains normally
    - The loss changed (e.g., from ~2.51 to ~2.62)
    - Generated outputs are different
    
    This indicates that the positional encoding mechanism has been successfully modified.
    
    ---
    
    ## Conclusion
    
    Both GELU and RoPE are key improvements used in modern Transformer architectures.
    
    - GELU improves activation behavior and gradient flow
    - RoPE enhances positional encoding by using rotation
    
    The successful implementation and observable differences in model outputs confirm that both modifications are correctly integrated into the microGPT model.
