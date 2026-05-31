# Key Value Cache for Inference

## 1. Need for KV Cache during Inference


During **inference**, the model generates text **autoregressively** — one token at a time. Each newly generated token is appended to the sequence, and the model runs another forward pass to predict the next one. This means token positions are revealed sequentially, not all at once:

```
Step 1: ["Hello"] → predict "I"
Step 2: ["Hello", "I"] → predict "am"
Step 3: ["Hello", "I", "am"] → predict "a"
...
```

This sequential nature creates a key inefficiency: at every step, the model recomputes the key and value vectors for **all previous tokens**, even though those vectors haven't changed since the last step. KV cache is the solution to this redundancy.

## 2. Why is KV cache not used during training?


The main purpose of using a KV cache during inference is to avoid recomputing keys and values for previous tokens every time a new token is generated. During standard transformer training, the entire input sequence is available at once. The model computes queries, keys, and values for all tokens in a single batched forward pass, while causal masking ensures that each token attends only to earlier positions. Unlike autoregressive inference, there is no repeated recomputation of past tokens' K and V representations across decoding steps. Since training already computes each token's K and V exactly once per forward pass, the introduction of a KV increases GPU vRAM utilization without improving the throughput.

Training also requires backpropagation through the full computation graph. Intermediate activations, including K and V tensors, must be retained so gradients can be computed. These activations are already stored as part of the training graph, so maintaining a separate inference-style KV cache would typically add complexity without reducing computation or memory usage.

Therefore, conventional full-sequence training generally does not use KV caching. This is not because caching is fundamentally incompatible with gradient computation, but because the inference-time motivation for KV caching—avoiding repeated recomputation of past tokens—is absent during standard training. Specialized training methods may still use cached states, but they require carefully defined gradient handling and differ from the KV caches used during autoregressive inference.

## 3. Key considerations associated with the use of KV Cache

- **Substantial throughput improvement.** Without a KV cache, the key and value vectors for a token may be recomputed $O(N)$ times during generation of a sequence of length $N$. With KV caching, each token's key and value vectors are computed once and reused thereafter. Consequently, the time complexity of K/V projection is optimized from $O(N^2)$ to $O(N)$. The overall time complexity of autoregressively generating a sequence of length $N$ is still $O(N^2)$ because each new token attends to all previously generated tokens during the computation of the attention scores.

- **Increased memory usage.** The cache stores key and value tensors for every layer, every head, and every token in the context. For large models or long contexts, this can be significant. Memory grows as: `2 × n_layers × n_kv_heads × seq_len × head_dim × dtype_bytes`. The value of `dtype_bytes` is 2 for FP16 precision. Taking the [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) model as a representative example, the parameter values are `n_layers = 64`, `n_kv_heads = 8` and `head_dim` = 128. In this case, the cache size for a context length of 131,072 tokens is

$$
\text{KV cache memory} = \frac{2 \times 64 \times 8 \times 131,072 \times 128 \times 2}{10^9} \approx 34.4 GB
$$

This is comparable to the memory required for the model weights, which is around 64 GB. The use of Grouped Query Attention (GQA) limits the KV cache size. The [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) model has 64 attention heads but only 8 key-value heads. Without GQA, the KV cache size would be 8x larger, since the key and value vectors are no longer shared across attention heads.

- **Decoding becomes increasingly memory-bound.** Auto-regressive generation of each new token requires reading of model weights and the key and value vectors of all previous tokens from vRAM. As explained [here](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) and [here](https://medium.com/@arjunravi726/why-llm-inference-is-memory-bound-not-compute-bound-ba59c48739e0), the ratio of FLOPs to number of bytes transferred to and from memory is relatively low for this operation, making it memory-bound.

- **Specialized techniques are required to optimize KV cache management.** As explained in [this paper](https://arxiv.org/pdf/2603.20397), key classes of techniques used to optimize KV cache memory utilization include the following:

1. Cache eviction methods that selectively discard less critical tokens.
2. Compression and reconstruction techniques.
3. Hybrid memory solutions that make use of multi-tier storage.
4. State-of-the-art attention algorithms with custom methods for context processing.
5. Hybrid strategies that incorporate multiple optimization techniques.


## 4. How does KV cache work?

Reference: https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms

### The core idea

In the attention mechanism, keys (K) and values (V) for any token depend only on that token's embedding — not on any future tokens. This means that once K and V are computed for a token, they remain valid for all future generation steps. Only the query (Q) changes at each step, because the query represents "what the current token is looking for."

KV cache exploits this by saving K and V at the time they are first computed, and reusing them in all subsequent steps.

### Without KV cache

At step *n*, the full attention computation for a single seqeunce is:
1. Project all *n* tokens through W_K, W_Q, W_V → three matrices of shape `(n, d_out)`
2. Compute attention scores: Q @ K^T, shape `(n, n)`
3. Apply causal mask, softmax, multiply by V

The FLOPs for step 1 is proportional to `n`.

### With KV cache

At step *n*:
1. Load cached K and V for positions 0 to *n-1* — no computation needed
2. Compute K and V only for the new token — O(1) projection
3. Append new K, V to cache
4. Compute Q only for the new token — O(1) projection
5. Compute attention: Q_new @ K_all^T — only one query row, so the score matrix is shape `(1, n)` instead of `(n, n)`

The use of the KV cache optimizes the time complexity of Q, K and V projections from $O(n)$ to $O(1)$ at each decoding step.

See this [page](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) for more details.

## 5. Exercise

Modify the `gpt_ch04.py` script in this [repository](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/03_kv-cache/gpt_ch04.py) to include KV cache. Compare your results for inference throughput and memory utilization with the original script as well as the `gpt_with_kv_cache.py` script in the same repository. 
